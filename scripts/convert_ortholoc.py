"""Конвертер OrthoLoC → канонический формат обучающей пары (DATASET_SPEC_FINETUNE).

Из каждого сэмпла OrthoLoC делаются пары двух режимов:

- ``asis`` — наклонный кадр как есть против окна подложки: warp выводится из
  ``point_map`` (кадр-пиксель → мировая XY → пиксель DOP);
- ``rect`` — кадр орто-ректифицируется на сетку DOP точной проекцией
  XYZ-точек ``dsm`` в камеру кадра (``K``/``extrinsics``); пара становится
  «орто ↔ орто», warp — целочисленный сдвиг сеток.

Окно подложки B сэмплируется со случайным смещением/размером под целевую
ко-видимость 0.3–0.85 (§2.1 спеки) — детерминированно от имени сэмпла.

Формат записи ``pair_<sample>_<mode>.npz`` (§2 спеки, байтовый уровень —
JPEG-in-npz ради объёма, см. §2 спеки после правки):

    image_a_jpeg  uint8 1-D — JPEG q95 стороны A (RGB)
    image_b_jpeg  uint8 1-D — JPEG q95 стороны B (RGB)
    warp_ab       H_a×W_a×2 float16 — (x, y) в ПИКСЕЛЯХ B; NaN вне маски
    mask_ab       H_a×W_a uint8 — 1 = соответствие валидно
    meta          json-строка (§2.3: сцена, наклон, GSD, covis, провенанс)
    pinhole       json-строка или "null" — эмуляция пути B (только rect)

Чтение — :func:`load_pair` из этого же модуля.

    python scripts/convert_ortholoc.py --splits test_inPlace --modes asis,rect \\
        --out data/pairs_ortholoc
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zlib
from datetime import date
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: Минимальная ко-видимость пары и размеры окна B (§2.1 спеки). Верхней
#: границы нет: если отпечаток A меньше окна, covis = 1.0 — это нормальная
#: боевая ситуация (окно подложки шире кадра), а не вырождение.
COVIS_MIN = 0.30
B_SIZES = (1024, 896, 768, 640)
JPEG_QUALITY = 95


# --- чистая геометрия (тестируется без файлов) --------------------------------

def warp_from_pointmap(pm: np.ndarray, scale, wb: int, hb: int):
    """point_map (H×W×3, мировые XYZ) → координаты пикселей DOP (float64).

    Формула проверена бенчмарком (`bench_ortholoc.py`): DOP ортографичен,
    центр мировых координат — центр растра, пиксель-центровая конвенция.
    """
    sx, sy = float(scale[0]), float(scale[1])
    gt_x = pm[..., 0] / sx + (wb - 1) / 2.0
    gt_y = pm[..., 1] / sy + (hb - 1) / 2.0
    return gt_x, gt_y


def window_covis(gt_x, gt_y, finite, win) -> float:
    """Доля пикселей A, чьё соответствие попадает в окно B (x0, y0, w, h)."""
    x0, y0, w, h = win
    ok = finite & (gt_x >= x0) & (gt_x < x0 + w) & (gt_y >= y0) & (gt_y < y0 + h)
    return float(ok.mean())


def choose_window(gt_x, gt_y, full_w, full_h, rng,
                  sizes=B_SIZES, covis_min=COVIS_MIN, tries=12):
    """Окно B со случайным смещением/размером; детерминированно при данном rng.

    Штрафуется только низкая ко-видимость (< covis_min): первая проба с
    ``covis ≥ covis_min`` принимается (смещение окна при этом уже случайно —
    источник разнообразия сдвигов). Если все пробы ниже порога — берётся
    лучшая (честный fallback, пара не выбрасывается, порог решает фильтр
    вызывающего кода).
    """
    finite = np.isfinite(gt_x) & np.isfinite(gt_y)
    best = None
    for _ in range(tries):
        s = int(rng.choice(sizes))
        s = min(s, full_w, full_h)
        x0 = int(rng.integers(0, full_w - s + 1))
        y0 = int(rng.integers(0, full_h - s + 1))
        win = (x0, y0, s, s)
        c = window_covis(gt_x, gt_y, finite, win)
        if best is None or c > best[1]:
            best = (win, c)
        if c >= covis_min:
            return win, c
    return best


def warp_to_window(gt_x, gt_y, win):
    """Перевод warp в координаты окна B + маска. Возвращает (float16 H×W×2, uint8)."""
    x0, y0, w, h = win
    wx = gt_x - x0
    wy = gt_y - y0
    mask = (np.isfinite(wx) & np.isfinite(wy)
            & (wx >= 0) & (wx < w) & (wy >= 0) & (wy < h))
    warp = np.stack([wx, wy], axis=-1).astype(np.float16)
    warp[~mask] = np.nan
    return warp, mask.astype(np.uint8)


def project_to_camera(pts_xyz: np.ndarray, K: np.ndarray, ext: np.ndarray):
    """Мировые точки (…×3) → пиксели камеры (map_x, map_y, глубина z)."""
    R, t = ext[:, :3], ext[:, 3]
    pc = pts_xyz @ R.T + t
    z = pc[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * pc[..., 0] / z + K[0, 2]
        v = K[1, 1] * pc[..., 1] / z + K[1, 2]
    return u, v, z


def median_gsd(pm: np.ndarray) -> float:
    """Медианный наземный шаг соседних пикселей кадра — эффективный GSD A."""
    dx = np.diff(pm[..., :2], axis=1)
    step = np.sqrt((dx ** 2).sum(-1))
    return float(np.nanmedian(step))


def vegetation_weight(rgb: np.ndarray) -> np.ndarray:
    """Мягкая маска растительности 0..1 по индексу ExG = 2G − R − B.

    Без семантики: чистая колориметрия. Порог мягкий (сигмоида) и маска
    размыта — чтобы перекраска не давала резких контуров, по которым матчер
    мог бы «читерить».
    """
    f = rgb.astype(np.float32)
    exg = 2 * f[..., 1] - f[..., 0] - f[..., 2]
    w = 1.0 / (1.0 + np.exp(-(exg - 24.0) / 12.0))
    return cv2.blur(w, (7, 7))


def pseudo_season(rgb: np.ndarray, season: str, rng) -> np.ndarray:
    """Псевдосезонная фотометрия. Это аппроксимация appearance gap, НЕ сезон:
    нет снежной геометрии, теней и листопада — честная роль этой аугментации
    описана в спеке §2.3а; настоящий кросс-сезон остаётся за obs://orto.

    - ``autumn``: зелень → жёлто-оранжевая (hue), чуть темнее;
    - ``spring``: зелень выцветает в серо-бурую, общая приглушённость;
    - ``winter``: сильная десатурация и высветление всего кадра (снежный
      колорит), зелень глушится почти в серое, лёгкий холодный сдвиг.
    """
    k = float(rng.uniform(0.8, 1.2))                 # сила эффекта — джиттер
    veg = vegetation_weight(rgb)[..., None]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if season == "autumn":
        target_h = 16.0                              # оранжево-коричневый
        h_new = h + (target_h - h) * np.clip(0.85 * k, 0, 1)
        s_new = np.clip(s * (1.0 + 0.1 * k), 0, 255)
        v_new = np.clip(v * (1.0 - 0.06 * k), 0, 255)
    elif season == "spring":
        target_h = 32.0                              # блёкло-жёлто-зелёный
        h_new = h + (target_h - h) * np.clip(0.6 * k, 0, 1)
        s_new = s * (1.0 - 0.55 * min(k, 1.0))
        v_new = np.clip(v * (1.0 - 0.10 * k), 0, 255)
    elif season == "winter":
        h_new = h + (105.0 - h) * 0.10 * k           # лёгкий холодный сдвиг
        s_new = s * (1.0 - 0.92 * min(k, 1.0))       # зелень гасится почти в серое
        v_new = np.clip(v * (1.0 - 0.45 * k) + 255 * 0.52 * k, 0, 255)
    else:
        raise ValueError(f"неизвестный сезон {season!r}")
    w = veg[..., 0]
    if season == "winter":                           # зима красит весь кадр,
        w = np.clip(w * 0.4 + 0.6, 0, 1)             # зелень — сильнее прочего
    hsv2 = np.stack([
        (h * (1 - w) + h_new * w) % 180.0,
        s * (1 - w) + s_new * w,
        v * (1 - w) + v_new * w,
    ], axis=-1).astype(np.uint8)
    return cv2.cvtColor(hsv2, cv2.COLOR_HSV2RGB)


def jpeg_bytes(rgb: np.ndarray) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.reshape(-1)


def balance_per_scene(files: list[Path], cap: int, split: str) -> list[Path]:
    """Не больше ``cap`` сэмплов на сцену: детерминированная выборка,
    стратифицированная по варианту (R/xDOP), чтобы балансировка не съедала
    кросс-источниковую долю. Никаких фильтров по наклону/GSD — оси
    разнообразия остаются в выборке."""
    per: dict[str, dict[str, list[Path]]] = {}
    for f in files:
        scene = f.stem.split("_")[0]
        variant = "xDOP" if "_xDOP" in f.stem else "R"
        per.setdefault(scene, {}).setdefault(variant, []).append(f)
    out: list[Path] = []
    for scene in sorted(per):
        groups = per[scene]
        total = sum(len(v) for v in groups.values())
        take_all = total <= cap
        rng = np.random.default_rng(zlib.crc32(f"balance:{split}:{scene}".encode()))
        for variant in sorted(groups):
            grp = sorted(groups[variant])
            n = len(grp) if take_all else max(1, round(cap * len(grp) / total))
            idx = rng.permutation(len(grp))[:n]
            out += [grp[i] for i in sorted(idx)]
    return sorted(out)


def load_pair(path: Path) -> dict:
    """Чтение канонической пары: декодирует JPEG, парсит мету."""
    d = np.load(path, allow_pickle=False)
    out = {
        "image_a": cv2.cvtColor(cv2.imdecode(d["image_a_jpeg"], cv2.IMREAD_COLOR),
                                cv2.COLOR_BGR2RGB),
        "image_b": cv2.cvtColor(cv2.imdecode(d["image_b_jpeg"], cv2.IMREAD_COLOR),
                                cv2.COLOR_BGR2RGB),
        "warp_ab": d["warp_ab"].astype(np.float32),
        "mask_ab": d["mask_ab"],
        "meta": json.loads(str(d["meta"])),
    }
    ph = str(d["pinhole"])
    out["pinhole"] = None if ph == "null" else json.loads(ph)
    return out


# --- режимы конвертации --------------------------------------------------------

def convert_asis(d, rng):
    """Наклонный кадр как есть против окна подложки."""
    q = d["image_query"]
    dop = d["image_dop"]
    pm = d["point_map"].astype(np.float64)
    hb, wb = dop.shape[:2]
    gt_x, gt_y = warp_from_pointmap(pm, np.asarray(d["scale"], float), wb, hb)
    win, covis = choose_window(gt_x, gt_y, wb, hb, rng)
    x0, y0, w, h = win
    warp, mask = warp_to_window(gt_x, gt_y, win)
    b = dop[y0:y0 + h, x0:x0 + w]
    gsd_b = float(abs(d["scale"][0]))
    return q, b, warp, mask, dict(
        gsd_a=round(median_gsd(pm), 4), gsd_b=round(gsd_b, 4),
        covis_frac=round(covis, 4), b_window=[x0, y0, w, h]), None


def convert_rect(d, rng, min_valid=0.25):
    """Орто-ректификация кадра на сетку DOP: точная проекция точек dsm в камеру."""
    q = d["image_query"]
    dop = d["image_dop"]
    pts = d["dsm"].astype(np.float64)                 # мировые XYZ сетки DOP
    K = d["intrinsics"].astype(np.float64)
    ext = d["extrinsics"].astype(np.float64)
    hq, wq = q.shape[:2]
    hb, wb = dop.shape[:2]

    u, v, z = project_to_camera(pts, K, ext)
    valid = (z > 0) & (u >= 0) & (u <= wq - 1) & (v >= 0) & (v <= hq - 1)
    if valid.mean() < min_valid:
        return None                                   # кадр почти не кроет сетку
    a_full = cv2.remap(q, u.astype(np.float32), v.astype(np.float32),
                       cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    a_full[~valid] = 0

    ys, xs = np.nonzero(valid)
    ax0, ay0 = int(xs.min()), int(ys.min())
    ax1, ay1 = int(xs.max()) + 1, int(ys.max()) + 1
    a = a_full[ay0:ay1, ax0:ax1]
    a_valid = valid[ay0:ay1, ax0:ax1]

    # warp на общей сетке: пиксель A (x, y) → орто (x+ax0, y+ay0)
    ha, wa = a.shape[:2]
    gx, gy = np.meshgrid(np.arange(wa, dtype=np.float64) + ax0,
                         np.arange(ha, dtype=np.float64) + ay0)
    gx[~a_valid] = np.nan
    gy[~a_valid] = np.nan
    win, covis = choose_window(gx, gy, wb, hb, rng)
    x0, y0, w, h = win
    warp, mask = warp_to_window(gx, gy, win)
    b = dop[y0:y0 + h, x0:x0 + w]

    gsd_b = float(abs(d["scale"][0]))
    h_virt = 40000.0                                   # §1.2а: err ≈ (Δh/H)·r
    pinhole = dict(
        model="fronto_parallel", note="точно при depth=H_virt−DSM (см. спеку §1.2а); "
        "константная глубина — приближение (Δh/H_virt)·r px",
        H_virt=h_virt, f_px=h_virt / gsd_b, depth_const=h_virt,
        t_ab_px=[ax0 - x0, ay0 - y0],                  # сдвиг сеток A→B
    )
    return a, b, warp, mask, dict(
        gsd_a=round(gsd_b, 4), gsd_b=round(gsd_b, 4),
        covis_frac=round(covis, 4), b_window=[x0, y0, w, h],
        a_window=[ax0, ay0, ax1 - ax0, ay1 - ay0],
        a_valid_frac=round(float(a_valid.mean()), 4)), pinhole


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="data/OrthoLoC")
    parser.add_argument("--splits", default="test_inPlace")
    parser.add_argument("--modes", default="asis,rect")
    parser.add_argument("--scenes", default="", help="фильтр сцен через запятую")
    parser.add_argument("--max-tilt", type=float, default=0.0,
                        help="брать только сэмплы с наклоном ≤ N° (0 = все)")
    parser.add_argument("--per-scene", type=int, default=0,
                        help="балансировка: не больше N сэмплов на сцену "
                             "(детерминированная выборка, стратифицированная "
                             "по R/xDOP; 0 = все)")
    parser.add_argument("--pseudo-seasons", default="",
                        help="через запятую из winter,autumn,spring: на долю "
                             "--season-frac пар добавляется запись с "
                             "псевдосезонной фотометрией стороны A")
    parser.add_argument("--season-frac", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=10,
                                cwd=Path(__file__).parents[1]).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    scenes = {s.strip() for s in args.scenes.split(",") if s.strip()}
    out_root = Path(args.out)
    manifest = out_root / "manifest.csv"
    out_root.mkdir(parents=True, exist_ok=True)
    new_manifest = not manifest.exists()
    mf = manifest.open("a", encoding="utf-8")
    if new_manifest:
        mf.write("pair,split,scene,mode,pair_kind,tilt_deg,covis_frac,bytes\n")

    seasons = [s.strip() for s in args.pseudo_seasons.split(",") if s.strip()]
    n_done = n_skip = 0
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        files = sorted((Path(args.root) / split).glob("*.npz"))
        if args.per_scene:
            files = balance_per_scene(files, args.per_scene, split)
        if args.limit:
            files = files[: args.limit]
        out_dir = out_root / split
        out_dir.mkdir(exist_ok=True)
        for f in files:
            scene = f.stem.split("_")[0]
            if scenes and scene not in scenes:
                continue
            d = np.load(f)
            ext = d["extrinsics"]
            tilt = float(np.degrees(np.arccos(np.clip(-ext[2, 2], -1, 1))))
            if args.max_tilt and tilt > args.max_tilt:
                continue
            for mode in modes:
                dst = out_dir / f"pair_{f.stem}_{mode}.npz"
                if dst.exists():
                    continue
                rng = np.random.default_rng(zlib.crc32(f"{f.stem}:{mode}".encode()))
                res = (convert_asis(d, rng) if mode == "asis"
                       else convert_rect(d, rng))
                if res is None:
                    n_skip += 1
                    continue
                a, b, warp, mask, extra, pinhole = res
                if mask.mean() < 0.05:                # соответствий почти нет
                    n_skip += 1
                    continue
                meta = dict(
                    source="ortholoc", split=split, scene=scene, sample=f.stem,
                    pair_kind="xDOP" if "_xDOP" in f.stem else "R",
                    tilt_deg=round(tilt, 2), rectified=(mode == "rect"),
                    season_a=None, season_b=None, date_a=None, date_b=None,
                    scale_ratio=round(extra["gsd_a"] / extra["gsd_b"], 3),
                    gen_commit=commit, gen_date=date.today().isoformat(),
                    **extra)
                b_jpeg = jpeg_bytes(b)
                pinhole_s = np.str_(json.dumps(pinhole, ensure_ascii=False)
                                    if pinhole else "null")
                np.savez_compressed(
                    dst,
                    image_a_jpeg=jpeg_bytes(a), image_b_jpeg=b_jpeg,
                    warp_ab=warp, mask_ab=mask,
                    meta=np.str_(json.dumps(meta, ensure_ascii=False)),
                    pinhole=pinhole_s)
                mf.write(f"{dst.name},{split},{scene},{mode},{meta['pair_kind']},"
                         f"{meta['tilt_deg']},{meta['covis_frac']},{dst.stat().st_size}\n")
                n_done += 1
                # псевдосезонная запись: та же геометрия, перекрашенная сторона A
                if seasons and rng.random() < args.season_frac:
                    season = str(rng.choice(seasons))
                    dst_s = out_dir / f"pair_{f.stem}_{mode}_{season[0]}.npz"
                    if not dst_s.exists():
                        a_s = pseudo_season(a, season, rng)
                        meta_s = dict(meta, season_a=f"pseudo_{season}")
                        np.savez_compressed(
                            dst_s,
                            image_a_jpeg=jpeg_bytes(a_s), image_b_jpeg=b_jpeg,
                            warp_ab=warp, mask_ab=mask,
                            meta=np.str_(json.dumps(meta_s, ensure_ascii=False)),
                            pinhole=pinhole_s)
                        mf.write(f"{dst_s.name},{split},{scene},{mode},"
                                 f"{meta['pair_kind']},{meta['tilt_deg']},"
                                 f"{meta['covis_frac']},{dst_s.stat().st_size}\n")
                        n_done += 1
                if n_done % 200 == 0:
                    mf.flush()
                    print(f"{n_done} пар...", flush=True)
    mf.close()
    print(f"готово: {n_done} пар, пропущено {n_skip} → {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
