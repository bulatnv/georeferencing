"""Этап Г (§5 задания): генерация пар «вид виртуальной камеры ↔ подложка».

Сторона A — кадр виртуального борта, отрендеренный из ортофотоплана лучами
камеры на плоскость z = 0; сторона B — north-up кроп подложки. Разметка —
плотный warp A→B с маской ко-видимости, формат записи — канонический
(DATASET_SPEC_FINETUNE §2), тот же, что у конвертера OrthoLoC.

Камера повторяет боевую: 1024×576, f_px 735 (FOV 69.7°) — не «на глаз»,
иначе кадр видит вчетверо меньше земли, чем реальный борт (§5 задания).

    python open_orto/scripts/generate.py --raster <файл>.tif \\
        --shift-field open_orto/work/shift/shift_field_<stem>.npz \\
        --count 200 --out open_orto/dataset
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import zlib
from datetime import date
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate_check import overview_mask  # noqa: E402
from geom import (  # noqa: E402
    camera_rotation,
    footprint_corners,
    fov_deg,
    intrinsics,
    nadir_gsd,
    pixels_to_ground,
    rect_overlap_frac,
    valid_mask,
)
from rasters import NEUTRAL_GRAY, BasemapSource, Grid, OrthoSource  # noqa: E402

# Параметры генерации — редакция 2 плана орто↔орто (§5 задания), перенесены как есть.
FRAME_W, FRAME_H, F_PX = 1024, 576, 735.0
#: Высоты съёмки. Редакция 4 (28.08): понижены с 250–400 м. Кадр стал ближе
#: к земле — GSD_A 0.24–0.41 м вместо 0.34–0.54, след кадра 244–418 м вместо
#: 348–558. Меньший наземный контекст — измеренно трудный для матчеров режим
#: (Э2.2 трека F), и именно он ближе к нашим низковысотным боевым кадрам.
HEIGHT_RANGE = (175.0, 300.0)
#: Курс кадра — произвольный (ортоплан можно вращать как угодно), а курс
#: кропа подложки отличается от него не более чем на DELTA_YAW_MAX: именно
#: столько остаточной невязки курса допускает боевой тракт после
#: предповорота. Раньше подложка была жёстко north-up, и разброс взаимного
#: поворота ограничивался диапазоном самого кадра.
YAW_RANGE = (0.0, 360.0)
DELTA_YAW_MAX = 25.0
TILT_RANGE = (0.0, 10.0)
SCALE_RANGE = (0.85, 1.20)         # GSD_A / GSD_B
B_PX_RANGE = (768, 2048)
INSIDE_QUOTA = 0.65
INSIDE_WIDTH_FRAC = (0.50, 0.85)   # доля ширины следа A в кропе B
PARTIAL_OVERLAP = (0.60, 0.90)
SAME_SOURCE_QUOTA = 0.15           # контрольная ось (§5 «Контрольная ось»)
MIN_VALID_A = 0.90
MIN_VALID_B = 0.85
ANCHOR_STEP_M = 150.0
EROSION_M = 150.0


class ShiftField:
    """Поле сдвигов привязки с билинейной выборкой и фолбэком на константу."""

    def __init__(self, path: str | Path | None):
        self.pts = None
        self.vec = None
        self.global_dxy = (0.0, 0.0)
        self.source = "none"
        if path is None:
            return
        d = np.load(path, allow_pickle=False)
        self.pts = np.stack([d["x"], d["y"]], axis=-1)
        self.vec = np.stack([d["dx"], d["dy"]], axis=-1)
        self.global_dxy = (float(d["global_dx"]), float(d["global_dy"]))
        self.source = Path(path).name

    def at(self, gx: float, gy: float, radius_m: float = 700.0):
        """Компенсация в точке: (dx, dy, источник). Взвешенная по расстоянию
        медиана ближних узлов, иначе — глобальная константа (§5.7 задания)."""
        if self.pts is None or len(self.pts) == 0:
            return self.global_dxy[0], self.global_dxy[1], "global"
        d = np.hypot(self.pts[:, 0] - gx, self.pts[:, 1] - gy)
        sel = d <= radius_m
        if sel.sum() >= 3:
            w = 1.0 / np.maximum(d[sel], 1.0)
            dx = float(np.average(self.vec[sel, 0], weights=w))
            dy = float(np.average(self.vec[sel, 1], weights=w))
            return dx, dy, "field"
        return self.global_dxy[0], self.global_dxy[1], "global"


def build_anchors(ortho: OrthoSource, step_m: float = ANCHOR_STEP_M):
    """Сетка якорей по рабочей зоне (валидные данные с эрозией)."""
    mask, _ = overview_mask(ortho, width=1500)
    b = ortho.bounds
    sx = (b.right - b.left) / mask.shape[1]
    sy = (b.top - b.bottom) / mask.shape[0]
    er = max(1, int(EROSION_M / sx))
    core = cv2.erode(mask.astype(np.uint8), np.ones((er, er), np.uint8)).astype(bool)
    out = []
    for gy in np.arange(b.bottom + EROSION_M, b.top - EROSION_M, step_m):
        for gx in np.arange(b.left + EROSION_M, b.right - EROSION_M, step_m):
            j = int((gx - b.left) / sx)
            i = int((b.top - gy) / sy)
            if 0 <= i < core.shape[0] and 0 <= j < core.shape[1] and core[i, j]:
                out.append((float(gx), float(gy)))
    return out


def pull_anchor_inside(ortho: OrthoSource, gx: float, gy: float, span_m: float):
    """Подтяжка якоря вглубь съёмки (§5.2 задания): если в окне мало данных,
    пробуем смещения на 30 и 60% его размера по восьми направлениям."""
    best = None
    probe_px = 256
    for frac in (0.0, 0.3, 0.6):
        dirs = [(0.0, 0.0)] if frac == 0.0 else [
            (math.cos(a) * frac * span_m, math.sin(a) * frac * span_m)
            for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
        for dx, dy in dirs:
            grid = Grid(x=gx + dx, y=gy + dy, size_px=probe_px, gsd=span_m / probe_px)
            _, val = ortho.read_grid(grid)
            cov = float(val.mean())
            if best is None or cov > best[2]:
                best = (gx + dx, gy + dy, cov)
        if best and best[2] >= 0.98:
            break
    return best


def render_frame(ortho: OrthoSource, cam_xy, height_m: float, R, K):
    """Кадр A: луч пикселя → плоскость z=0 → выборка из ортоплана."""
    px, py = np.meshgrid(np.arange(FRAME_W, dtype=np.float64),
                         np.arange(FRAME_H, dtype=np.float64))
    gx, gy = pixels_to_ground(px, py, K, R, cam_xy, height_m)
    finite = np.isfinite(gx) & np.isfinite(gy)
    gxf = np.where(finite, gx, ortho.bounds.left - 1e6)
    gyf = np.where(finite, gy, ortho.bounds.top + 1e6)
    sx, sy = ortho.world_to_pixel(gxf, gyf)

    # окно растра под запрос, читаем в разрешении не мельче нужного
    inside = finite & (sx >= 0) & (sx < ortho.ds.width) & (sy >= 0) & (sy < ortho.ds.height)
    if inside.sum() < 100:
        return None, None, None, None
    x0 = max(0, int(np.floor(sx[inside].min())) - 4)
    y0 = max(0, int(np.floor(sy[inside].min())) - 4)
    x1 = min(ortho.ds.width, int(np.ceil(sx[inside].max())) + 4)
    y1 = min(ortho.ds.height, int(np.ceil(sy[inside].max())) + 4)
    from rasterio.windows import Window
    from rasterio.enums import Resampling
    win = Window(x0, y0, x1 - x0, y1 - y0)
    gsd_a = nadir_gsd(height_m, F_PX)
    scale = max(1.0, gsd_a / ortho.res_x)
    out_w = max(1, int(round(win.width / scale)))
    out_h = max(1, int(round(win.height / scale)))
    arr = ortho.ds.read(out_shape=(ortho.ds.count, out_h, out_w), window=win,
                        resampling=Resampling.average if scale > 1.5 else Resampling.bilinear)
    src = np.transpose(arr[:3], (1, 2, 0))
    map_x = ((sx - x0) * out_w / win.width).astype(np.float32)
    map_y = ((sy - y0) * out_h / win.height).astype(np.float32)
    rgb = cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    ok = (inside & (map_x >= 0) & (map_x <= out_w - 1)
          & (map_y >= 0) & (map_y <= out_h - 1))
    return rgb, valid_mask(rgb) & ok, gx, gy


def paint_invalid(rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Невалидное — нейтральный серый (§5.10): иначе сеть цепляется за
    красные полосы и чёрные дыры как за признак, которого в подложке нет."""
    out = rgb.copy()
    out[~valid] = NEUTRAL_GRAY
    return out


def jpeg_bytes(rgb: np.ndarray, q: int = 95) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.reshape(-1)


def plan_sample(rng, layout: str):
    """Случайный план сэмпла: высота, углы, масштаб, размер кропа B."""
    height = float(rng.uniform(*HEIGHT_RANGE))
    yaw = float(rng.uniform(*YAW_RANGE))
    delta_yaw = float(rng.uniform(-DELTA_YAW_MAX, DELTA_YAW_MAX))
    tilt = float(rng.uniform(*TILT_RANGE))
    tilt_az = float(rng.uniform(0.0, 360.0))
    scale = float(rng.uniform(*SCALE_RANGE))     # GSD_A / GSD_B
    gsd_a = nadir_gsd(height, F_PX)
    gsd_b = gsd_a / scale
    if layout == "inside":
        wfrac = float(rng.uniform(*INSIDE_WIDTH_FRAC))
        b_px = int(np.clip(round(FRAME_W * scale / wfrac), *B_PX_RANGE))
    else:
        b_px = int(rng.integers(B_PX_RANGE[0], B_PX_RANGE[1] + 1))
    return dict(height=height, yaw=yaw, delta_yaw=delta_yaw,
                yaw_b=(yaw + delta_yaw) % 360.0, tilt=tilt, tilt_az=tilt_az,
                scale=scale, gsd_a=gsd_a, gsd_b=gsd_b, b_px=b_px, layout=layout)


def footprint_overlap(K, R, plan, cam_xy, grid_b):
    """Доля следа кадра, попавшая в кроп B. Считается **в системе сетки B**:
    после поворота подложки её кроп — не axis-aligned прямоугольник в мире,
    и сравнивать габаритами было бы неверно."""
    poly = footprint_corners(FRAME_W, FRAME_H, K, R, cam_xy, plan["height"])
    if np.isnan(poly).any():
        return None
    px, py = grid_b.pixel_from_world(poly[:, 0], poly[:, 1])
    n = grid_b.size_px - 1
    return rect_overlap_frac(np.stack([px, py], axis=-1), (0.0, 0.0, n, n))


def place_camera(K, R, plan, grid_b, rng):
    """Позиция камеры под заданную компоновку: (cx, cy) либо None.

    Центр следа смещён наклоном на ``H·tan(tilt)`` в сторону азимута —
    позиция камеры сдвигается назад на эту величину, иначе при наклоне часть
    кадров вылезает за край кропа B (§5.4 задания; баг, пойманный V-тестом).
    """
    ax, ay = grid_b.x, grid_b.y
    span = grid_b.size_px * grid_b.gsd
    off = plan["height"] * math.tan(math.radians(plan["tilt"]))
    off_x = off * math.sin(math.radians(plan["tilt_az"]))
    off_y = off * math.cos(math.radians(plan["tilt_az"]))

    if plan["layout"] == "inside":
        half_foot = FRAME_W / 2 * plan["gsd_a"] * 1.25
        room = max(0.0, span / 2 - half_foot)
        for _ in range(16):
            cx = ax - off_x + float(rng.uniform(-room, room))
            cy = ay - off_y + float(rng.uniform(-room, room))
            frac = footprint_overlap(K, R, plan, (cx, cy), grid_b)
            if frac is not None and frac >= 0.9999:
                return cx, cy
        return None

    ang = float(rng.uniform(0, 2 * math.pi))
    lo, hi = 0.0, span
    for _ in range(28):
        mid = (lo + hi) / 2
        cx = ax - off_x + mid * math.cos(ang)
        cy = ay - off_y + mid * math.sin(ang)
        frac = footprint_overlap(K, R, plan, (cx, cy), grid_b)
        if frac is None:
            hi = mid
            continue
        if frac > PARTIAL_OVERLAP[1]:
            lo = mid
        elif frac < PARTIAL_OVERLAP[0]:
            hi = mid
        else:
            return cx, cy
    return None


def make_sample(ortho, base, field, anchor, plan, rng, *, same_source: bool):
    """Один сэмпл: (запись для npz, мета) либо (None, причина отказа)."""
    K = intrinsics(FRAME_W, FRAME_H, F_PX)
    R = camera_rotation(plan["yaw"], plan["tilt"], plan["tilt_az"])
    gsd_b, b_px = plan["gsd_b"], plan["b_px"]
    span_b = b_px * gsd_b
    ax, ay = anchor

    pulled = pull_anchor_inside(ortho, ax, ay, span_b)
    if pulled is None or pulled[2] < 0.90:
        return None, "якорь: мало данных"
    ax, ay, _ = pulled
    # сетка кропа B повёрнута на свой курс: разница с курсом кадра ≤ DELTA_YAW_MAX
    grid_b = Grid(x=ax, y=ay, size_px=b_px, gsd=gsd_b, rot_deg=plan["yaw_b"])

    placed = place_camera(K, R, plan, grid_b, rng)
    if placed is None:
        return None, "перекрытие не подобралось"
    cx, cy = placed

    area_frac = footprint_overlap(K, R, plan, (cx, cy), grid_b)
    if area_frac is None:
        return None, "след за горизонтом"
    if plan["layout"] == "inside" and area_frac < 0.999:
        return None, "след не помещается в кроп B"
    if plan["layout"] == "partial" and not (PARTIAL_OVERLAP[0] <= area_frac <= PARTIAL_OVERLAP[1]):
        return None, f"перекрытие {area_frac:.2f} вне диапазона"

    a_rgb, a_val, gx, gy = render_frame(ortho, (cx, cy), plan["height"], R, K)
    if a_rgb is None:
        return None, "кадр вне растра"
    if float(a_val.mean()) < MIN_VALID_A and plan["layout"] == "inside":
        return None, f"кадр покрыт на {a_val.mean():.2f}"

    if same_source:
        b_rgb, b_val = ortho.read_grid(grid_b)
        comp = (0.0, 0.0)
        comp_src = "same_source"
        zoom = None
    else:
        comp_dx, comp_dy, comp_src = field.at(ax, ay)
        comp = (comp_dx, comp_dy)
        b_rgb, b_val, info = base.read_grid(grid_b, shift_m=comp)
        zoom = info["zoom"]
    if float(b_val.mean()) < MIN_VALID_B:
        return None, f"кроп B покрыт на {b_val.mean():.2f}"

    # warp: пиксель A → земля → пиксель кропа B (компенсация уже учтена в
    # чтении B, поэтому здесь координаты берутся в сетке ортоплана)
    wx, wy = grid_b.pixel_from_world(gx, gy)
    mask = (np.isfinite(wx) & np.isfinite(wy) & a_val
            & (wx >= 0) & (wx <= b_px - 1) & (wy >= 0) & (wy <= b_px - 1))
    if mask.sum() < 100:
        return None, "нет ко-видимости"
    # валидность стороны B в точках соответствия
    bx = np.clip(np.round(np.where(mask, wx, 0)).astype(int), 0, b_px - 1)
    by = np.clip(np.round(np.where(mask, wy, 0)).astype(int), 0, b_px - 1)
    mask &= b_val[by, bx]
    covis = float(mask.mean())
    if plan["layout"] == "inside" and covis < 0.90:
        return None, f"covis {covis:.2f} < 0.90"
    if plan["layout"] == "partial" and not (0.55 <= covis <= 0.95):
        return None, f"covis {covis:.2f} вне диапазона"

    warp = np.stack([wx, wy], axis=-1).astype(np.float16)
    warp[~mask] = np.nan
    meta = dict(
        source="orto_basemap", scene=ortho.path.stem,
        pair_kind="same_source" if same_source else "orto_basemap",
        pair_layout=plan["layout"], rectified=False,
        tilt_deg=round(plan["tilt"], 2), tilt_az_deg=round(plan["tilt_az"], 1),
        yaw_deg=round(plan["yaw"], 2), yaw_b_deg=round(plan["yaw_b"], 2),
        delta_yaw_deg=round(plan["delta_yaw"], 2),
        height_m=round(plan["height"], 1),
        fov_deg=round(fov_deg(FRAME_W, F_PX), 2),
        gsd_a=round(plan["gsd_a"], 4), gsd_b=round(gsd_b, 4),
        scale_ratio=round(plan["scale"], 3),
        b_px=b_px, footprint_a_m=round(FRAME_W * plan["gsd_a"], 1),
        footprint_b_m=round(span_b, 1), area_frac=round(area_frac, 4),
        covis_frac=round(covis, 4), valid_a_frac=round(float(a_val.mean()), 4),
        valid_b_frac=round(float(b_val.mean()), 4),
        anchor_xy=[round(ax, 1), round(ay, 1)],
        basemap_provider=None if same_source else "esri_world_imagery",
        basemap_zoom=zoom, compensation_m=[round(comp[0], 2), round(comp[1], 2)],
        compensation_src=comp_src, shift_field=field.source,
        season_a=None, season_b=None, date_a=None, date_b=None,
    )
    return dict(image_a=paint_invalid(a_rgb, a_val), image_b=paint_invalid(b_rgb, b_val),
                warp=warp, mask=mask.astype(np.uint8), meta=meta), None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raster", required=True)
    ap.add_argument("--shift-field", default=None)
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="open_orto/dataset")
    args = ap.parse_args()

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ortho = OrthoSource(args.raster)
    base = BasemapSource(ortho)
    field = ShiftField(args.shift_field)
    anchors = build_anchors(ortho)
    print(f"якорей: {len(anchors)} | поле сдвигов: {field.source} "
          f"(константа {field.global_dxy[0]:.2f}, {field.global_dxy[1]:.2f} м)")

    rng = np.random.default_rng(args.seed)
    manifest = (out / "manifest.csv")
    new = not manifest.exists()
    mf = manifest.open("a", encoding="utf-8")
    if new:
        mf.write("pair,scene,layout,pair_kind,height_m,tilt_deg,yaw_deg,scale_ratio,"
                 "b_px,area_frac,covis_frac,compensation_src,bytes\n")

    made, tries, reasons = 0, 0, {}
    while made < args.count and tries < args.count * 12:
        tries += 1
        layout = "inside" if rng.random() < INSIDE_QUOTA else "partial"
        same_source = rng.random() < SAME_SOURCE_QUOTA
        plan = plan_sample(rng, layout)
        ax, ay = anchors[int(rng.integers(len(anchors)))]
        rec, why = make_sample(ortho, base, field, (ax, ay), plan, rng,
                               same_source=same_source)
        if rec is None:
            reasons[why] = reasons.get(why, 0) + 1
            continue
        rec["meta"].update(gen_commit=commit, gen_date=date.today().isoformat(),
                           seed=int(args.seed), index=made)
        name = f"pair_{ortho.path.stem[:8]}_{made:05d}_{layout}" \
               f"{'_ss' if same_source else ''}.npz"
        dst = out / name
        np.savez_compressed(
            dst,
            image_a_jpeg=jpeg_bytes(rec["image_a"]), image_b_jpeg=jpeg_bytes(rec["image_b"]),
            warp_ab=rec["warp"], mask_ab=rec["mask"],
            meta=np.str_(json.dumps(rec["meta"], ensure_ascii=False)),
            pinhole=np.str_("null"))
        m = rec["meta"]
        mf.write(f"{name},{m['scene']},{layout},{m['pair_kind']},{m['height_m']},"
                 f"{m['tilt_deg']},{m['yaw_deg']},{m['scale_ratio']},{m['b_px']},"
                 f"{m['area_frac']},{m['covis_frac']},{m['compensation_src']},"
                 f"{dst.stat().st_size}\n")
        made += 1
        if made % 20 == 0:
            mf.flush()
            print(f"  {made}/{args.count} (попыток {tries})", flush=True)
    mf.close()
    print(f"готово: {made} пар за {tries} попыток → {out}")
    if reasons:
        print("отказы:", dict(sorted(reasons.items(), key=lambda kv: -kv[1])))
    ortho.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
