"""Сборка поставки OrthoLoC в том же виде, что `openaerialmap_dataset`.

Пары уже лежат в каноническом формате (`convert_ortholoc.py`), но обучаться и
валидироваться на них нельзя: нет сплитов без утечки, нет классов качества и
весов, нет манифеста с теми колонками, которые читает загрузчик из
`TRAINING.md`. Этот скрипт достраивает недостающее.

**Сплиты строятся заново, а не берутся из датасета.** В OrthoLoC `train` и
`val` — одни и те же 48 сцен (val держит отложенные кадры тех же территорий),
а `test_inPlace` — три сцены, входящие в `train`. По нашему правилу «делить по
территориям» такое деление даёт утечку. Честный held-out здесь ровно один:
`test_outPlace` (L08, L50, L51) — сцены, не встречающиеся больше нигде. Он и
становится приёмкой, `val` набирается из отдельных сцен обучения, а исходная
метка сплита сохраняется колонкой `src_split`.

**Состав прореживается по опыту своего корпуса.** Пары `rect` (ортофото против
ортофото) — лёгкая ось того же рода, что `same_source`: там разметка
аналитическая, и линия RoMa решает такие пары почти нацело. Держать их
половиной корпуса значит тратить эпоху на нулевой градиент, поэтому ось
сокращается до доли `--rect-frac`. Сезонные перекраски не берутся вовсе:
замерено на своём корпусе, что синтетическая перекраска почти не создаёт
appearance-разрыва (корреляция яркости 0.99 против 0.99 у неперекрашенных).

    python scripts/build_ortholoc_dataset.py --out ortholoc_dataset
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

import ortholoc_store as store  # noqa: E402

#: Сцены, которых нет ни в обучении, ни в валидации исходного датасета:
#: единственный честный held-out этого корпуса.
HELDOUT_SCENES = ("L08", "L50", "L51")

#: Вес пары в смеси по классу разметки — те же значения, что в
#: `openaerialmap_dataset`, чтобы корпуса можно было смешивать одним сэмплером.
WEIGHTS = {"registered": 1.0, "approx": 0.3, "exact": 0.5}

#: Ожидаемая ошибка разметки, px.
#:
#: - `rect`: геометрия аналитическая (warp — целочисленный сдвиг сеток);
#: - чужое ортофото: измеренный сдвиг привязки сцены (`audit_ortholoc.py`);
#: - своё ортофото: прямого измерения нет. Взята **оценка сверху** — медиана
#:   EPE сильнейшего ядра на таких парах held-out (0.97 px), куда входит и
#:   ошибка самой модели, то есть настоящий шум разметки заведомо меньше;
#: - сцена без замера привязки: величина заведомо больше типичной измеренной
#:   (медиана по корпусу 1.5 px), плюс класс понижается до `approx`.
SIGMA_RECT = 0.10
SIGMA_OWN_DOP = 0.97
SIGMA_FALLBACK = 3.0

#: Виды пар этого корпуса. Боевой тип — кадр против **чужого** ортофото:
#: другой источник и другая дата, то есть настоящий appearance-разрыв.
KIND_XDOP = "frame_xdop"
KIND_DOP = "frame_dop"
KIND_RECT = "rect_ortho"

FIELDS = ["pair", "corpus", "scene", "pair_kind", "mode", "layout", "season",
          "height_m", "tilt_deg", "scale_ratio", "covis_frac", "b_px",
          "src_split", "geo_cluster", "split", "gt_class", "gt_sigma_px",
          "gt_sigma_src", "weight", "slice", "modality", "bytes"]


def sample_of(pair: str) -> str:
    """Имя сэмпла OrthoLoC, из которого построена пара."""
    return re.sub(r"^pair_|_(asis|rect)(_[saw])?\.npz$", "", pair)


def variant_of(sample: str) -> str:
    m = re.match(r"L\d+_(R|xDOP|xDOPDSM)\d+$", sample)
    return m.group(1) if m else "?"


def classify(mode: str, variant: str, scene: str, shifts: dict):
    """Вид пары, класс разметки и ожидаемая ошибка, px.

    Возвращает и **происхождение** этой ошибки: измерена она или оценена. Без
    такой пометки оценка через несколько недель читается как измерение.

    Три случая различаются тем, откуда берётся сторона B и что при этом может
    разойтись:

    - ``rect`` — обе стороны на одной сетке, warp есть целочисленный сдвиг:
      расходиться нечему, класс ``exact``;
    - кадр против **своего** ортофото: GT приходит из ``point_map`` датасета,
      расходиться источникам не с чем, но и проверить точность 3D-реконструкции
      нечем — ставится оценка сверху;
    - кадр против **чужого** ортофото: сюда добавляется расхождение привязки
      двух источников, измеренное отдельно (`audit_ortholoc.py`). Там, где
      измерения нет, класс понижается до ``approx``: неизмеренная привязка —
      не то же самое, что подтверждённая.
    """
    if mode == "rect":
        return KIND_RECT, "exact", SIGMA_RECT, "аналитическая"
    if variant in ("xDOP", "xDOPDSM"):
        if scene in shifts:
            return KIND_XDOP, "registered", shifts[scene], "измерено"
        return KIND_XDOP, "approx", SIGMA_FALLBACK, "оценка"
    return KIND_DOP, "registered", SIGMA_OWN_DOP, "оценка"


def load_audit(path: Path) -> dict:
    """Сдвиг привязки по сценам: медиана измеренного расхождения источников."""
    if not path.exists():
        return {}
    rows = [r for r in csv.DictReader(path.open(encoding="utf-8"))
            if r["status"] == "измерено" and r["shift_px"] != ""]
    by = defaultdict(list)
    for r in rows:
        by[r["scene"]].append(float(r["shift_px"]))
    return {s: float(np.median(v)) for s, v in by.items()}


def height_of(root: Path, src_split: str, sample: str, cache: dict) -> float:
    """Высота съёмки над медианной вершиной меша сцены, м.

    Читается из исходного сэмпла: в паре её нет. Чтение ленивое — берутся два
    мелких массива, а не изображения.
    """
    key = (src_split, sample)
    if key in cache:
        return cache[key]
    path = root / src_split / f"{sample}.npz"
    val = float("nan")
    if path.exists():
        try:
            with store.open_sample(path) as d:
                ext = np.asarray(d["extrinsics"], dtype=np.float64)
                R, t = ext[:, :3], ext[:, 3]
                val = float((-R.T @ t)[2] - d.median_vertex_z)
        except Exception:  # noqa: BLE001
            pass
    cache[key] = val
    return val


def pick_rect(rows: list, frac: float, rng) -> set:
    """Какие `rect`-пары оставить: доля от числа боевых, поровну по сценам.

    Ось нужна как контроль забывания, поэтому важно покрытие территорий, а не
    объём: с каждой сцены берётся своя квота, а не случайные пары корпуса.
    """
    battle = [r for r in rows if r["mode"] == "asis"]
    target = int(round(len(battle) * frac))
    rect_by_scene = defaultdict(list)
    for r in rows:
        if r["mode"] == "rect":
            rect_by_scene[r["scene"]].append(r)
    if not rect_by_scene:
        return set()
    per = max(1, target // len(rect_by_scene))
    keep = set()
    for scene, rs in sorted(rect_by_scene.items()):
        idx = rng.permutation(len(rs))[:per]
        keep |= {rs[i]["pair"] for i in idx}
    return keep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", default="data/pairs_ortholoc")
    ap.add_argument("--raw", default="data/OrthoLoC")
    ap.add_argument("--out", default="ortholoc_dataset")
    ap.add_argument("--audit", default="eval_out/ortholoc_audit.csv")
    ap.add_argument("--rect-frac", type=float, default=0.15,
                    help="доля контрольной оси от числа боевых пар")
    ap.add_argument("--val-scenes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pairs_root = Path(args.pairs)
    src = list(csv.DictReader((pairs_root / "manifest.csv").open(encoding="utf-8")))
    rng = np.random.default_rng(args.seed)
    shifts = load_audit(Path(args.audit))
    print(f"пар в исходном манифесте: {len(src)}; "
          f"сцен с измеренной привязкой: {len(shifts)}")

    # ——— отбор состава: сезонные не берём, rect прореживаем
    plain = [r for r in src if not re.search(r"_[saw]\.npz$", r["pair"])]
    for r in plain:
        r["scene"] = r["scene"] or sample_of(r["pair"]).split("_")[0]
    keep_rect = pick_rect(plain, args.rect_frac, rng)
    chosen = [r for r in plain
              if r["mode"] == "asis" or r["pair"] in keep_rect]
    print(f"отобрано: {len(chosen)} пар "
          f"(боевых {sum(1 for r in chosen if r['mode']=='asis')}, "
          f"контрольных {sum(1 for r in chosen if r['mode']=='rect')}); "
          f"отброшено сезонных {len(src)-len(plain)}, "
          f"лишних rect {len(plain)-len(chosen)}")

    # ——— сплиты по сценам
    scenes = sorted({r["scene"] for r in chosen})
    held = [s for s in scenes if s in HELDOUT_SCENES]
    rest = [s for s in scenes if s not in HELDOUT_SCENES]
    val = sorted(np.array(rest)[rng.permutation(len(rest))[:args.val_scenes]].tolist())
    split_of = {s: ("heldout" if s in held else "val" if s in val else "train")
                for s in scenes}

    cache: dict = {}
    out_rows = []
    for r in chosen:
        sample = sample_of(r["pair"])
        var = variant_of(sample)
        scene = r["scene"]
        kind, gt_class, sigma, sigma_src = classify(r["mode"], var, scene, shifts)
        covis = float(r["covis_frac"])
        out_rows.append(dict(
            pair=r["pair"], corpus="ortholoc", scene=scene, pair_kind=kind,
            mode=r["mode"], layout="inside" if covis >= 0.999 else "partial",
            season="", height_m=round(height_of(Path(args.raw), r["split"],
                                                sample, cache), 1),
            tilt_deg=r["tilt_deg"], scale_ratio="", covis_frac=r["covis_frac"],
            b_px="", src_split=r["split"], geo_cluster=f"c_{scene}",
            split=split_of[scene], gt_class=gt_class,
            gt_sigma_px=round(float(sigma), 3), gt_sigma_src=sigma_src,
            weight=f"{WEIGHTS[gt_class]:.2f}",
            slice="inplace" if r["split"] == "test_inPlace" else "",
            modality="photo", bytes=r["bytes"]))

    # ——— досчёт того, чего нет в исходном манифесте
    for row, r in zip(out_rows, chosen):
        path = pairs_root / r["split"] / r["pair"]
        if path.exists():
            d = np.load(path, allow_pickle=False)
            meta = json.loads(str(d["meta"]))
            row["scale_ratio"] = meta.get("scale_ratio", "")
            row["b_px"] = meta.get("b_window", [0, 0, 0, 0])[2]

    print("\nсплиты:")
    for sp in ("train", "val", "heldout"):
        rs = [r for r in out_rows if r["split"] == sp]
        gb = sum(int(r["bytes"]) for r in rs) / 2**30
        kinds = Counter(r["pair_kind"] for r in rs)
        print(f"  {sp:8} пар {len(rs):6}  сцен {len({r['scene'] for r in rs}):3}  "
              f"{gb:5.2f} ГБ  боевых {kinds[KIND_XDOP]+kinds[KIND_DOP]:5} "
              f"(xDOP {kinds[KIND_XDOP]}), контроль {kinds[KIND_RECT]}")
    tot = sum(int(r["bytes"]) for r in out_rows) / 2**30
    w = sum(float(r["weight"]) for r in out_rows)
    print(f"\nитого {len(out_rows)} пар, {tot:.2f} ГБ")
    print("  доли в смеси:", {k: f"{100*sum(float(r['weight']) for r in out_rows if r['gt_class']==k)/w:.0f} %"
                              for k in WEIGHTS})
    if args.dry_run:
        return 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    linked = 0
    for r, row in zip(chosen, out_rows):
        srcp = pairs_root / r["split"] / r["pair"]
        dstp = out / r["pair"]
        if dstp.exists():
            dstp.unlink()
        try:
            import os
            os.link(srcp, dstp)                 # жёсткая ссылка: место не тратится
        except OSError:
            import shutil
            shutil.copy2(srcp, dstp)
        linked += 1
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(out_rows)
    files = len(list(out.glob("*.npz")))
    print(f"\nсобрано: файлов {files}, строк манифеста {len(out_rows)}"
          + ("  ← расходится!" if files != len(out_rows) else ""))
    print(f"поставка: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
