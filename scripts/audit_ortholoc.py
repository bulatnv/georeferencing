"""Расхождение привязки между источниками ортофото в OrthoLoC.

У сэмплов вида ``R`` сторона B — «своё» ортофото территории, у ``xDOP`` — то
же место, снятое другим источником (другой заказ аэросъёмки, другая дата).
Кадр и его попиксельная 3D (``point_map``) у них общие, ``scale`` и размер DOP
совпадают, поэтому **GT-карта у обоих вариантов одна и та же** — проверено на
выборке. Это значит, что формула GT молча предполагает: оба ортофото
привязаны к одной мировой сетке идеально.

Если чужое ортофото смещено относительно своего, весь этот сдвиг уходит прямо
в разметку ``xDOP``-пар — тот же дефект, что расхождение геопривязки
ортоплана и мозаики Esri в нашем корпусе, и такой же незаметный глазом.

Здесь он измеряется: два ортофото одной территории сводятся матчером, по его
соответствиям строится 4-DoF подобие (обе стороны ортографичны, модель
применима), и сдвиг этой модели и есть систематическая ошибка разметки
``xDOP``-пары. Матчер тут измеритель, а не источник разметки: его
соответствия никуда не записываются.

    python scripts/audit_ortholoc.py --per-scene 12 --out eval_out/ortholoc_audit.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE.parent))

import ortholoc_store as store  # noqa: E402
from aero_geoloc.matcher import create_matcher  # noqa: E402

#: Минимум инлайеров, при котором замер считается состоявшимся. Ниже —
#: «без структуры»: пара ортофото не сводится, и сказать о привязке нечего.
MIN_INLIERS = 30
#: Порог RANSAC, px. Берётся с запасом: измеряем систематический сдвиг
#: привязки, а не субпиксельную точность соответствий.
RANSAC_THR = 3.0
#: Признаки того, что модель построена на мусоре, а не на реальном сведении:
#: замерено — у состоявшихся замеров доля инлайеров 0.81 по медиане, а два
#: промаха из 336 дали 0.02–0.03 при повороте 17° и 178°. Ортофото одной
#: территории не могут расходиться поворотом в градусы: это разные места.
MIN_INLIER_FRAC = 0.10
MAX_ROT_DEG = 5.0

FIELDS = ["scene", "sample", "n_pairs", "n_inliers", "inlier_frac",
          "dx_px", "dy_px", "shift_px", "shift_m", "rot_deg", "scale_dev",
          "gsd", "status"]


def groups(root: Path, splits) -> dict:
    """Сэмплы, у которых есть и ``R``, и ``xDOP``: (сцена, номер) → пути."""
    out = defaultdict(dict)
    for split in splits:
        for p in (root / split).glob("*.npz"):
            m = re.match(r"(L\d+)_(R|xDOP|xDOPDSM)(\d+)$", p.stem)
            if m:
                out[(split, m.group(1), m.group(3))][m.group(2)] = p
    return {k: v for k, v in out.items() if "R" in v and "xDOP" in v}


def measure(matcher, path_r: Path, path_x: Path) -> dict:
    """Сдвиг привязки между двумя ортофото одной территории."""
    with store.open_sample(path_r) as a, store.open_sample(path_x) as b:
        dop_r = cv2.cvtColor(a["image_dop"], cv2.COLOR_RGB2GRAY)
        dop_x = cv2.cvtColor(b["image_dop"], cv2.COLOR_RGB2GRAY)
        gsd = float(abs(np.asarray(a["scale"])[0]))
    corr = matcher.match(dop_r, dop_x)
    n = len(corr)
    row = dict(n_pairs=n, n_inliers=0, inlier_frac="", dx_px="", dy_px="",
               shift_px="", shift_m="", rot_deg="", scale_dev="",
               gsd=round(gsd, 4), status="без структуры")
    if n < MIN_INLIERS:
        return row
    model, inl = cv2.estimateAffinePartial2D(
        corr.pts_q.astype(np.float32), corr.pts_r.astype(np.float32),
        method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THR,
        maxIters=5000, confidence=0.999)
    if model is None or inl is None or int(inl.sum()) < MIN_INLIERS:
        return row
    dx, dy = float(model[0, 2]), float(model[1, 2])
    scale = float(np.hypot(model[0, 0], model[1, 0]))
    rot = float(np.degrees(np.arctan2(model[1, 0], model[0, 0])))
    if float(inl.mean()) < MIN_INLIER_FRAC or abs(rot) > MAX_ROT_DEG:
        row.update(n_inliers=int(inl.sum()),
                   inlier_frac=round(float(inl.mean()), 4),
                   rot_deg=round(rot, 3), status="не сведено")
        return row
    row.update(n_inliers=int(inl.sum()),
               inlier_frac=round(float(inl.mean()), 4),
               dx_px=round(dx, 2), dy_px=round(dy, 2),
               shift_px=round(float(np.hypot(dx, dy)), 2),
               shift_m=round(float(np.hypot(dx, dy)) * gsd, 3),
               rot_deg=round(rot, 3), scale_dev=round(scale - 1.0, 5),
               status="измерено")
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="data/OrthoLoC")
    ap.add_argument("--splits", default="train,val,test_inPlace,test_outPlace")
    ap.add_argument("--matcher", default="romav2",
                    help="сильнейшее ядро на этом датасете (BENCH_ORTHOLOC §2)")
    ap.add_argument("--per-scene", type=int, default=12,
                    help="сколько сэмплов мерить на сцену")
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--out", default="eval_out/ortholoc_audit.csv")
    args = ap.parse_args()

    root = Path(args.root)
    found = groups(root, args.splits.split(","))
    by_scene = defaultdict(list)
    for key in sorted(found):
        by_scene[key[1]].append(key)
    rng = np.random.default_rng(args.seed)
    tasks = []
    for scene, keys in sorted(by_scene.items()):
        idx = rng.permutation(len(keys))[:args.per_scene]
        tasks += [keys[i] for i in sorted(idx)]
    print(f"сцен {len(by_scene)}, сэмплов с парой источников {len(found)}, "
          f"замеров {len(tasks)}")

    matcher = create_matcher(args.matcher)
    rows, t0 = [], time.time()
    for i, key in enumerate(tasks, 1):
        split, scene, num = key
        row = measure(matcher, found[key]["R"], found[key]["xDOP"])
        rows.append(dict(scene=scene, sample=f"{scene}_{num}", **row))
        if i % 50 == 0 or i == len(tasks):
            el = time.time() - t0
            print(f"  {i}/{len(tasks)}  {el/60:.1f} мин, "
                  f"осталось ~{el/i*(len(tasks)-i)/60:.0f} мин", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["status"] == "измерено"]
    print(f"\nизмерено {len(ok)} из {len(rows)}")
    if ok:
        sh = np.array([r["shift_px"] for r in ok])
        sm = np.array([r["shift_m"] for r in ok])
        rot = np.abs([r["rot_deg"] for r in ok])
        sc = np.abs([r["scale_dev"] for r in ok])
        print(f"  сдвиг привязки: медиана {np.median(sh):.2f} px "
              f"({np.median(sm):.2f} м), 90-й перцентиль {np.percentile(sh, 90):.2f} px")
        print(f"  |поворот| медиана {np.median(rot):.3f}°, "
              f"|масштаб−1| медиана {np.median(sc)*100:.3f} %")
        print("\n  по сценам (медиана сдвига, px):")
        for scene in sorted({r["scene"] for r in ok}):
            v = [r["shift_px"] for r in ok if r["scene"] == scene]
            print(f"    {scene:5} n={len(v):3}  {np.median(v):6.2f}")
    print(f"отчёт: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
