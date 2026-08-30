"""Разбор карантина: почему подложка разошлась с ортопланом.

Вердикт «привязка сдвинута» говорит только, что модель матчера расходится с
нашей разметкой. Причин у этого несколько, и лечатся они по-разному:

- **постоянный сдвиг** — компенсация привязки не подошла этому месту площадки
  (поле сдвигов неоднородно, а пара взята вдали от узлов). Лечится плотнее
  сеткой или прицельным замером ячейки;
- **поворот или масштаб** — георефа ортоплана расходится с подложкой не
  переносом, а аффинно. Компенсацией сдвига это не чинится в принципе;
- **нелинейный разброс** — рельеф и параллакс: расхождение растёт от центра к
  краям кадра и не описывается одной моделью;
- **ошибка самого арбитра** — на повторяющейся структуре (поля, теплицы,
  ряды застройки) матчер строит согласованную, но неверную гомографию. Тогда
  разметка на самом деле цела, а в карантин пара попала зря.

Последний случай отделяется перекрёстной проверкой **вторым независимым
ядром**: если два разных матчера сошлись на одном и том же смещении, ошибка в
разметке; если разошлись — ошибся арбитр.

    python open_orto/scripts/diagnose_quarantine.py --per-scene 3
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[2]))

from audit_basemap import MIN_OWN_FRAC, MIN_OWN_INLIERS, own_model  # noqa: E402
from bench_pairs import CONFIGS, load_pair  # noqa: E402
from aero_geoloc.matcher import create_matcher  # noqa: E402

#: Насколько согласованными должны быть два ядра, чтобы верить их общему
#: вердикту (px). Разница больше — арбитры не сошлись, разметка не опровергнута.
CROSS_AGREE_PX = 6.0
#: Порог, выше которого компонента считается значимой.
ROT_DEG = 0.5
SCALE_FRAC = 0.01
NONLINEAR_PX = 6.0


def decompose(H, warp, mask, step: int = 16):
    """Разложение расхождения модели матчера с нашим GT на компоненты.

    Возвращает словарь: медианный сдвиг (dx, dy), его модуль, поворот и
    масштаб подобия, подогнанного по расхождениям, и нелинейный остаток —
    то, что не объясняется ни сдвигом, ни поворотом, ни масштабом.
    """
    h, w = mask.shape
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    ok = mask[ys, xs]
    if ok.sum() < 30:
        return None
    pts = np.stack([xs[ok], ys[ok]], axis=-1).astype(np.float32)
    gt = warp[ys[ok], xs[ok]]
    good = np.isfinite(gt).all(axis=1)
    if good.sum() < 30:
        return None
    src = pts[good]
    gt = gt[good].astype(np.float32)
    pred = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)

    diff = pred - gt
    dx, dy = float(np.median(diff[:, 0])), float(np.median(diff[:, 1]))

    # подобие GT → предсказание матчера: поворот и масштаб расхождения
    M, _ = cv2.estimateAffinePartial2D(gt, pred, method=cv2.LMEDS)
    rot = scale = float("nan")
    resid = float("nan")
    if M is not None:
        scale = float(math.hypot(M[0, 0], M[1, 0]))
        rot = float(math.degrees(math.atan2(M[1, 0], M[0, 0])))
        fitted = (gt @ M[:, :2].T) + M[:, 2]
        resid = float(np.median(np.hypot(*(pred - fitted).T)))
    return dict(dx=dx, dy=dy, shift=math.hypot(dx, dy), rot=rot, scale=scale,
                nonlinear=resid,
                spread=float(np.median(np.hypot(*(diff - [dx, dy]).T))))


def classify(rec, cross_ok: bool | None, gsd_b: float) -> str:
    """Причина расхождения одной пары — по компонентам и перекрёстной проверке."""
    if cross_ok is False:
        return "арбитры не сошлись"
    if rec is None:
        return "не разложено"
    if abs(rec["rot"]) > ROT_DEG or abs(rec["scale"] - 1.0) > SCALE_FRAC:
        return "поворот/масштаб георефы"
    if rec["nonlinear"] > NONLINEAR_PX:
        return "нелинейность (рельеф/параллакс)"
    return "постоянный сдвиг"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="open_orto/dataset_base_quarantine")
    ap.add_argument("--matcher", default="minima_roma")
    ap.add_argument("--cross", default="minima_loftr", help="второе ядро для сверки")
    ap.add_argument("--per-scene", type=int, default=3)
    ap.add_argument("--limit-scenes", type=int, default=0,
                    help="взять не больше N площадок (для контрольного замера "
                         "по подтверждённому корпусу)")
    ap.add_argument("--out", default="open_orto/work/diagnose_quarantine.csv")
    args = ap.parse_args()
    from cpu_affinity import pin_to_performance
    pin_to_performance(verbose=False)

    root = Path(args.dataset)
    by_scene: dict[str, list[Path]] = defaultdict(list)
    for f in sorted(root.glob("*.npz")):
        by_scene[f.name.split("_")[1]].append(f)
    if args.limit_scenes:
        by_scene = dict(sorted(by_scene.items())[: args.limit_scenes])
    print(f"площадок в карантине: {len(by_scene)}, "
          f"проверяем до {args.per_scene} пар с каждой", flush=True)

    m1 = create_matcher(args.matcher, **CONFIGS[args.matcher])
    m2 = create_matcher(args.cross, **CONFIGS[args.cross])

    fields = ["pair", "scene", "причина", "shift_px", "shift_m", "dx", "dy",
              "rot_deg", "scale", "nonlinear_px", "cross_diff_px",
              "compensation_src", "field_nodes", "gsd_b"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for i, (tag, files) in enumerate(sorted(by_scene.items()), 1):
            for f in files[: args.per_scene]:
                pair = load_pair(f)
                meta = pair["meta"]
                a = cv2.cvtColor(pair["image_a"], cv2.COLOR_BGR2GRAY)
                b = cv2.cvtColor(pair["image_b"], cv2.COLOR_BGR2GRAY)
                H1, frac1, n1 = own_model(m1.match(a, b))
                if H1 is None or n1 < MIN_OWN_INLIERS or frac1 < MIN_OWN_FRAC:
                    continue
                rec = decompose(H1, pair["warp"], pair["mask"])
                # перекрёстная проверка: сошлись ли два независимых ядра
                cross_diff = float("nan")
                cross_ok = None
                H2, frac2, n2 = own_model(m2.match(a, b))
                if H2 is not None and n2 >= MIN_OWN_INLIERS and frac2 >= MIN_OWN_FRAC:
                    rec2 = decompose(H2, pair["warp"], pair["mask"])
                    if rec and rec2:
                        cross_diff = math.hypot(rec["dx"] - rec2["dx"],
                                                rec["dy"] - rec2["dy"])
                        cross_ok = cross_diff <= CROSS_AGREE_PX
                gsd_b = float(meta.get("gsd_b") or 0.0)
                nodes = 0
                fld = Path("open_orto/work/shift") / f"shift_field_{meta['scene']}.npz"
                if fld.exists():
                    nodes = int(len(np.load(fld, allow_pickle=False)["x"]))
                row = dict(
                    pair=f.name, scene=meta["scene"],
                    **{"причина": classify(rec, cross_ok, gsd_b)},
                    shift_px=round(rec["shift"], 2) if rec else "",
                    shift_m=round(rec["shift"] * gsd_b, 2) if rec else "",
                    dx=round(rec["dx"], 2) if rec else "",
                    dy=round(rec["dy"], 2) if rec else "",
                    rot_deg=round(rec["rot"], 3) if rec else "",
                    scale=round(rec["scale"], 4) if rec else "",
                    nonlinear_px=round(rec["nonlinear"], 2) if rec else "",
                    cross_diff_px=round(cross_diff, 2) if cross_diff == cross_diff else "",
                    compensation_src=meta.get("compensation_src", ""),
                    field_nodes=nodes, gsd_b=gsd_b)
                w.writerow(row)
                rows.append(row)
            fh.flush()
            if i % 10 == 0:
                print(f"  {i}/{len(by_scene)} площадок, замеров {len(rows)}", flush=True)

    print(f"\nразобрано пар: {len(rows)} → {out}")
    if rows:
        tally: dict[str, int] = defaultdict(int)
        for r in rows:
            tally[r["причина"]] += 1
        for k, n in sorted(tally.items(), key=lambda t: -t[1]):
            print(f"  {n:4}  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
