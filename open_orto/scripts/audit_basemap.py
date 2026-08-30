"""Аудит разметки пар «орто ↔ подложка» независимым арбитром — матчером.

Зачем отдельно от `report.py`. Тамошний контроль (фазовая корреляция кадра с
натянутой подложкой) на этих парах **врёт**: при смене фактуры — орто снято
зимой, подложка летняя — корреляция по градиентам вырождается в шум и выдаёт
десятки пикселей там, где разметка верна. Замерено: 37 px против 3 px по
матчеру на той же паре. Для `same_source` тот контроль остаётся точным (0.02
px), для подложки нужен другой арбитр.

Матчер здесь — **измеритель, а не источник разметки**: его соответствия ни во
что не записываются, иначе модель обучалась бы на собственных предсказаниях.

Три исхода различаются так:

- матчер согласован сам с собой (много инлайеров своей RANSAC-модели) и
  согласен с нашим GT → **разметка верна**;
- матчер согласован сам с собой, но систематически расходится с GT →
  **привязка площадки сдвинута**, пары брак: они выглядят нормально и учат
  модель неверному соответствию;
- матчер не согласован сам с собой → **площадка без структуры** (сплошное
  поле, лес, вода) либо непосильный кросс-сезон. Разметка тут не опровергнута,
  но и не подтверждена; такие пары помечаются, а не выбрасываются молча —
  трудные случаи датасету нужны.

    python open_orto/scripts/audit_basemap.py --dataset open_orto/dataset_base \\
        --per-scene 4 --out open_orto/work/audit_basemap.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[2]))

from bench_pairs import CONFIGS, load_pair  # noqa: E402
from aero_geoloc.matcher import create_matcher  # noqa: E402

#: Сколько инлайеров своей модели нужно, чтобы считать матчер сработавшим.
MIN_OWN_INLIERS = 25
#: Доля инлайеров своей модели — та же проверка с другой стороны.
MIN_OWN_FRAC = 0.30
#: Расхождение модели матчера с нашим GT, выше которого привязка считается
#: сдвинутой. Порог с запасом к измеренному пределу источников (~4 px:
#: геопривязка орто, мозаика подложки, параллакс, разный рельеф).
MAX_GT_DIFF_PX = 12.0
#: Доля подтверждённых пар, ниже которой площадка не считается проверенной.
MIN_CONFIRMED_FRAC = 0.34


def own_model(corr, ransac_px: float = 6.0):
    """RANSAC-гомография по соответствиям матчера: (H, доля инлайеров, число).

    Согласованность матчера с самим собой не зависит от нашей разметки —
    именно поэтому она отделяет «матчер не смог» от «разметка сдвинута».
    """
    n = len(corr)
    if n < 12:
        return None, 0.0, 0
    H, inl = cv2.findHomography(corr.pts_q.astype(np.float32),
                                corr.pts_r.astype(np.float32),
                                cv2.USAC_MAGSAC, ransac_px, maxIters=5000)
    if H is None or inl is None:
        return None, 0.0, 0
    inl = inl.ravel().astype(bool)
    return H, float(inl.mean()), int(inl.sum())


def gt_disagreement(H, warp, mask) -> float | None:
    """Медиана расхождения модели матчера с нашим GT, px (по узлам сетки)."""
    h, w = mask.shape
    ys, xs = np.mgrid[0:h:16, 0:w:16]
    ok = mask[ys, xs]
    if ok.sum() < 20:
        return None
    pts = np.stack([xs[ok], ys[ok]], axis=-1).astype(np.float32)
    gt = warp[ys[ok], xs[ok]]
    good = np.isfinite(gt).all(axis=1)
    if good.sum() < 20:
        return None
    pred = cv2.perspectiveTransform(pts[good].reshape(-1, 1, 2), H).reshape(-1, 2)
    return float(np.median(np.hypot(pred[:, 0] - gt[good, 0], pred[:, 1] - gt[good, 1])))


def verdict(rows) -> tuple[str, dict]:
    """Вердикт по площадке из построчных замеров её пар."""
    worked = [r for r in rows if r["own_ok"]]
    confirmed = [r for r in worked if r["gt_diff"] is not None
                 and r["gt_diff"] <= MAX_GT_DIFF_PX]
    shifted = [r for r in worked if r["gt_diff"] is not None
               and r["gt_diff"] > MAX_GT_DIFF_PX]
    stat = dict(
        pairs=len(rows), matcher_worked=len(worked),
        confirmed=len(confirmed), shifted=len(shifted),
        gt_diff_med=(round(float(np.median([r["gt_diff"] for r in confirmed])), 2)
                     if confirmed else ""),
    )
    if not worked:
        return "без структуры", stat
    if len(shifted) > len(confirmed):
        return "привязка сдвинута", stat
    if len(confirmed) / max(len(rows), 1) >= MIN_CONFIRMED_FRAC:
        return "разметка подтверждена", stat
    return "проверено частично", stat


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="open_orto/dataset_base")
    ap.add_argument("--matcher", default="minima_roma")
    ap.add_argument("--per-scene", type=int, default=4,
                    help="сколько пар площадки проверять")
    ap.add_argument("--limit-scenes", type=int, default=0)
    ap.add_argument("--out", default="open_orto/work/audit_basemap.csv")
    args = ap.parse_args()

    root = Path(args.dataset)
    by_scene: dict[str, list[Path]] = defaultdict(list)
    for f in sorted(root.glob("*.npz")):
        by_scene[f.name.split("_")[1]].append(f)
    scenes = sorted(by_scene)
    if args.limit_scenes:
        scenes = scenes[: args.limit_scenes]
    print(f"площадок {len(scenes)}, проверяем до {args.per_scene} пар с каждой",
          flush=True)

    matcher = create_matcher(args.matcher, **CONFIGS[args.matcher])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done_scenes = set()
    if out.exists() and out.stat().st_size > 0:
        done_scenes = {r["scene_tag"] for r in csv.DictReader(out.open(encoding="utf-8"))}
        print(f"уже проверено площадок: {len(done_scenes)}", flush=True)

    fields = ["scene_tag", "scene", "вердикт", "pairs", "matcher_worked",
              "confirmed", "shifted", "gt_diff_med"]
    with out.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if not done_scenes:
            w.writeheader()
        tally: dict[str, int] = defaultdict(int)
        for i, tag in enumerate(scenes, 1):
            if tag in done_scenes:
                continue
            rows, scene_name = [], ""
            for f in by_scene[tag][: args.per_scene * 3]:
                if len(rows) >= args.per_scene:
                    break
                pair = load_pair(f)
                m = pair["meta"]
                if m.get("pair_kind") == "same_source":
                    continue
                scene_name = m.get("scene", "")
                corr = matcher.match(cv2.cvtColor(pair["image_a"], cv2.COLOR_BGR2GRAY),
                                     cv2.cvtColor(pair["image_b"], cv2.COLOR_BGR2GRAY))
                H, frac, n_inl = own_model(corr)
                own_ok = H is not None and n_inl >= MIN_OWN_INLIERS and frac >= MIN_OWN_FRAC
                rows.append(dict(own_ok=own_ok,
                                 gt_diff=gt_disagreement(H, pair["warp"], pair["mask"])
                                 if own_ok else None))
            if not rows:
                continue
            v, stat = verdict(rows)
            tally[v] += 1
            w.writerow(dict(scene_tag=tag, scene=scene_name, **{"вердикт": v}, **stat))
            fh.flush()
            if i % 10 == 0:
                print(f"  {i}/{len(scenes)}: " +
                      ", ".join(f"{k} {n}" for k, n in sorted(tally.items())), flush=True)
    print("итог: " + ", ".join(f"{k} — {n}" for k, n in sorted(tally.items())))
    print(f"построчно: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
