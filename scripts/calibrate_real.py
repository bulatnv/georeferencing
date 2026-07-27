"""Калибровка доверия на реальных бортовых кадрах против Esri.

Прогоняет ``localize`` на всех надирных снимках датасета, сравнивает результат с
GPS (ground truth) и собирает сигналы (инлайеры, NCC, confidence, эллипс) в CSV.
Затем считает то, что синтетика показать не может:

- **доля ложных срабатываний** — локализовано, но неверно (ошибка > порога): это
  самое опасное, «уверенно-неверная точка», её должно быть ~0;
- **порог сигнала**, отделяющий верные локализации от ложных (индекс Юдена);
- **reliability**: совпадает ли эмпирическая доля верных с заявленным доверием.

    python scripts/calibrate_real.py --data for_binding --out calib.csv

Нужны: torch + LightGlue, Pillow, сеть (тайлы Esri; кэшируются в --cache).
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from aero_geoloc.basemap import ESRI_WORLD_IMAGERY, TileBasemap, TileCache  # noqa: E402
from aero_geoloc.drone import basemap_zoom_for, frame_at_mpp, load_drone_shot  # noqa: E402
from aero_geoloc.geo import ground_mpp, haversine_m  # noqa: E402
from aero_geoloc.localize import localize  # noqa: E402
from aero_geoloc.matcher import create_matcher  # noqa: E402
from aero_geoloc.retrieval import calibrate_uniqueness_threshold  # noqa: E402

FIELDS = ["dataset", "file", "model", "altitude_m", "localized", "error_m", "correct",
          "status", "confidence", "n_inliers", "inlier_ratio", "photometric_ncc",
          "ellipse_major_m", "reason"]


def collect(args) -> list[dict]:
    max_zoom = ESRI_WORLD_IMAGERY.max_zoom
    matcher = create_matcher(args.matcher)
    prerotate = args.matcher in ("lightglue", "loftr")
    basemap = TileBasemap(cache=TileCache(args.cache))
    rows: list[dict] = []

    for dataset in sorted(p for p in Path(args.data).iterdir() if p.is_dir()):
        files = sorted(dataset.glob("*.JPG"))
        if len(files) > 30:  # прореживаем только крупные наборы, малые берём целиком
            files = files[:: args.step]
        for path in files:
            try:
                shot = load_drone_shot(path)
            except ValueError:
                continue  # нет GPS/фокуса/DJI-XMP — не наш кадр
            if not shot.is_nadir:
                continue
            z = basemap_zoom_for(shot, max_zoom=max_zoom)
            frame, camera = frame_at_mpp(shot, ground_mpp(shot.true_lat, z))
            result = localize(
                frame, camera, shot.prior(sigma_m=args.sigma_m), basemap,
                matcher=matcher, max_zoom=max_zoom, prerotate=prerotate, min_ncc=args.min_ncc,
                min_inliers=10, coarse_min_inliers=8, ransac_threshold_px=6.0,
            )
            d = result.diagnostics
            err = (
                haversine_m(shot.true_lat, shot.true_lon, result.center_lat, result.center_lon)
                if result.is_localized else float("nan")
            )
            rows.append({
                "dataset": dataset.name, "file": path.name, "model": shot.model,
                "altitude_m": round(shot.altitude_m, 1),
                "localized": int(result.is_localized),
                "error_m": round(err, 2) if result.is_localized else "",
                "correct": int(result.is_localized and err < args.correct_m),
                "status": result.status.value, "confidence": round(result.confidence, 4),
                "n_inliers": d.get("n_inliers", 0), "inlier_ratio": round(d.get("inlier_ratio", 0.0), 4),
                "photometric_ncc": (round(d["photometric_ncc"], 4) if d.get("photometric_ncc") is not None else ""),
                "ellipse_major_m": (round(result.error_ellipse_m[0], 3) if result.is_localized else ""),
                "reason": d.get("reason", ""),
            })
            tag = (f"OK {err:.1f}м" if result.is_localized else "отказ")
            print(f"  {dataset.name}/{path.name}: {tag}")
    return rows


def report(rows: list[dict], correct_m: float) -> None:
    n = len(rows)
    loc = [r for r in rows if r["localized"]]
    correct = [r for r in loc if r["correct"]]
    false_pos = [r for r in loc if not r["correct"]]  # локализовано, но ошибка ≥ порога
    print(f"\n{'='*64}\nКАЛИБРОВКА ({n} кадров, порог верности {correct_m:.0f} м)\n{'='*64}")

    print("\nпо датасетам (лок / верно / ЛОЖНЫХ / отказ):")
    for ds in sorted({r["dataset"] for r in rows}):
        sub = [r for r in rows if r["dataset"] == ds]
        sl = [r for r in sub if r["localized"]]
        sc = [r for r in sl if r["correct"]]
        sf = [r for r in sl if not r["correct"]]
        errs = [float(r["error_m"]) for r in sc]
        med = f", мед.ошибки {statistics.median(errs):.1f}м" if errs else ""
        print(f"  {ds:14} n={len(sub):3d}  лок={len(sl):3d}  верно={len(sc):3d}  ложных={len(sf):3d}  отказ={len(sub)-len(sl):3d}{med}")

    print(f"\nИТОГ: локализовано {len(loc)}/{n}, верно {len(correct)}, "
          f"**ЛОЖНЫХ срабатываний {len(false_pos)}** ({100*len(false_pos)/max(n,1):.1f}% от всех)")
    if false_pos:
        print("  ложные (локализовано, но далеко от GPS):")
        for r in sorted(false_pos, key=lambda r: -float(r["error_m"]))[:8]:
            print(f"    {r['dataset']}/{r['file']}: ошибка {r['error_m']}м, инл={r['n_inliers']}, "
                  f"conf={r['confidence']}, ncc={r['photometric_ncc']}, статус={r['status']}")

    # Порог сигнала, отделяющий верные от ложных СРЕДИ ЛОКАЛИЗОВАННЫХ.
    if correct and false_pos:
        for sig in ("n_inliers", "confidence", "photometric_ncc"):
            vals = [float(r[sig]) for r in loc if r[sig] != ""]
            labs = [bool(r["correct"]) for r in loc if r[sig] != ""]
            if len(set(labs)) < 2:
                continue
            thr = calibrate_uniqueness_threshold(vals, labs)
            keep = np.array(vals) >= thr
            y = np.array(labs)
            tpr = int((keep & y).sum()) / max(int(y.sum()), 1)
            fpr = int((keep & ~y).sum()) / max(int((~y).sum()), 1)
            print(f"\nпорог по '{sig}' ≥ {thr:.3f}: удержано верных {tpr:.0%}, пропущено ложных {fpr:.0%}")
    else:
        print("\nразделяющий порог не считается: нет одновременно верных и ложных срабатываний")

    # Reliability по confidence.
    if loc:
        print("\nreliability (доля верных по бинам confidence):")
        edges = [0.0, 0.3, 0.5, 0.7, 0.85, 1.01]
        for lo, hi in zip(edges, edges[1:]):
            b = [r for r in loc if lo <= r["confidence"] < hi]
            if b:
                acc = sum(r["correct"] for r in b) / len(b)
                print(f"  conf [{lo:.2f},{hi:.2f}): n={len(b):3d}  верных {acc:.0%}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="for_binding", help="каталог с датасетами")
    parser.add_argument("--matcher", default="lightglue", choices=["sift", "akaze", "lightglue", "loftr"])
    parser.add_argument("--cache", default="tiles")
    parser.add_argument("--sigma-m", type=float, default=25.0)
    parser.add_argument("--step", type=int, default=1, help="прореживание больших датасетов (каждый N-й)")
    parser.add_argument("--correct-m", type=float, default=30.0, help="порог ошибки для «верно»")
    parser.add_argument("--min-ncc", type=float, default=0.3,
                        help="порог NCC для LOCALIZED; для дрон↔спутник ~0 (NCC ненадёжен)")
    parser.add_argument("--out", default="calib.csv")
    args = parser.parse_args()

    print(f"матчер {args.matcher}, σ={args.sigma_m}м, шаг {args.step}, порог верности {args.correct_m}м")
    rows = collect(args)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report(rows, args.correct_m)
    print(f"\nсырьё → {args.out} ({len(rows)} строк)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
