"""E2: геометрические сигналы позы — видно ли ложную позу, не глядя на яркости.

Второй эксперимент трека B ([ROADMAP.md](../docs/ROADMAP.md), фаза 1). E1 меряет
СОГЛАСИЕ картинок, здесь меряется САМОСОГЛАСОВАННОСТЬ позы — свойство, которое от
сезона не зависит по построению:

``inlier_spread``          насколько широко инлайеры разбросаны по кадру, в долях
                           его диагонали. Тридцать точек, сбитых в один угол,
                           подозрительны в любой сезон, а счётчик инлайеров этого
                           не видит вовсе.
``bootstrap_scatter_px``   насколько уезжает центр кадра, если пересобрать позу по
                           подвыборке инлайеров. Верная поза опирается на много
                           независимых точек и не шатается; ложная держится на
                           случайном совпадении и гуляет.

Откуда берутся ложные позы
--------------------------
Не из синтетики и не из логов: кадр матчится с окном подложки, **сдвинутым** на
заданное расстояние от истины. Всё, кроме места, у такой пары совпадает с верной —
тот же кадр, тот же сезон, тот же масштаб и поворот. Значит, разница в сигналах
объясняется только тем, что место чужое.

Порог инлайеров здесь намеренно ослаблен (``--min-inliers 4``): интересны как раз
**краевые** ложные позы, которые собрались вопреки текущей защите. Позы, не
собравшиеся вовсе, тоже записываются — это дешёвые негативы, и их доля сама по
себе показывает, сколько работы делает простой счётчик.

    python scripts/e2_geometry.py
    python scripts/e2_geometry.py --cases Saratov,Volgograd2 --matcher loftr

Ограничение, которое надо назвать вслух: на кросс-сезонных кейсах матчер не даёт
позы **ни на верном окне, ни на ложных**, поэтому эксперимент говорит о них
только одно — «сигналам не на чем работать, пока трек A не даст позу».
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from aero_geoloc.basemap import ESRI_WORLD_IMAGERY, TileBasemap, TileCache  # noqa: E402
from aero_geoloc.dataset import EvalCase, load_dataset  # noqa: E402
from aero_geoloc.geo import ground_mpp, haversine_m  # noqa: E402
from aero_geoloc.matcher import create_matcher  # noqa: E402
from aero_geoloc.oracle import alignment_for, north_up_crop, offset_lonlat, to_gray  # noqa: E402
from aero_geoloc.pose import (  # noqa: E402
    UNIFORM_SPREAD,
    bootstrap_center_scatter_px,
    estimate_similarity,
    inlier_spread,
)
from aero_geoloc.quality import aligned_ncc, aligned_structural  # noqa: E402
from aero_geoloc.similarity import dense_dino  # noqa: E402

FIELDS = ["case", "regime", "pair", "offset_m", "bearing_deg", "pose_found",
          "n_correspondences", "n_inliers", "inlier_ratio", "rmse_px", "ncc", "dino",
          "inlier_spread", "bootstrap_scatter_px", "certainty_mean", "certainty_cover",
          "centre_error_m", "size_px"]

#: Сигналы и направление «лучше»: у разброса центра меньше — лучше.
DIRECTION = {"n_inliers": +1, "inlier_ratio": +1, "rmse_px": -1, "ncc": +1, "dino": +1,
             "inlier_spread": +1, "bootstrap_scatter_px": -1,
             "certainty_mean": +1, "certainty_cover": +1}


def _rows_for_case(case: EvalCase, args, basemap, matcher, max_zoom, poses) -> list[dict]:
    align = alignment_for(case, poses, tolerance_m=args.pose_tolerance_m)
    if align is None:
        print("  пропуск: оракульную позу построить не из чего")
        return []

    z_fine = case.basemap_zoom(max_zoom=max_zoom)
    mpp = ground_mpp(case.prior.lat, z_fine)
    frame, _ = case.frame_at_mpp(mpp)
    query = north_up_crop(frame, align.yaw_deg)
    side = query.shape[0]
    centre = ((side - 1) / 2.0, (side - 1) / 2.0)
    diagonal = math.hypot(side, side)
    rows = []

    def measure(pair: str, lat: float, lon: float, offset_m: float, bearing: float):
        ref, georef = basemap(lon, lat, z_fine, side, side)
        gray_ref = to_gray(ref)
        row = {f: "" for f in FIELDS}
        row.update(case=case.name, regime=case.regime, pair=pair,
                   offset_m=round(offset_m), bearing_deg=round(bearing), size_px=side)

        corr = matcher.match(query, gray_ref)
        row["n_correspondences"] = len(corr)
        # Свидетельства уровня пары есть не у всех ядер — у разреженных их просто нет.
        for key, value in corr.evidence.items():
            if key in FIELDS:
                row[key] = round(float(value), 5)
        # Кадр уже приведён к масштабу и повороту подложки, поэтому ожидание
        # простое: масштаб около 1, поворот около 0. Ограничения оставлены
        # широкими — мы ИЩЕМ краевые ложные позы, а не отсекаем их.
        pose = estimate_similarity(
            corr, ransac_threshold_px=args.ransac_px, min_inliers=args.min_inliers,
            scale_bounds=(0.7, 1.4), expected_rotation_deg=0.0, rotation_tolerance_deg=25.0,
        ) if len(corr) >= 3 else None
        if pose is None:
            row["pose_found"] = 0
            return row

        mask = pose.inlier_mask
        row.update(
            pose_found=1,
            n_inliers=pose.n_inliers,
            inlier_ratio=round(pose.inlier_ratio, 4),
            rmse_px=round(pose.reprojection_rmse_px, 3),
            ncc=round(float(aligned_ncc(query, gray_ref, pose.transform)), 4),
            # Меры считаются НА НАЙДЕННОЙ ПОЗЕ, а не на оракульном выравнивании:
            # порог гейта применяется именно к этим числам, и калибровать его по
            # оракульным значениям значило бы калибровать не то (E3).
            dino=("" if args.no_dino else round(
                float(aligned_structural(query, gray_ref, pose.transform, dense_dino())), 4)),
            inlier_spread=round(inlier_spread(corr.pts_q[mask], diagonal), 4),
            bootstrap_scatter_px=round(
                bootstrap_center_scatter_px(corr, mask, centre, draws=args.draws), 4),
        )
        # Ошибка меряется НА ЗЕМЛЕ, а не в пикселях окна. Первая версия сравнивала
        # центр кадра с центром окна — и объявляла ложными все позы в сдвинутых
        # окнах, включая идеально верные: если окно сдвинуто на 150 м, верная поза
        # обязана положить центр кадра на 150 м от центра окна. Через Georef
        # двусмысленности нет: точка переводится в координаты и сверяется с истиной.
        px, py = pose.transform.apply(np.asarray(centre))
        found_lon, found_lat = georef.pixel_to_lonlat(float(px), float(py))
        row["centre_error_m"] = round(
            float(haversine_m(align.lat, align.lon, found_lat, found_lon)), 1)
        return row

    rows.append(measure("positive", align.lat, align.lon, 0.0, 0.0))
    for distance in args.offsets_m:
        for bearing in args.bearings:
            dlat, dlon = offset_lonlat(align.lat, align.lon, distance, bearing)
            rows.append(measure("negative", dlat, dlon, distance, bearing))
    return rows


def _label(row: dict, tolerance_m: float) -> str:
    """Верная поза или ложная — по РЕЗУЛЬТАТУ, а не по тому, какое окно подали.

    Ловушка, обнаруженная первым же прогоном: окно, сдвинутое на 150 м, при
    отпечатке кадра 240 м всё ещё **содержит** истинное место, и матчер честно
    находит там верное совмещение. Такая пара — не ложная поза, а верная поза в
    сдвинутом окне, и считать её негативом значит мерить не то: NCC у «ложных»
    выходил 0.442 против 0.444 у верных, то есть шум вместо сигнала.

    Поэтому метка ставится по тому, куда легла поза НА ЗЕМЛЕ: верная поза выводит
    центр кадра к истине независимо от того, какое окно ей подали.
    """
    if row["pose_found"] != 1:
        return "нет позы"
    return "верная" if float(row["centre_error_m"]) <= tolerance_m else "ложная"


def _auc(positives: list[float], negatives: list[float], direction: int) -> float:
    if not positives or not negatives:
        return float("nan")
    wins = sum(1.0 if direction * p > direction * n else 0.5 if p == n else 0.0
               for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def report(rows: list[dict], tolerance_m: float) -> None:
    width = 100
    print(f"\n{'='*width}\nE2: ГЕОМЕТРИЧЕСКИЕ СИГНАЛЫ на позах, собранных матчером\n{'='*width}")

    for row in rows:
        row["label"] = _label(row, tolerance_m)
    posed = [r for r in rows if r["label"] != "нет позы"]
    print(f"поз собрано: {len(posed)} из {len(rows)} пар; из них ВЕРНЫХ "
          f"{sum(r['label']=='верная' for r in posed)}, ЛОЖНЫХ "
          f"{sum(r['label']=='ложная' for r in posed)}")

    shifted = [r for r in rows if r["pair"] == "negative"]
    recovered = [r for r in shifted if r["label"] == "верная"]
    print(f"сдвинутых окон: {len(shifted)}; поза не собралась в "
          f"{sum(r['label']=='нет позы' for r in shifted)} "
          f"({100*sum(r['label']=='нет позы' for r in shifted)/max(len(shifted),1):.0f}%) — "
          f"столько делает сам порог инлайеров")
    print(f"из сдвинутых окон ВОССТАНОВЛЕНО верное место: {len(recovered)} — это не "
          f"ложные позы, а прямое доказательство того, ради чего сделано перекрытие "
          f"сетки индекса")

    for regime in sorted({r["regime"] for r in rows}):
        subset = [r for r in posed if r["regime"] == regime]
        pos = [r for r in subset if r["label"] == "верная"]
        neg = [r for r in subset if r["label"] == "ложная"]
        print(f"\n--- {regime}: {len(pos)} верных поз, {len(neg)} ложных ---")
        if not pos or not neg:
            print("    сравнивать не с чем — разделяющую способность не измерить")
            continue
        print(f"{'сигнал':<22}{'верные':>12}{'ложные':>12}{'AUC':>8}   вывод")
        for name, direction in DIRECTION.items():
            p = [float(r[name]) for r in pos if r[name] != "" and math.isfinite(float(r[name]))]
            n = [float(r[name]) for r in neg if r[name] != "" and math.isfinite(float(r[name]))]
            if not p or not n:
                continue
            auc = _auc(p, n, direction)
            verdict = ("разделяет идеально" if auc >= 0.999 else
                       "разделяет" if auc >= 0.9 else
                       "слабо" if auc >= 0.7 else "НЕ РАЗДЕЛЯЕТ")
            print(f"{name:<22}{statistics.median(p):>12.3f}{statistics.median(n):>12.3f}"
                  f"{auc:>8.3f}   {verdict}")
        print(f"    (ориентир: равномерный разброс инлайеров ≈ {UNIFORM_SPREAD:.2f})")

    print(f"\n{'='*width}\nЧТО ГОВОРЯТ СИГНАЛЫ О КОНКРЕТНЫХ ЛОЖНЫХ ПОЗАХ\n{'='*width}")
    wrong = sorted((r for r in posed if r["label"] == "ложная"),
                   key=lambda r: -float(r["n_inliers"]))
    print(f"{'кейс':<16}{'сдвиг':>8}{'инл':>6}{'spread':>9}{'scatter':>10}"
          f"{'NCC':>8}{'ошибка,м':>11}")
    for r in wrong[:12]:
        print(f"{r['case']:<16}{r['offset_m']:>8}{r['n_inliers']:>6}"
              f"{float(r['inlier_spread']):>9.3f}{float(r['bootstrap_scatter_px']):>10.3f}"
              f"{float(r['ncc']):>8.3f}{float(r['centre_error_m']):>11.0f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default="datasets/test_images.yaml")
    parser.add_argument("--cases", default="")
    parser.add_argument("--matcher", default="lightglue")
    parser.add_argument("--poses", default="eval_out/regress.csv")
    parser.add_argument("--pose-tolerance-m", type=float, default=150.0)
    parser.add_argument("--offsets", default="150,400,1000")
    parser.add_argument("--bearings", default="0,90,180,270")
    parser.add_argument("--min-inliers", type=int, default=4,
                        help="намеренно ниже боевого: нужны КРАЕВЫЕ ложные позы")
    parser.add_argument("--ransac-px", type=float, default=6.0)
    parser.add_argument("--draws", type=int, default=32, help="бутстрэп-выборок на позу")
    parser.add_argument("--no-dino", action="store_true", help="без плотных фич DINOv2")
    parser.add_argument("--cache", default="tiles")
    parser.add_argument("--out", default="eval_out/e2_geometry.csv")
    parser.add_argument("--correct-m", type=float, default=50.0,
                        help="поза считается верной, если её центр не дальше "
                             "стольких метров от истины")
    parser.add_argument("--from-csv", default="", help="только пересчитать отчёт")
    args = parser.parse_args()

    if args.from_csv:
        with open(args.from_csv, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for row in rows:
            row["pose_found"] = 1 if str(row["pose_found"]) == "1" else 0
        report(rows, args.correct_m)
        return 0

    args.offsets_m = [float(v) for v in args.offsets.split(",") if v.strip()]
    args.bearings = [float(v) for v in args.bearings.split(",") if v.strip()]

    dataset = load_dataset(args.manifest)
    cases = [c for c in dataset.cases if c.has_truth]
    if args.cases:
        wanted = {c.strip() for c in args.cases.split(",") if c.strip()}
        cases = [c for c in cases if c.name in wanted]

    basemap = TileBasemap(cache=TileCache(args.cache))
    matcher = create_matcher(args.matcher)
    max_zoom = ESRI_WORLD_IMAGERY.max_zoom
    poses = {}
    path = Path(args.poses)
    if path.exists():
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    poses[row["case"]] = (float(row["found_lat"]), float(row["found_lon"]),
                                          float(row["heading_deg"]))
                except (KeyError, ValueError):
                    continue

    print(f"E2: {len(cases)} кейсов, матчер {args.matcher}, сдвиги {args.offsets_m} м × "
          f"{len(args.bearings)} азимутов, min_inliers={args.min_inliers}")

    rows: list[dict] = []
    for case in cases:
        print(f"\n[{case.name}] {case.regime}", flush=True)
        t0 = time.perf_counter()
        try:
            case_rows = _rows_for_case(case, args, basemap, matcher, max_zoom, poses)
        except Exception as exc:  # noqa: BLE001
            print(f"  ОШИБКА: {type(exc).__name__}: {exc}")
            continue
        rows.extend(case_rows)
        if case_rows:
            posed = sum(r["pose_found"] == 1 for r in case_rows)
            head = case_rows[0]
            print(f"  {len(case_rows)} пар за {time.perf_counter()-t0:.0f} с, поз {posed}; "
                  f"верное окно: инлайеры={head['n_inliers']} spread={head['inlier_spread']} "
                  f"scatter={head['bootstrap_scatter_px']} ncc={head['ncc']}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    report(rows, args.correct_m)
    print(f"\nсырьё → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
