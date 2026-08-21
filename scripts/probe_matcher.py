"""Оракульная проба ядра: заведомо верное выравнивание → матч → поза → ошибка.

Инструмент перепроверки из ``docs/RESEARCH_A_ROMAV2_RECHECK.md`` §3. Раньше
оракульная проба была ad-hoc внутри ``e2_geometry.py``; здесь она вынесена в
воспроизводимый прогон, чьи числа сопоставимы с таблицей §3
``RESEARCH_A_RESULTS.md``: те же ``north_up_crop``, окно того же размера и
центра, тот же ``estimate_similarity`` с широкими границами.

Проба даёт матчеру **заведомо верное выравнивание** — она меряет ядро, а не
тракт (retrieval, окна, гейты в ней не участвуют).

    python scripts/probe_matcher.py --cases DRZ_00755,Ufa3,Ufa2 \\
        --matcher romav2 --min-conf 0.0 --model-threshold none \\
        --out eval_out/probe_romav2_raw.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aero_geoloc.basemap import ESRI_WORLD_IMAGERY, TileBasemap, TileCache  # noqa: E402
from aero_geoloc.dataset import load_dataset  # noqa: E402
from aero_geoloc.geo import ground_mpp, haversine_m  # noqa: E402
from aero_geoloc.matcher import create_matcher  # noqa: E402
from aero_geoloc.oracle import alignment_for, north_up_crop, offset_lonlat, to_gray  # noqa: E402
from aero_geoloc.pose import estimate_similarity  # noqa: E402

#: Колонки CSV — по спецификациям RECHECK §3 (RoMa v2) и LOFTR_RECHECK §3, плюс
#: контекст (source/size/max_side/пара), без которого строки не сравнить: именно
#: неразличимость строк разных режимов дважды портила замеры LoFTR.
FIELDS = [
    "case", "matcher", "pair", "min_conf", "model_threshold", "coarse_thr",
    "max_side", "size_px", "footprint_m", "align_source",
    "n_sampled", "n_model_out", "n_pairs_after_filter", "pose_found",
    "n_inliers", "err_m", "rmse_px",
    "loftr_coarse_thr",
    "conf_p10", "conf_p50", "conf_p90", "conf_p99", "conf_max",
    "conf_frac_gt020", "conf_frac_gt050",
    "overlap_p10", "overlap_p50", "overlap_p90", "overlap_p99",
    "overlap_frac_gt005", "overlap_frac_gt050", "overlap_mean",
    "overlap_field_p10", "overlap_field_p50", "overlap_field_p90",
    "overlap_field_p99", "overlap_field_frac_gt005", "overlap_field_frac_gt050",
    "certainty_mean", "certainty_cover",
    "precision_median", "precision_ba_median", "precision_ba_shape", "sec",
]


def load_poses(path: Path) -> dict[str, tuple[float, float, float]]:
    """Позы пайплайна для кейсов без EXIF-курса (оракул ``manual`` иначе слеп)."""
    poses: dict[str, tuple[float, float, float]] = {}
    if not path.exists():
        return poses
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                poses[row["case"]] = (float(row["found_lat"]), float(row["found_lon"]),
                                      float(row["heading_deg"]))
            except (KeyError, ValueError):
                continue
    return poses


def probe_case(case, align, basemap, matcher, args, max_zoom,
               *, pair="pos", centre=None) -> dict:
    """Одна строка CSV: матч на оракульном выравнивании и поза из него.

    ``pair="false"`` с центром ``centre`` — отрицательное плечо (Р4): то же
    окно, но сдвинутое; ошибка при этом всё равно меряется против истинного
    центра ``align`` — «нашёл верное место в сдвинутом окне» это не ошибка.
    """
    z_fine = case.basemap_zoom(max_zoom=max_zoom)
    mpp = ground_mpp(case.prior.lat, z_fine)
    frame, _ = case.frame_at_mpp(mpp)
    query = north_up_crop(frame, align.yaw_deg)
    side = query.shape[0]
    lat, lon = centre if centre is not None else (align.lat, align.lon)
    ref, georef = basemap(lon, lat, z_fine, side, side)
    gray_ref = to_gray(ref)

    started = time.perf_counter()
    corr = matcher.match(query, gray_ref)
    sec = time.perf_counter() - started

    row = {f: "" for f in FIELDS}
    row.update(case=case.name, matcher=args.matcher, pair=pair,
               min_conf=args.min_conf if args.min_conf is not None else "default",
               model_threshold=args.model_threshold or "default",
               coarse_thr=args.coarse_thr if args.coarse_thr is not None else "default",
               max_side=args.max_side or 0,
               size_px=side, footprint_m=round(side * mpp),
               align_source=align.source,
               n_pairs_after_filter=len(corr), pose_found=0, sec=round(sec, 2))
    for key, value in corr.evidence.items():
        if key in FIELDS and key != "precision_ba_shape":
            row[key] = round(float(value), 5)
    if "precision_ba_shape" in corr.evidence:
        row["precision_ba_shape"] = corr.evidence["precision_ba_shape"]

    # Границы те же, что в e2_geometry: кадр уже приведён к масштабу и повороту
    # подложки, ожидание — масштаб ~1, поворот ~0; границы оставлены широкими.
    pose = estimate_similarity(
        corr, ransac_threshold_px=args.ransac_px, min_inliers=args.min_inliers,
        scale_bounds=(0.7, 1.4), expected_rotation_deg=0.0,
        rotation_tolerance_deg=25.0,
    ) if len(corr) >= 3 else None
    if pose is None:
        return row

    centre = ((side - 1) / 2.0, (side - 1) / 2.0)
    cx, cy = pose.transform.apply([centre])[0]
    lon, lat = georef.pixel_to_lonlat(float(cx), float(cy))
    row.update(
        pose_found=1, n_inliers=pose.n_inliers,
        rmse_px=round(pose.reprojection_rmse_px, 3),
        err_m=round(haversine_m(align.lat, align.lon, lat, lon), 1),
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default="datasets/test_images.yaml")
    parser.add_argument("--cases", required=True, help="имена кейсов через запятую")
    parser.add_argument("--matcher", required=True)
    parser.add_argument("--min-conf", type=float, default=None,
                        help="переопределить порог conf ядра; None = дефолт ядра")
    parser.add_argument("--model-threshold", default=None,
                        help="только romav2: порог ур. 6 в модели — число либо "
                             "'none' (сырое поле); None = дефолт ядра")
    parser.add_argument("--coarse-thr", type=float, default=None,
                        help="только loftr/minima_loftr: внутренний порог kornia "
                             "coarse_matching.thr; None = дефолт пакета (0.2)")
    parser.add_argument("--max-side", type=int, default=0,
                        help="обёртка ResizedMatcher: колпак на длинную сторону. "
                             "Для LoFTR ОБЯЗАТЕЛЕН (исторические числа сняты при "
                             "640/1024; полное разрешение — известная ловушка)")
    parser.add_argument("--offset-m", type=float, default=0.0,
                        help="отрицательное плечо: тот же кейс на окне, сдвинутом "
                             "на столько метров (0 = выключено). Строка пишется, "
                             "только если сдвиг больше наземного следа окна")
    parser.add_argument("--min-inliers", type=int, default=6)
    parser.add_argument("--ransac-px", type=float, default=6.0)
    parser.add_argument("--poses", default="eval_out/eval.csv",
                        help="CSV с found_lat/found_lon/heading_deg для manual-кейсов")
    parser.add_argument("--pose-tolerance-m", type=float, default=150.0)
    parser.add_argument("--cache", default="tiles")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    dataset = load_dataset(args.manifest)
    wanted = [n.strip() for n in args.cases.split(",") if n.strip()]
    cases = [dataset.by_name(n) for n in wanted]

    kwargs = {}
    if args.min_conf is not None:
        kwargs["min_conf"] = args.min_conf
    if args.model_threshold is not None:
        if args.matcher != "romav2":
            parser.error("--model-threshold применим только к romav2")
        kwargs["model_threshold"] = (None if args.model_threshold.lower() == "none"
                                     else float(args.model_threshold))
    if args.coarse_thr is not None:
        if args.matcher not in ("loftr", "minima_loftr"):
            parser.error("--coarse-thr применим только к loftr/minima_loftr")
        kwargs["coarse_thr"] = args.coarse_thr
    if args.max_side:
        kwargs["max_side"] = args.max_side
    matcher = create_matcher(args.matcher, **kwargs)

    basemap = TileBasemap(cache=TileCache(args.cache))
    max_zoom = ESRI_WORLD_IMAGERY.max_zoom
    poses = load_poses(Path(args.poses))

    rows = []
    for case in cases:
        align = alignment_for(case, poses, tolerance_m=args.pose_tolerance_m)
        if align is None:
            print(f"[{case.name}] пропуск: оракульную позу построить не из чего")
            continue
        row = probe_case(case, align, basemap, matcher, args, max_zoom)
        rows.append(row)
        print(f"[{case.name}] пар={row['n_pairs_after_filter']} "
              f"поза={'да' if row['pose_found'] else 'НЕТ'} "
              f"инл={row['n_inliers'] or '—'} err={row['err_m'] or '—'} м "
              f"({row['sec']} с)")
        if args.offset_m > 0:
            # Р4: на крупных кадрах сдвиг меньше следа окна «ложным» не является —
            # окно накрыло бы верное место, и строка только путала бы анализ.
            if args.offset_m <= float(row["footprint_m"]):
                print(f"[{case.name}] false-плечо пропущено: сдвиг {args.offset_m:.0f} м "
                      f"не больше следа окна {row['footprint_m']} м")
            else:
                centre = offset_lonlat(align.lat, align.lon, args.offset_m, 45.0)
                frow = probe_case(case, align, basemap, matcher, args, max_zoom,
                                  pair="false", centre=centre)
                rows.append(frow)
                print(f"[{case.name}·false] пар={frow['n_pairs_after_filter']} "
                      f"инл={frow['n_inliers'] or '—'}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
