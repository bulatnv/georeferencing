"""A/B ретривал-энкодеров Этажа 1 на едином стенде: DINOv2 (сырой) vs MegaLoc (VPR).

Меняется ТОЛЬКО ``Encoder`` — та же сцена, та же нарезка клеток, тот же prerotate.
Метрика — **ранг клетки, ближайшей к истине** в ранжировании ретривала (и top-1
до истины, и Recall@K). Ранг в хвосте = appearance gap Этажа 1; в топе = ядро
довело бы до Этажа 2. Ровно та диагностика, что вскрыла провал DRZ_19206 (506/2116)
на сыром DINOv2 — здесь она отвечает, поднимает ли MegaLoc верную клетку в top-K.

    python scripts/compare_encoders.py --image test_images/00049.JPG --offset-km 1 --bearing 210
    python scripts/compare_encoders.py --image for_binding/DRZ/DRZ_19206.JPG --offset-km 1 --bearing 120

Нужны: torch + DINOv2/MegaLoc (torch.hub), Pillow, сеть (тайлы Esri + веса).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aero_geoloc.basemap import ESRI_WORLD_IMAGERY, TileBasemap, TileCache  # noqa: E402
from aero_geoloc.drone import basemap_zoom_for, frame_at_mpp, load_drone_shot  # noqa: E402
from aero_geoloc.geo import Georef, ground_mpp, haversine_m, zoom_for_mpp  # noqa: E402
from aero_geoloc.localize import normalize_gray  # noqa: E402
from aero_geoloc.retrieval import DinoV2Encoder, MegaLocEncoder, TerrainIndex  # noqa: E402


def _offset_lonlat(lat, lon, distance_m, bearing_deg):
    d_north = distance_m * math.cos(math.radians(bearing_deg))
    d_east = distance_m * math.sin(math.radians(bearing_deg))
    lat2 = lat + d_north / 111320.0
    lon2 = lon + d_east / (111320.0 * math.cos(math.radians(lat)))
    return lat2, lon2


def _evaluate(name, encoder, basemap, region, cell_px, overlap, frame, shot):
    """Построить индекс данным энкодером и вернуть диагностику по истинной клетке."""
    t0 = time.perf_counter()
    index = TerrainIndex(encoder).build(
        basemap, region, cell_size_px=cell_px, overlap=overlap, rotations_deg=(0.0,)
    )
    build_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    rr = index.query(normalize_gray(frame), k=len(index), prerotate_deg=-shot.yaw_deg)
    query_s = time.perf_counter() - t0
    dists = [haversine_m(shot.true_lat, shot.true_lon, c.center_lat, c.center_lon) for c in rr.cells]
    nearest_rank = min(range(len(dists)), key=lambda i: dists[i])  # 0-based
    return {
        "name": name,
        "dim": encoder.dim,
        "n_cells": len(index),
        "true_cell_m": dists[nearest_rank],
        "true_cell_rank": nearest_rank + 1,  # 1-based для отчёта
        "top1_m": dists[0],
        "uniqueness": rr.uniqueness,
        "build_s": build_s,
        "query_s": query_s,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", required=True)
    parser.add_argument("--offset-km", type=float, default=1.0)
    parser.add_argument("--bearing", type=float, default=120.0)
    parser.add_argument("--index-mpp", type=float, default=0.37)
    parser.add_argument("--index-cell-m", type=float, default=125.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--margin-km", type=float, default=0.5)
    parser.add_argument("--dem", action="store_true")
    parser.add_argument("--declination", type=float, default=0.0)
    parser.add_argument("--cache", default="tiles")
    args = parser.parse_args()

    mz = ESRI_WORLD_IMAGERY.max_zoom
    shot = load_drone_shot(args.image, use_dem=args.dem, magnetic_declination_deg=args.declination)
    if not shot.is_nadir:
        print(f"кадр не надирный (наклон {shot.pitch_from_nadir_deg:.0f}°) — вне модели")
        return 1
    basemap = TileBasemap(cache=TileCache(args.cache))

    z_fine = basemap_zoom_for(shot, max_zoom=mz)
    frame, _camera = frame_at_mpp(shot, ground_mpp(shot.true_lat, z_fine))

    offset_m = args.offset_km * 1000.0
    prior_lat, prior_lon = _offset_lonlat(shot.true_lat, shot.true_lon, offset_m, args.bearing)
    z_index = zoom_for_mpp(args.index_mpp, prior_lat, max_zoom=mz)
    mpp_index = ground_mpp(prior_lat, z_index)
    cell_px = max(32, round(args.index_cell_m / mpp_index))
    radius_m = offset_m + args.margin_km * 1000.0
    region_px = int(2 * radius_m / mpp_index)
    region = Georef(prior_lon, prior_lat, z_index, region_px, region_px)

    print(f"{Path(args.image).name}: истина ({shot.true_lat:.5f},{shot.true_lon:.5f}), "
          f"приор сдвинут на {args.offset_km} км@{args.bearing:.0f}°")
    print(f"стенд: регион {region_px}px @z{z_index}, клетка {cell_px}px≈{cell_px*mpp_index:.0f}м, "
          f"prerotate=-{shot.yaw_deg:.0f}°\n")

    rows = []
    for name, enc in [("DINOv2 (сырой)", DinoV2Encoder()), ("MegaLoc (VPR)", MegaLocEncoder())]:
        r = _evaluate(name, enc, basemap, region, cell_px, args.overlap, frame, shot)
        rows.append(r)
        print(f"{r['name']:<16} dim={r['dim']:>5}  клеток={r['n_cells']:>4}  "
              f"ранг верной клетки={r['true_cell_rank']:>4}/{r['n_cells']}  "
              f"(она в {r['true_cell_m']:.0f}м)  top-1 в {r['top1_m']:.0f}м  "
              f"уник={r['uniqueness']:.3f}  [индекс {r['build_s']:.0f}с]")

    print("\nИТОГ:")
    base, meg = rows[0], rows[1]
    for r in rows:
        in_top5 = "✓ top-5" if r["true_cell_rank"] <= 5 else ("top-10" if r["true_cell_rank"] <= 10 else "в хвосте")
        print(f"  {r['name']:<16} верная клетка на {r['true_cell_rank']}/{r['n_cells']} → {in_top5}")
    if meg["true_cell_rank"] < base["true_cell_rank"]:
        print(f"  → MegaLoc поднял верную клетку с {base['true_cell_rank']} до {meg['true_cell_rank']} места")
    elif meg["true_cell_rank"] > base["true_cell_rank"]:
        print(f"  → MegaLoc опустил верную клетку с {base['true_cell_rank']} до {meg['true_cell_rank']} места")
    else:
        print("  → паритет по рангу верной клетки")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
