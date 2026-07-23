"""Оффлайн-индексация региона подложки в «хеш местности» (фаза 3, ``docs/RETRIEVAL.md``).

Строит :class:`~aero_geoloc.retrieval.TerrainIndex` над регионом через
:class:`~aero_geoloc.basemap.BasemapSource` и сохраняет его в ``.npz`` для рантайма
(``localize(..., index=TerrainIndex.load(path, encoder))``). Источник по умолчанию —
реальные тайлы Esri (нужна сеть/кэш); ``--synthetic`` строит по процедурной сцене
без сети (для демонстрации/тестов).

    python scripts/build_index.py --lat 55.7558 --lon 37.6173 --zoom 17 --extent-px 4096 --out index.npz
    python scripts/build_index.py --synthetic --out index.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aero_geoloc.basemap import TileBasemap, TileCache, fetch_basemap  # noqa: E402
from aero_geoloc.geo import Georef  # noqa: E402
from aero_geoloc.retrieval import AveragePoolEncoder, TerrainIndex  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lat", type=float, default=55.7558)
    parser.add_argument("--lon", type=float, default=37.6173)
    parser.add_argument("--zoom", type=int, default=17)
    parser.add_argument("--extent-px", type=int, default=4096, help="сторона региона в пикселях")
    parser.add_argument("--cell-size", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--rotations", type=int, default=1, help="число углов ротационной аугментации")
    parser.add_argument("--grid", type=int, default=24, help="сетка стенд-энкодера")
    parser.add_argument("--cache", default="tiles", help="каталог кэша тайлов")
    parser.add_argument("--synthetic", action="store_true", help="строить по процедурной сцене без сети")
    parser.add_argument("--out", default="index.npz")
    args = parser.parse_args()

    encoder = AveragePoolEncoder(grid=args.grid)
    rotations = tuple(i * 360.0 / args.rotations for i in range(args.rotations))

    if args.synthetic:
        from aero_geoloc.testbench import SceneBasemap, make_synthetic_scene

        scene = make_synthetic_scene(args.extent_px, center_lon=args.lon, center_lat=args.lat)
        basemap, region = SceneBasemap(scene), scene.georef
    else:
        cache = TileCache(args.cache)
        # Прогреваем кэш регионом одним запросом (иначе построение дёргало бы тайлы поштучно).
        fetch_basemap(args.lon, args.lat, args.zoom, args.extent_px, args.extent_px, cache=cache)
        basemap = TileBasemap(cache=cache)
        region = Georef(args.lon, args.lat, args.zoom, args.extent_px, args.extent_px)

    print(f"регион {args.extent_px}px @z{region.zoom}, клетка {args.cell_size}px, "
          f"overlap {args.overlap}, ротаций {args.rotations}")
    started = time.perf_counter()
    index = TerrainIndex(encoder).build(
        basemap, region, cell_size_px=args.cell_size, overlap=args.overlap, rotations_deg=rotations,
    )
    elapsed = time.perf_counter() - started

    index.save(args.out)
    size_mb = Path(args.out).stat().st_size / 1e6
    print(f"проиндексировано {len(index)} записей за {elapsed:.1f} с → {args.out} ({size_mb:.1f} МБ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
