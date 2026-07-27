"""Локализация реальных бортовых снимков против подложки Esri — валидация фазы 4.

Для каждого снимка: EXIF/XMP → камера + приор (GPS как центр), приведение
масштаба к разрешению Esri, ``localize`` с обучаемым матчером и предповоротом,
ошибка против GPS. Настоящий appearance gap (борт ↔ спутник), которого нет на
синтетике.

    python scripts/localize_real.py --images test_images --matcher lightglue

Нужны: torch + матчер (``pip install torch``, ``pip install git+.../LightGlue``),
Pillow, сеть (тайлы Esri; кэшируются).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aero_geoloc.basemap import ESRI_WORLD_IMAGERY, TileBasemap, TileCache  # noqa: E402
from aero_geoloc.drone import basemap_zoom_for, frame_at_mpp, load_drone_shot  # noqa: E402
from aero_geoloc.geo import ground_mpp, haversine_m  # noqa: E402
from aero_geoloc.localize import localize  # noqa: E402
from aero_geoloc.matcher import create_matcher  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", default="test_images", help="каталог со снимками")
    parser.add_argument("--matcher", default="lightglue", choices=["sift", "akaze", "lightglue", "loftr"])
    parser.add_argument("--cache", default="tiles", help="каталог кэша тайлов")
    parser.add_argument("--sigma-m", type=float, default=25.0, help="σ приора позиции (GPS точен)")
    args = parser.parse_args()

    max_zoom = ESRI_WORLD_IMAGERY.max_zoom
    matcher = create_matcher(args.matcher)
    basemap = TileBasemap(cache=TileCache(args.cache))
    prerotate = args.matcher in ("lightglue", "loftr")  # обучаемые не инвариантны к повороту

    paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))
    print(f"матчер: {args.matcher}, предповорот: {prerotate}, снимков: {len(paths)}\n")
    for path in paths:
        try:
            shot = load_drone_shot(path)
        except ValueError as exc:
            print(f"  пропуск — {exc}")
            continue
        head = f"{path.name} [{shot.model}, {shot.altitude_m:.0f}м]"
        if not shot.is_nadir:
            print(f"  {head}\n    → пропуск: косой кадр {shot.pitch_from_nadir_deg:+.0f}° от надира "
                  "(вне надирной модели подобия)")
            continue
        z = basemap_zoom_for(shot, max_zoom=max_zoom)
        frame, camera = frame_at_mpp(shot, ground_mpp(shot.true_lat, z))
        result = localize(
            frame, camera, shot.prior(sigma_m=args.sigma_m), basemap,
            matcher=matcher, max_zoom=max_zoom, prerotate=prerotate,
            min_inliers=10, coarse_min_inliers=8, ransac_threshold_px=6.0,
        )
        if result.is_localized:
            err = haversine_m(shot.true_lat, shot.true_lon, result.center_lat, result.center_lon)
            print(f"  {head}\n    → {result.status.value}: ошибка vs GPS = {err:.1f} м, "
                  f"инлайеров={result.diagnostics.get('n_inliers')}, "
                  f"эллипс(1σ)={result.error_ellipse_m[0]:.1f} м, курс={result.heading_deg:.0f}°")
        else:
            print(f"  {head}\n    → {result.status.value}: {result.diagnostics.get('reason')} "
                  f"(честный отказ, не уверенно-неверная точка)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
