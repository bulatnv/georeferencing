"""Аэро-геолокализация надирных снимков по картографической подложке.

Реализация ведётся по фазам, см. ``docs/PLAN.md``.
Готово: фаза 0 (геометрия, камера) и фаза 1 (сквозной скелет на синтетике).

Пакет целиком требует OpenCV — он импортируется из :mod:`aero_geoloc.matcher`,
:mod:`aero_geoloc.pose`, :mod:`aero_geoloc.localize` и
:mod:`aero_geoloc.testbench`. Сами модули фундамента (:mod:`aero_geoloc.geo`,
:mod:`aero_geoloc.camera`, :mod:`aero_geoloc.types`) от OpenCV не зависят и
переносимы как есть, но импорт через пакет всё равно потянет cv2.
"""

from __future__ import annotations

from .basemap import (
    ESRI_WORLD_IMAGERY,
    MissingTileError,
    TileCache,
    TileProvider,
    fetch_basemap,
    get_provider,
    load_tile,
)
from .camera import Camera
from .geo import (
    EARTH_MEAN_RADIUS_M,
    EARTH_RADIUS_M,
    EQUATOR_MPP_Z0,
    MAX_LATITUDE,
    TILE_SIZE_PX,
    Georef,
    exact_zoom_for_mpp,
    ground_mpp,
    haversine_m,
    lonlat_to_world_px,
    world_px_to_lonlat,
    world_size_px,
    zoom_for_mpp,
)
from .localize import localize_against_reference, normalize_gray
from .matcher import AKAZEMatcher, Correspondences, Matcher, SIFTMatcher, create_matcher
from .pose import PoseEstimate, SimilarityTransform, estimate_similarity
from .types import LocalizationRequest, LocalizationResult, Prior, Status

__version__ = "0.2.0.dev0"

__all__ = [
    "AKAZEMatcher",
    "Camera",
    "Correspondences",
    "ESRI_WORLD_IMAGERY",
    "Georef",
    "MissingTileError",
    "TileCache",
    "TileProvider",
    "fetch_basemap",
    "get_provider",
    "load_tile",
    "LocalizationRequest",
    "LocalizationResult",
    "Matcher",
    "PoseEstimate",
    "Prior",
    "SIFTMatcher",
    "SimilarityTransform",
    "Status",
    "create_matcher",
    "estimate_similarity",
    "localize_against_reference",
    "normalize_gray",
    "EARTH_MEAN_RADIUS_M",
    "EARTH_RADIUS_M",
    "EQUATOR_MPP_Z0",
    "MAX_LATITUDE",
    "TILE_SIZE_PX",
    "exact_zoom_for_mpp",
    "ground_mpp",
    "haversine_m",
    "lonlat_to_world_px",
    "world_px_to_lonlat",
    "world_size_px",
    "zoom_for_mpp",
    "__version__",
]
