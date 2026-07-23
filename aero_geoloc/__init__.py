"""Аэро-геолокализация надирных снимков по картографической подложке.

Реализация ведётся по фазам, см. ``docs/PLAN.md``.
Готово: фаза 0 (геометрия, камера) и фаза 1 (сквозной скелет на синтетике).

Матчинг, pose, оркестрация и стенд требуют OpenCV и импортируются отсюда же;
:mod:`aero_geoloc.geo`, :mod:`aero_geoloc.camera` и :mod:`aero_geoloc.types`
остаются свободными от тяжёлых зависимостей и доступны напрямую.
"""

from __future__ import annotations

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
    "Georef",
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
