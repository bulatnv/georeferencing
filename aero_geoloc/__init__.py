"""Аэро-геолокализация надирных снимков по картографической подложке.

Реализация ведётся по фазам, см. ``docs/PLAN.md``.
Готово: фаза 0 — геометрия Web Mercator и модель камеры.
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
from .types import LocalizationRequest, LocalizationResult, Prior, Status

__version__ = "0.1.0.dev0"

__all__ = [
    "Camera",
    "Georef",
    "LocalizationRequest",
    "LocalizationResult",
    "Prior",
    "Status",
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
