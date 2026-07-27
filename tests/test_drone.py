"""Тесты загрузки бортовых снимков и сквозной локализации на реальных данных.

Парсинг метаданных проверяется, если снимки на месте (они не в репозитории —
это чужие фото ~25 МБ). Сквозная локализация против Esri дополнительно требует
torch + LightGlue + сеть — как сетевые тесты, по умолчанию пропускается.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

IMAGES = Path(__file__).resolve().parents[1] / "test_images"
NADIR_IMG = IMAGES / "00049.JPG"
OBLIQUE_IMG = IMAGES / "DJI_0058.JPG"

_HAS_PIL = importlib.util.find_spec("PIL") is not None
_HAS_LIGHTGLUE = importlib.util.find_spec("lightglue") is not None
_NET = os.environ.get("AERO_GEOLOC_NETWORK_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not (NADIR_IMG.exists() and _HAS_PIL),
    reason="нет test_images/00049.JPG или Pillow — реальные данные недоступны",
)


def test_load_drone_shot_parses_metadata():
    from aero_geoloc.drone import load_drone_shot

    shot = load_drone_shot(NADIR_IMG)
    assert shot.true_lat == pytest.approx(51.2158, abs=1e-3)
    assert shot.true_lon == pytest.approx(6.1676, abs=1e-3)
    assert shot.altitude_m == pytest.approx(89.9, abs=1.0)
    assert shot.yaw_deg == pytest.approx(-97.5, abs=1.0)
    assert shot.pitch_from_nadir_deg == pytest.approx(3.8, abs=1.0)  # gimbal −86.2° + 90°
    assert shot.model == "FC220"
    assert shot.is_nadir
    assert shot.image_bgr.shape[2] == 3


def test_prior_centres_on_gps():
    from aero_geoloc.drone import load_drone_shot

    shot = load_drone_shot(NADIR_IMG)
    prior = shot.prior(sigma_m=30.0)
    assert (prior.lat, prior.lon) == (shot.true_lat, shot.true_lon)
    assert prior.yaw_deg == shot.yaw_deg
    assert prior.sigma_m == 30.0


def test_frame_at_mpp_resamples_to_target_resolution():
    from aero_geoloc.drone import frame_at_mpp, load_drone_shot

    shot = load_drone_shot(NADIR_IMG)
    target = 0.2
    frame, camera = frame_at_mpp(shot, target)
    assert camera.gsd(shot.altitude_m) == pytest.approx(target, rel=0.01)  # GSD ≈ mpp подложки
    assert frame.shape[1] < shot.image_bgr.shape[1]  # кадр уменьшен (детальнее подложки)


@pytest.mark.skipif(not OBLIQUE_IMG.exists(), reason="нет DJI_0058.JPG")
def test_oblique_shot_flagged_non_nadir():
    from aero_geoloc.drone import load_drone_shot

    shot = load_drone_shot(OBLIQUE_IMG)
    assert not shot.is_nadir  # gimbal pitch 0° = горизонт, ~90° от надира


@pytest.mark.skipif(
    not (_HAS_LIGHTGLUE and _NET),
    reason="нужны LightGlue + сеть (AERO_GEOLOC_NETWORK_TESTS=1)",
)
def test_localize_real_image_against_esri():
    """Сквозная локализация настоящего кадра против Esri через appearance gap."""
    from aero_geoloc.basemap import ESRI_WORLD_IMAGERY, TileBasemap, TileCache
    from aero_geoloc.drone import basemap_zoom_for, frame_at_mpp, load_drone_shot
    from aero_geoloc.geo import ground_mpp, haversine_m
    from aero_geoloc.localize import localize
    from aero_geoloc.matcher import LightGlueMatcher

    shot = load_drone_shot(NADIR_IMG)
    mz = ESRI_WORLD_IMAGERY.max_zoom
    z = basemap_zoom_for(shot, max_zoom=mz)
    frame, camera = frame_at_mpp(shot, ground_mpp(shot.true_lat, z))
    result = localize(
        frame, camera, shot.prior(sigma_m=25.0),
        TileBasemap(cache=TileCache("tiles")),
        matcher=LightGlueMatcher(), max_zoom=mz, prerotate=True,
        min_inliers=10, coarse_min_inliers=8, ransac_threshold_px=6.0,
    )
    assert result.is_localized
    err = haversine_m(shot.true_lat, shot.true_lon, result.center_lat, result.center_lon)
    assert err < 25.0  # георефа Esri + GPS + наклон ≈ единицы метров
