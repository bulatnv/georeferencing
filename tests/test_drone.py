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

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "test_images"
NADIR_IMG = IMAGES / "00049.JPG"
OBLIQUE_IMG = IMAGES / "DJI_0058.JPG"


def _first_jpg(directory: Path) -> Path | None:
    return next(iter(sorted(directory.glob("*.JPG"))), None) if directory.exists() else None


SENSEFLY = _first_jpg(ROOT / "for_binding" / "images")  # survey: Camera:Yaw/Pitch/Roll, только абс. высота

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


# --- survey-камеры (senseFly/S.O.D.A.: Camera:Yaw/Pitch/Roll, только абс. высота) ---


@pytest.mark.skipif(SENSEFLY is None or not _HAS_PIL, reason="нет survey-снимка для теста")
def test_survey_camera_metadata_and_altitude_override():
    """Survey-камера: курс/наклон из Camera:*; высота — из явного override."""
    from aero_geoloc.drone import load_drone_shot

    shot = load_drone_shot(SENSEFLY, altitude_override_m=150.0)
    assert shot.altitude_m == 150.0  # override, т.к. в XMP нет AGL
    assert "senseFly" in shot.model
    assert isinstance(shot.yaw_deg, float)  # Camera:Yaw
    assert isinstance(shot.pitch_from_nadir_deg, float) and isinstance(shot.roll_deg, float)


@pytest.mark.skipif(SENSEFLY is None or not _HAS_PIL, reason="нет survey-снимка для теста")
def test_survey_without_ground_elevation_raises():
    """У survey-камеры нет AGL в XMP — без рельефа/override загрузчик честно падает."""
    from aero_geoloc.drone import load_drone_shot

    with pytest.raises(ValueError, match="высот"):
        load_drone_shot(SENSEFLY)


@pytest.mark.skipif(
    SENSEFLY is None or not _NET, reason="нужны survey-снимок + сеть (DEM)",
)
def test_survey_agl_from_dem():
    """AGL = абс. высота (EXIF) − рельеф (DEM) даёт разумную survey-высоту."""
    from aero_geoloc.drone import load_drone_shot, lookup_ground_elevation
    from PIL import Image, ExifTags

    g = Image.open(SENSEFLY).getexif().get_ifd(ExifTags.IFD.GPSInfo)
    lat = float(g[2][0]) + float(g[2][1]) / 60 + float(g[2][2]) / 3600
    lon = float(g[4][0]) + float(g[4][1]) / 60 + float(g[4][2]) / 3600
    shot = load_drone_shot(SENSEFLY, ground_elevation_m=lookup_ground_elevation(lat, lon))
    assert 50.0 < shot.altitude_m < 400.0  # правдоподобная высота полёта survey


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


@pytest.mark.skipif(
    not (importlib.util.find_spec("torch") and _NET),
    reason="нужны torch (DINOv2 через torch.hub) + сеть",
)
def test_dinov2_retrieval_finds_real_region():
    """DINOv2-индекс над реальным регионом Esri находит место кадра через season gap.

    Стенд-энкодер (AveragePool) на настоящем appearance gap промахивается —
    DINOv2 опознаёт клетку. Проверяем именно это (Recall), сигнал уникальности
    на почти-однородном пригороде слаб и здесь не утверждается.
    """
    from aero_geoloc.basemap import ESRI_WORLD_IMAGERY, TileBasemap, TileCache
    from aero_geoloc.drone import basemap_zoom_for, frame_at_mpp, load_drone_shot
    from aero_geoloc.geo import Georef, ground_mpp, haversine_m
    from aero_geoloc.localize import normalize_gray
    from aero_geoloc.retrieval import DinoV2Encoder, TerrainIndex

    shot = load_drone_shot(NADIR_IMG)
    z = basemap_zoom_for(shot, max_zoom=ESRI_WORLD_IMAGERY.max_zoom)
    mpp = ground_mpp(shot.true_lat, z)
    frame, _ = frame_at_mpp(shot, mpp)

    cell_px, region_px = 640, 1920  # GPS попадает в центр клетки сетки
    region = Georef(shot.true_lon, shot.true_lat, z, region_px, region_px)
    index = TerrainIndex(DinoV2Encoder()).build(
        TileBasemap(cache=TileCache("tiles")), region,
        cell_size_px=cell_px, overlap=0.5, rotations_deg=(0.0,),
    )
    result = index.query(normalize_gray(frame), k=3, prerotate_deg=-shot.yaw_deg)
    radius = cell_px * mpp * 0.6
    distances = [
        haversine_m(shot.true_lat, shot.true_lon, c.center_lat, c.center_lon) for c in result.cells
    ]
    assert min(distances) <= radius  # верная клетка в top-3 несмотря на разрыв сезонов


@pytest.mark.skipif(
    not (_HAS_LIGHTGLUE and importlib.util.find_spec("torch") and _NET),
    reason="нужны torch + LightGlue + сеть",
)
def test_two_floors_retrieval_plus_lightglue_on_real_image():
    """Оба этажа на реальных данных: DINOv2 retrieval (ГДЕ примерно) → LightGlue (ГДЕ точно).

    Умеренный приор (σ=150 м) — кандидата даёт индекс, а не центр приора; точный
    уровень уточняет его LightGlue. Так проверяется вся связка на настоящем gap.
    """
    from aero_geoloc.basemap import ESRI_WORLD_IMAGERY, TileBasemap, TileCache
    from aero_geoloc.drone import basemap_zoom_for, frame_at_mpp, load_drone_shot
    from aero_geoloc.geo import Georef, ground_mpp, haversine_m
    from aero_geoloc.localize import localize
    from aero_geoloc.matcher import LightGlueMatcher
    from aero_geoloc.retrieval import DinoV2Encoder, TerrainIndex

    shot = load_drone_shot(NADIR_IMG)
    mz = ESRI_WORLD_IMAGERY.max_zoom
    z = basemap_zoom_for(shot, max_zoom=mz)
    frame, camera = frame_at_mpp(shot, ground_mpp(shot.true_lat, z))
    basemap = TileBasemap(cache=TileCache("tiles"))

    region = Georef(shot.true_lon, shot.true_lat, z, 1920, 1920)
    index = TerrainIndex(DinoV2Encoder()).build(
        basemap, region, cell_size_px=640, overlap=0.5, rotations_deg=(0.0,)
    )
    result = localize(
        frame, camera, shot.prior(sigma_m=150.0), basemap,
        index=index, matcher=LightGlueMatcher(), prerotate=True, max_zoom=mz,
        min_inliers=10, ransac_threshold_px=6.0,
    )
    assert result.diagnostics["retrieval"] is True  # грубый уровень — это DINOv2-индекс
    assert result.is_localized
    err = haversine_m(shot.true_lat, shot.true_lon, result.center_lat, result.center_lon)
    assert err < 25.0


@pytest.mark.skipif(
    not (_HAS_LIGHTGLUE and importlib.util.find_spec("torch") and _NET),
    reason="нужны torch + LightGlue + сеть",
)
def test_wide_prior_localization_from_augmented_prior():
    """Исходная задача: локализация из ГРУБОГО приора (сдвинут на 600 м) через retrieval.

    GPS используется только как истина; приор искусственно сдвинут — как в бою,
    где точного положения нет. DINOv2-индекс региона схлопывает диск в кандидатов,
    LightGlue уточняет позу. Мультигипотезность (top-K) ловит верную клетку, даже
    если top-1 — ложное совпадение на самоподобной местности.
    """
    import math

    from aero_geoloc.basemap import ESRI_WORLD_IMAGERY, TileBasemap, TileCache
    from aero_geoloc.drone import basemap_zoom_for, frame_at_mpp, load_drone_shot
    from aero_geoloc.geo import Georef, ground_mpp, haversine_m, zoom_for_mpp
    from aero_geoloc.localize import localize
    from aero_geoloc.matcher import LightGlueMatcher
    from aero_geoloc.retrieval import DinoV2Encoder, TerrainIndex
    from aero_geoloc.types import Prior

    shot = load_drone_shot(NADIR_IMG)
    mz = ESRI_WORLD_IMAGERY.max_zoom
    z_fine = basemap_zoom_for(shot, max_zoom=mz)
    frame, camera = frame_at_mpp(shot, ground_mpp(shot.true_lat, z_fine))
    basemap = TileBasemap(cache=TileCache("tiles"))

    # Приор сдвинут на 600 м от истины (грубая область вместо точного GPS).
    offset_m, bearing = 600.0, 60.0
    prior_lat = shot.true_lat + offset_m * math.cos(math.radians(bearing)) / 111320.0
    prior_lon = shot.true_lon + offset_m * math.sin(math.radians(bearing)) / (
        111320.0 * math.cos(math.radians(shot.true_lat))
    )
    prior = Prior(lat=prior_lat, lon=prior_lon, sigma_m=900.0, altitude_m=shot.altitude_m,
                  altitude_sigma_m=20.0, yaw_deg=shot.yaw_deg, pitch_deg=shot.pitch_from_nadir_deg)

    # Индекс региона вокруг приора: грубый зум, клетка ≈ footprint кадра (125 м).
    z_index = zoom_for_mpp(0.37, prior_lat, max_zoom=mz)
    mpp_index = ground_mpp(prior_lat, z_index)
    cell_px = round(125.0 / mpp_index)
    region_px = int(2 * (offset_m + 300.0) / mpp_index)
    region = Georef(prior_lon, prior_lat, z_index, region_px, region_px)
    index = TerrainIndex(DinoV2Encoder()).build(
        basemap, region, cell_size_px=cell_px, overlap=0.5, rotations_deg=(0.0,)
    )

    result = localize(frame, camera, prior, basemap, index=index, matcher=LightGlueMatcher(),
                      prerotate=True, max_zoom=mz, min_ncc=0.05, min_inliers=10, ransac_threshold_px=6.0)
    assert result.is_localized
    err = haversine_m(shot.true_lat, shot.true_lon, result.center_lat, result.center_lon)
    assert err < 30.0  # позиция восстановлена, хотя приор был в 600 м
