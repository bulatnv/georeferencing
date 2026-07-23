"""Тесты геометрии Web Mercator и Georef (фаза 0, ``docs/PLAN.md``).

Всё офлайн: ни сети, ни матчинга. Ошибка на этом слое отравит всё выше,
поэтому проверки идут против аналитических эталонов, а не против «как
получилось».
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aero_geoloc.geo import (
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

# Разнородные точки: экватор, средние широты, юг, отрицательная долгота.
SAMPLE_POINTS = [
    (0.0, 0.0),  # Null Island
    (37.6173, 55.7558),  # Москва
    (-0.1276, 51.5074),  # Лондон
    (-74.0060, 40.7128),  # Нью-Йорк
    (151.2093, -33.8688),  # Сидней
    (179.9, -84.0),  # у самого края проекции
]


# --- базовая проекция -------------------------------------------------------


def test_world_size_matches_tile_pyramid():
    assert world_size_px(0) == TILE_SIZE_PX
    assert world_size_px(10) == TILE_SIZE_PX * 1024


@pytest.mark.parametrize("zoom", [0, 1, 10, 18, 22])
@pytest.mark.parametrize("lon,lat", SAMPLE_POINTS)
def test_lonlat_world_px_roundtrip(lon, lat, zoom):
    """lonlat → pixel → lonlat совпадает до 1e-6° (критерий приёмки фазы 0)."""
    x, y = lonlat_to_world_px(lon, lat, zoom)
    lon_back, lat_back = world_px_to_lonlat(x, y, zoom)
    assert lon_back == pytest.approx(lon, abs=1e-9)
    assert lat_back == pytest.approx(lat, abs=1e-9)


def test_world_px_anchor_points():
    """Опорные точки пирамиды на zoom 0, считаются вручную."""
    # Левый верхний угол мира.
    assert lonlat_to_world_px(-180.0, MAX_LATITUDE, 0) == pytest.approx((0.0, 0.0), abs=1e-6)
    # Центр мира.
    assert lonlat_to_world_px(0.0, 0.0, 0) == pytest.approx((128.0, 128.0), abs=1e-12)
    # Правый нижний угол.
    assert lonlat_to_world_px(180.0, -MAX_LATITUDE, 0) == pytest.approx(
        (256.0, 256.0), abs=1e-6
    )


def test_latitude_is_clamped_beyond_mercator_limit():
    y_limit = lonlat_to_world_px(0.0, MAX_LATITUDE, 5)[1]
    y_beyond = lonlat_to_world_px(0.0, 89.9, 5)[1]
    assert y_beyond == pytest.approx(y_limit, abs=1e-9)


def test_projection_accepts_arrays():
    lons = np.array([0.0, 37.6173, -74.0060])
    lats = np.array([0.0, 55.7558, 40.7128])
    x, y = lonlat_to_world_px(lons, lats, 12)
    assert x.shape == lons.shape and y.shape == lats.shape
    for i, (lon, lat) in enumerate(zip(lons, lats)):
        xs, ys = lonlat_to_world_px(float(lon), float(lat), 12)
        assert x[i] == pytest.approx(xs)
        assert y[i] == pytest.approx(ys)


def test_zoom_step_doubles_world():
    """Смена зума на 1 ровно удваивает координаты — свойство пирамиды."""
    x10, y10 = lonlat_to_world_px(37.6173, 55.7558, 10)
    x11, y11 = lonlat_to_world_px(37.6173, 55.7558, 11)
    assert x11 == pytest.approx(2.0 * x10, rel=1e-12)
    assert y11 == pytest.approx(2.0 * y10, rel=1e-12)


# --- разрешение подложки ----------------------------------------------------


@pytest.mark.parametrize(
    "zoom,expected_mpp",
    [
        # Эталонная таблица Web Mercator на экваторе: 156543.034 / 2^z.
        (0, 156543.03392804097),
        (1, 78271.51696402048),
        (10, 152.8740565703525),
        (15, 4.777314267823516),
        (18, 0.5971642834779395),
        (19, 0.29858214173896974),
        (20, 0.14929107086948487),
    ],
)
def test_ground_mpp_reference_table_at_equator(zoom, expected_mpp):
    assert ground_mpp(0.0, zoom) == pytest.approx(expected_mpp, rel=1e-12)


@pytest.mark.parametrize("lat", [0.0, 30.0, 55.7558, 60.0, -45.0])
def test_ground_mpp_scales_with_cos_latitude(lat):
    assert ground_mpp(lat, 17) == pytest.approx(
        ground_mpp(0.0, 17) * math.cos(math.radians(lat)), rel=1e-12
    )


def test_ground_mpp_at_lat60_is_half_of_equator():
    """cos(60°) = 0.5 — удобный эталон «на глаз»."""
    assert ground_mpp(60.0, 18) == pytest.approx(ground_mpp(0.0, 18) / 2.0, rel=1e-12)


def test_equator_mpp_z0_equals_circumference_over_tile():
    assert EQUATOR_MPP_Z0 * TILE_SIZE_PX == pytest.approx(40075016.6856, abs=1e-3)


# --- подбор зума ------------------------------------------------------------


@pytest.mark.parametrize("target_mpp", [0.2, 0.5, 1.0, 2.4, 10.0])
@pytest.mark.parametrize("lat", [0.0, 55.7558, -33.8688])
def test_exact_zoom_for_mpp_is_inverse_of_ground_mpp(target_mpp, lat):
    z = exact_zoom_for_mpp(target_mpp, lat)
    assert ground_mpp(lat, z) == pytest.approx(target_mpp, rel=1e-12)


@pytest.mark.parametrize("mode", ["nearest", "finer", "coarser"])
def test_zoom_for_mpp_is_within_one_level_of_exact(mode):
    lat, target = 55.7558, 0.21
    exact = exact_zoom_for_mpp(target, lat)
    z = zoom_for_mpp(target, lat, mode=mode)
    assert abs(z - exact) <= 1.0
    assert isinstance(z, int)


def test_zoom_for_mpp_modes_bracket_the_target():
    lat, target = 55.7558, 0.21
    finer = zoom_for_mpp(target, lat, mode="finer")
    coarser = zoom_for_mpp(target, lat, mode="coarser")
    assert ground_mpp(lat, finer) <= target <= ground_mpp(lat, coarser)


def test_zoom_for_mpp_clamps_to_limits():
    assert zoom_for_mpp(1e-9, 0.0, max_zoom=22) == 22
    assert zoom_for_mpp(1e9, 0.0, min_zoom=0) == 0


def test_zoom_for_mpp_rejects_bad_input():
    with pytest.raises(ValueError):
        exact_zoom_for_mpp(0.0, 55.0)
    with pytest.raises(ValueError):
        zoom_for_mpp(0.5, 55.0, mode="whatever")


# --- haversine --------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b,expected_km,tol_km",
    [
        # Известные расстояния между городами (по большому кругу).
        ((55.7558, 37.6173), (59.9311, 30.3609), 634.0, 5.0),  # Москва — СПб
        ((51.5074, -0.1278), (48.8566, 2.3522), 343.5, 3.0),  # Лондон — Париж
        ((40.7128, -74.0060), (51.5074, -0.1278), 5570.0, 30.0),  # Нью-Йорк — Лондон
        ((-33.8688, 151.2093), (-37.8136, 144.9631), 713.0, 6.0),  # Сидней — Мельбурн
    ],
)
def test_haversine_against_known_city_distances(a, b, expected_km, tol_km):
    d_km = haversine_m(a[0], a[1], b[0], b[1]) / 1000.0
    assert d_km == pytest.approx(expected_km, abs=tol_km)


def test_haversine_one_degree_of_latitude():
    """1° широты ≈ 111.19 км на сферической модели."""
    assert haversine_m(0.0, 0.0, 1.0, 0.0) == pytest.approx(111194.9, abs=1.0)


def test_haversine_quarter_circumference():
    """Экватор → полюс = четверть окружности средней сферы WGS84."""
    assert haversine_m(0.0, 0.0, 90.0, 0.0) == pytest.approx(
        EARTH_MEAN_RADIUS_M * math.pi / 2.0, rel=1e-12
    )


def test_haversine_is_symmetric_and_zero_at_same_point():
    assert haversine_m(55.0, 37.0, 55.0, 37.0) == pytest.approx(0.0, abs=1e-9)
    assert haversine_m(55.0, 37.0, 56.0, 38.0) == pytest.approx(
        haversine_m(56.0, 38.0, 55.0, 37.0), rel=1e-12
    )


def test_haversine_broadcasts():
    lats = np.array([55.0, 56.0, 57.0])
    d = haversine_m(lats, 37.0, 55.0, 37.0)
    assert d.shape == lats.shape
    assert d[0] == pytest.approx(0.0, abs=1e-9)
    assert d[2] > d[1] > 0.0


# --- Georef -----------------------------------------------------------------


def make_georef(zoom: int = 18) -> Georef:
    return Georef(center_lon=37.6173, center_lat=55.7558, zoom=zoom, width=1024, height=768)


def test_georef_center_pixel_maps_to_center_lonlat():
    g = make_georef()
    cx, cy = g.center_pixel
    lon, lat = g.pixel_to_lonlat(cx, cy)
    assert lon == pytest.approx(g.center_lon, abs=1e-12)
    assert lat == pytest.approx(g.center_lat, abs=1e-12)


@pytest.mark.parametrize("zoom", [12, 16, 18, 20])
@pytest.mark.parametrize("px,py", [(0.0, 0.0), (511.5, 383.5), (1023.0, 767.0), (123.25, 45.75)])
def test_georef_pixel_lonlat_roundtrip(px, py, zoom):
    """pixel → lonlat → pixel точно до сильно меньше пикселя (критерий приёмки)."""
    g = make_georef(zoom)
    lon, lat = g.pixel_to_lonlat(px, py)
    px_back, py_back = g.lonlat_to_pixel(lon, lat)
    assert px_back == pytest.approx(px, abs=1e-6)
    assert py_back == pytest.approx(py, abs=1e-6)


def test_georef_lonlat_pixel_roundtrip():
    g = make_georef(17)
    for lon, lat in [(37.6173, 55.7558), (37.6200, 55.7570), (37.6100, 55.7500)]:
        px, py = g.lonlat_to_pixel(lon, lat)
        lon_back, lat_back = g.pixel_to_lonlat(px, py)
        assert lon_back == pytest.approx(lon, abs=1e-9)
        assert lat_back == pytest.approx(lat, abs=1e-9)


def test_georef_axes_orientation():
    """+x — на восток, +y — на юг (север вверх)."""
    g = make_georef()
    cx, cy = g.center_pixel
    lon_e, _ = g.pixel_to_lonlat(cx + 100.0, cy)
    _, lat_s = g.pixel_to_lonlat(cx, cy + 100.0)
    assert lon_e > g.center_lon
    assert lat_s < g.center_lat


#: ground_mpp живёт на сфере Web Mercator, haversine — на средней сфере WGS84.
#: См. «Две сферы, и это не баг» в docstring aero_geoloc.geo.
RADIUS_CONVENTION_RATIO = EARTH_MEAN_RADIUS_M / EARTH_RADIUS_M


def test_ground_mpp_and_haversine_differ_only_by_radius_convention():
    """Расхождение двух метрик — РОВНО отношение радиусов, и ничего больше.

    Тест страхует от двух разных ошибок сразу: если поедет соглашение о
    полупикселе или масштаб Georef — отношение перестанет быть константой;
    если кто-то «починит» расхождение, подогнав радиус, — тест это покажет.
    """
    assert RADIUS_CONVENTION_RATIO == pytest.approx(0.99888, abs=1e-5)

    for zoom in (14, 16, 18, 20):
        g = make_georef(zoom)
        cx, cy = g.center_pixel
        lon0, lat0 = g.pixel_to_lonlat(cx, cy)
        lon1, lat1 = g.pixel_to_lonlat(cx + 1.0, cy)
        step_m = haversine_m(lat0, lon0, lat1, lon1)
        assert step_m / g.mpp == pytest.approx(RADIUS_CONVENTION_RATIO, rel=1e-6)


def test_georef_pixel_step_equals_ground_mpp():
    """Сквозная сверка: шаг в 1 пиксель на земле = ground_mpp (в пределах 0.5%).

    Связывает Georef, ground_mpp и haversine — если разъедутся соглашения о
    полупикселе или о масштабе, тест это поймает.
    """
    g = make_georef(18)
    cx, cy = g.center_pixel
    lon0, lat0 = g.pixel_to_lonlat(cx, cy)
    lon1, lat1 = g.pixel_to_lonlat(cx + 1.0, cy)
    step_m = haversine_m(lat0, lon0, lat1, lon1)
    assert step_m == pytest.approx(g.mpp, rel=5e-3)


def test_georef_footprint_width_matches_mpp():
    """Ширина растра в метрах = width · mpp с точностью до конвенции радиусов."""
    g = make_georef(18)
    cy = g.center_pixel[1]
    lon_w, lat_w = g.pixel_to_lonlat(0.0, cy)
    lon_e, lat_e = g.pixel_to_lonlat(g.width - 1.0, cy)
    width_m = haversine_m(lat_w, lon_w, lat_e, lon_e)
    expected = (g.width - 1) * g.mpp * RADIUS_CONVENTION_RATIO
    assert width_m == pytest.approx(expected, rel=1e-6)


def test_georef_origin_is_corner_not_pixel_center():
    """Локальный (0,0) — центр первого пикселя, т.е. на полпикселя внутрь от угла."""
    g = make_georef()
    ox, oy = g.origin_world_px
    wx, wy = g.pixel_to_world_px(0.0, 0.0)
    assert wx == pytest.approx(ox + 0.5)
    assert wy == pytest.approx(oy + 0.5)


def test_georef_bounds_are_ordered_and_contain_center():
    g = make_georef(16)
    west, south, east, north = g.bounds()
    assert west < g.center_lon < east
    assert south < g.center_lat < north


def test_georef_bounds_width_matches_raster_size():
    g = make_georef(16)
    west, south, east, north = g.bounds()
    x_w, _ = lonlat_to_world_px(west, north, g.zoom)
    x_e, _ = lonlat_to_world_px(east, south, g.zoom)
    assert x_e - x_w == pytest.approx(g.width, abs=1e-6)


def test_georef_contains_pixel():
    g = make_georef()
    assert g.contains_pixel(0.0, 0.0)
    assert g.contains_pixel(g.width - 1.0, g.height - 1.0)
    assert not g.contains_pixel(g.width, 0.0)
    assert not g.contains_pixel(-1.0, 0.0)


def test_georef_accepts_arrays():
    g = make_georef()
    px = np.array([0.0, 100.0, 500.0])
    py = np.array([0.0, 100.0, 500.0])
    lon, lat = g.pixel_to_lonlat(px, py)
    assert lon.shape == px.shape and lat.shape == py.shape
    px_back, py_back = g.lonlat_to_pixel(lon, lat)
    np.testing.assert_allclose(px_back, px, atol=1e-6)
    np.testing.assert_allclose(py_back, py, atol=1e-6)


def test_georef_rejects_invalid_construction():
    with pytest.raises(ValueError):
        Georef(center_lon=37.0, center_lat=55.0, zoom=18, width=0, height=100)
    with pytest.raises(ValueError):
        Georef(center_lon=200.0, center_lat=55.0, zoom=18, width=100, height=100)
    with pytest.raises(ValueError):
        Georef(center_lon=37.0, center_lat=89.0, zoom=18, width=100, height=100)


def test_georef_is_immutable():
    g = make_georef()
    with pytest.raises(Exception):
        g.center_lat = 0.0  # type: ignore[misc]
