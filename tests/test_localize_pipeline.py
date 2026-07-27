"""Тесты полной оркестрации :func:`localize` — coarse-to-fine из фазы 2.

Подложку ``localize`` тянет сам через :class:`SceneBasemap` (офлайн-источник
поверх процедурной сцены), поэтому здесь проверяется вся связка: приведение
масштабов → грубый уровень «ГДЕ примерно» → точный уровень + ECC → гейт по
приору. Ground truth даёт генератор стенда.

Про ограничение синтетики: её текстура «схлопывается» при даунсемпле ×2 (мало
точек на грубом уровне), поэтому офлайн-тесты идут на **умеренном** приоре, где
грубый уровень остаётся в нативном разрешении. Реальный путь с даунсемплом
(широкий приор, грубый зум ниже точного) проверяется на настоящих тайлах Esri в
сетевом тесте — там структура выживает и на грубом уровне.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from aero_geoloc.geo import ground_mpp, haversine_m
from aero_geoloc.localize import localize
from aero_geoloc.testbench import (
    Scene,
    SampleSpec,
    SceneBasemap,
    default_camera,
    generate_sample,
    make_synthetic_scene,
)
from aero_geoloc.types import Prior


@pytest.fixture(scope="module")
def scene() -> Scene:
    # 3072 px вмещает грубое окно умеренного приора вокруг сдвинутого центра.
    return make_synthetic_scene(3072, seed=0)


@pytest.fixture(scope="module")
def camera():
    return default_camera(512)


# --- SceneBasemap: геометрия источника --------------------------------------


def test_scene_basemap_native_zoom_is_pixel_exact():
    scene = make_synthetic_scene(1024, seed=1)
    bm = SceneBasemap(scene)
    lon, lat = scene.georef.pixel_to_lonlat(*scene.georef.center_pixel)
    image, georef = bm(lon, lat, scene.georef.zoom, 256, 256)

    # На родном зуме — точный целочисленный кроп центра сцены, без ресемпла.
    x0 = int(round(scene.georef.center_pixel[0] - (256 - 1) / 2.0))
    y0 = int(round(scene.georef.center_pixel[1] - (256 - 1) / 2.0))
    np.testing.assert_array_equal(image, scene.image[y0 : y0 + 256, x0 : x0 + 256])
    assert (georef.zoom, georef.width, georef.height) == (scene.georef.zoom, 256, 256)


def test_scene_basemap_coarse_zoom_downsamples():
    scene = make_synthetic_scene(1024, seed=1)
    bm = SceneBasemap(scene)
    lon, lat = scene.georef.pixel_to_lonlat(*scene.georef.center_pixel)
    zoom = scene.georef.zoom - 1  # грубее сцены вдвое
    image, georef = bm(lon, lat, zoom, 128, 128)

    assert image.shape == (128, 128)
    assert georef.zoom == zoom
    # Ожидаем даунсемпл участка 256×256 сцены до 128×128 (INTER_AREA).
    x0 = int(round(scene.georef.center_pixel[0] - (256 - 1) / 2.0))
    y0 = int(round(scene.georef.center_pixel[1] - (256 - 1) / 2.0))
    expected = cv2.resize(
        scene.image[y0 : y0 + 256, x0 : x0 + 256], (128, 128), interpolation=cv2.INTER_AREA
    )
    np.testing.assert_array_equal(image, expected)


def test_scene_basemap_rejects_finer_than_scene():
    scene = make_synthetic_scene(512, seed=1)
    bm = SceneBasemap(scene)
    lon, lat = scene.georef.pixel_to_lonlat(*scene.georef.center_pixel)
    with pytest.raises(ValueError, match="детальнее сцены"):
        bm(lon, lat, scene.georef.zoom + 1, 128, 128)


# --- localize: сквозная точность --------------------------------------------


@pytest.mark.parametrize(
    "yaw,ratio,sigma,offset,bearing",
    [
        (0.0, 1.0, 50.0, 0.0, 0.0),
        (137.0, 1.1, 60.0, 40.0, 90.0),
        (270.0, 0.9, 70.0, 90.0, 200.0),
        (45.0, 1.05, 50.0, 60.0, 300.0),
        (180.0, 1.0, 80.0, 110.0, 250.0),
    ],
)
def test_localize_end_to_end_subpixel(scene, camera, yaw, ratio, sigma, offset, bearing):
    spec = SampleSpec(
        yaw_deg=yaw,
        altitude_ratio=ratio,
        prior_offset_m=offset,
        prior_offset_bearing_deg=bearing,
    )
    sample = generate_sample(scene, camera, spec, prior_sigma_m=sigma, reference_size=1900)
    result = localize(sample.query, camera, sample.prior, SceneBasemap(scene))

    assert result.is_localized
    err_m = haversine_m(sample.true_lat, sample.true_lon, result.center_lat, result.center_lon)
    err_px = err_m / ground_mpp(sample.true_lat, result.diagnostics["z_fine"])
    assert err_px < 0.2  # субпиксель на точном уровне за счёт ECC
    assert result.diagnostics["refined_ecc"] is True
    # Грубый кандидат сел рядом с истинным сдвигом приора.
    assert result.diagnostics["coarse_offset_m"] == pytest.approx(offset, abs=15.0)


def test_localize_diagnostics_expose_coarse_and_fine_zooms(scene, camera):
    sample = generate_sample(scene, camera, SampleSpec(yaw_deg=30.0), prior_sigma_m=50.0)
    result = localize(sample.query, camera, sample.prior, SceneBasemap(scene))
    d = result.diagnostics
    # Приведение масштабов: точный зум даёт mpp ≈ GSD (в пределах √2 у nearest).
    ratio = ground_mpp(sample.prior.lat, d["z_fine"]) / d["gsd"]
    assert 1 / np.sqrt(2) <= ratio <= np.sqrt(2)
    assert d["z_coarse"] <= d["z_fine"]


def test_localize_wide_prior_selects_coarser_zoom(scene, camera):
    """Широкий приор → грубый уровень уходит на зум ниже точного (даунсемпл)."""
    # σ=120 м: диск ±3σ не влезает в нативные ~2048 px, зум грубого уровня падает.
    sample = generate_sample(scene, camera, SampleSpec(yaw_deg=0.0), prior_sigma_m=120.0)
    result = localize(sample.query, camera, sample.prior, SceneBasemap(scene))
    # Локализация на синтетике при даунсемпле может не пройти (мало точек) — но
    # решение о зуме принимается до матчинга и обязано быть в диагностике.
    assert result.diagnostics["z_coarse"] < result.diagnostics["z_fine"]


# --- localize: цикл переуточнения масштаба ----------------------------------


def test_scale_loop_rescues_large_altitude_error(scene, camera):
    """Крупная ошибка высоты (ratio 2.0): цикл перетягивает подложку на верный зум.

    Приор высоты широкий (altitude_sigma=400), поэтому scale-гейт пропускает
    s≈2; без цикла матч идёт при s≈2 (подложка вдвое детальнее кадра) и
    деградирует, с циклом — переуточняется до s≈1 на зуме ниже.
    """
    spec = SampleSpec(yaw_deg=25.0, altitude_ratio=2.0, prior_offset_m=20.0)
    sample = generate_sample(
        scene, camera, spec, prior_sigma_m=40.0, altitude_sigma_m=400.0, reference_size=2400
    )
    bm = SceneBasemap(scene)

    without = localize(sample.query, camera, sample.prior, bm, scale_iters=0)
    withloop = localize(sample.query, camera, sample.prior, bm, scale_iters=2)

    assert without.is_localized and withloop.is_localized
    # Без цикла зум не трогается; с циклом уходит на уровень ниже (mpp ≈ GSD_true).
    assert without.diagnostics["z_fine"] == without.diagnostics["z_fine_initial"]
    assert withloop.diagnostics["z_fine"] < withloop.diagnostics["z_fine_initial"]
    assert withloop.diagnostics["scale_iters_done"] >= 1
    # Восстановленная высота и субпиксельная точность на согласованном масштабе.
    assert withloop.altitude_est_m == pytest.approx(1200.0, rel=0.05)
    err_px = haversine_m(
        sample.true_lat, sample.true_lon, withloop.center_lat, withloop.center_lon
    ) / ground_mpp(sample.true_lat, withloop.diagnostics["z_fine"])
    assert err_px < 0.2


def test_scale_loop_keeps_best_and_does_not_regress(scene, camera):
    """При умеренной ошибке (ratio 1.5) цикл не портит уже хорошее решение."""
    spec = SampleSpec(yaw_deg=25.0, altitude_ratio=1.5, prior_offset_m=20.0)
    sample = generate_sample(
        scene, camera, spec, prior_sigma_m=40.0, altitude_sigma_m=400.0, reference_size=2200
    )
    bm = SceneBasemap(scene)

    without = localize(sample.query, camera, sample.prior, bm, scale_iters=0)
    withloop = localize(sample.query, camera, sample.prior, bm, scale_iters=2)

    assert without.is_localized and withloop.is_localized
    # Соседний зум даёт s<1 (подложка грубее кадра) и меньше инлайеров — цикл его
    # отвергает по числу инлайеров и оставляет исходный зум.
    assert withloop.diagnostics["z_fine"] == without.diagnostics["z_fine"]
    assert withloop.diagnostics["n_inliers"] >= without.diagnostics["n_inliers"] - 2
    err_px = haversine_m(
        sample.true_lat, sample.true_lon, withloop.center_lat, withloop.center_lon
    ) / ground_mpp(sample.true_lat, withloop.diagnostics["z_fine"])
    assert err_px < 0.2


# --- низкие высоты и не-инвариантные матчеры --------------------------------


def test_max_zoom_clamps_fine_zoom(scene, camera):
    """Зум точного уровня не превышает максимум провайдера (низкая высота)."""
    sample = generate_sample(scene, camera, SampleSpec(yaw_deg=0.0), prior_sigma_m=50.0)
    result = localize(sample.query, camera, sample.prior, SceneBasemap(scene), max_zoom=17)
    assert result.diagnostics["z_fine"] <= 17
    assert result.diagnostics["z_coarse"] <= 17


def test_prerotate_preserves_geometry(scene, camera):
    """Предповорот кадра к северу и возврат координат точек не ломает геометрию.

    Для инвариантного к повороту SIFT результат обязан остаться точным — так
    проверяется корректность самого механизма предповорота (нужного LightGlue).
    """
    spec = SampleSpec(yaw_deg=137.0, prior_offset_m=20.0)
    sample = generate_sample(scene, camera, spec, prior_sigma_m=60.0, reference_size=1900)
    result = localize(sample.query, camera, sample.prior, SceneBasemap(scene), prerotate=True)
    assert result.is_localized
    err_px = haversine_m(
        sample.true_lat, sample.true_lon, result.center_lat, result.center_lon
    ) / ground_mpp(sample.true_lat, result.diagnostics["z_fine"])
    assert err_px < 0.5


# --- localize: честный отказ ------------------------------------------------


def test_localize_rejects_prior_far_from_truth(scene, camera):
    """Истина вне диска приора → NOT_LOCALIZED, а не уверенно-неверная точка."""
    from dataclasses import replace

    sample = generate_sample(scene, camera, SampleSpec(yaw_deg=30.0), prior_sigma_m=25.0)
    # Приор в 200 м от истины при σ=25 (3σ=75 м): истина вне диска и вне грубого окна.
    far = replace(sample.prior, lat=sample.true_lat + 200.0 / 111320.0, sigma_m=25.0)
    result = localize(sample.query, camera, far, SceneBasemap(scene))
    assert not result.is_localized


# --- сетевой дымовой тест: реальный путь с даунсемплом -----------------------


@pytest.mark.skipif(
    os.environ.get("AERO_GEOLOC_NETWORK_TESTS") != "1",
    reason="сетевой тест выключен; включить: AERO_GEOLOC_NETWORK_TESTS=1",
)
def test_localize_real_basemap_downsample_smoke(tmp_path):
    from aero_geoloc.basemap import TileCache, fetch_basemap

    cache = TileCache(tmp_path)
    img, gr = fetch_basemap(37.6173, 55.7558, 18, 3328, 3328, cache=cache)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    real_scene = Scene(image=gray, georef=gr)
    camera = default_camera(512)

    # σ=120 м вынуждает грубый уровень на зум ниже точного (даунсемпл ×2).
    spec = SampleSpec(yaw_deg=137.0, prior_offset_m=90.0, prior_offset_bearing_deg=274.0)
    sample = generate_sample(real_scene, camera, spec, prior_sigma_m=120.0, reference_size=1500)
    result = localize(sample.query, camera, sample.prior, SceneBasemap(real_scene))

    assert result.is_localized
    assert result.diagnostics["z_coarse"] < result.diagnostics["z_fine"]
    err_m = haversine_m(sample.true_lat, sample.true_lon, result.center_lat, result.center_lon)
    assert err_m < 0.5  # на реальной текстуре точность субметровая
