"""Тесты синтетического стенда (фаза 1, ``docs/PLAN.md``).

Стенд — измерительный прибор, и он должен быть проверен раньше того, что им
меряют: если ground truth стенда сам по себе кривой, все метрики выше
бессмысленны.
"""

from __future__ import annotations

import numpy as np
import pytest

from aero_geoloc.geo import haversine_m
from aero_geoloc.testbench import (
    Sample,
    SampleSpec,
    Scene,
    default_camera,
    evaluate,
    generate_sample,
    iter_specs,
    make_synthetic_scene,
    run_grid,
)
from aero_geoloc.types import LocalizationResult, Status


@pytest.fixture(scope="module")
def scene() -> Scene:
    return make_synthetic_scene(1024, seed=0)


# --- сцена ------------------------------------------------------------------


def test_scene_is_deterministic_by_seed():
    a = make_synthetic_scene(256, seed=42)
    b = make_synthetic_scene(256, seed=42)
    c = make_synthetic_scene(256, seed=43)
    np.testing.assert_array_equal(a.image, b.image)
    assert not np.array_equal(a.image, c.image)


def test_scene_is_textured_enough_to_match(scene):
    """Сцена должна быть богата структурой — иначе тест мерил бы её бедность."""
    assert scene.image.dtype == np.uint8
    assert scene.image.std() > 30.0


def test_scene_validates_georef_size():
    good = make_synthetic_scene(256, seed=0)
    with pytest.raises(ValueError, match="не совпадает"):
        Scene(image=np.zeros((128, 128), np.uint8), georef=good.georef)


def test_window_georef_matches_crop_position(scene):
    """Окно подложки привязано ровно к тому месту, откуда вырезано."""
    size = 256
    cx, cy = 400.0, 350.0
    window, georef = scene.window(cx, cy, size)

    assert window.shape == (size, size)
    # Центр окна и запрошенная точка совпадают с точностью до квантизации
    # начала окна: до 0.5 px по каждой оси, то есть 0.71 px по диагонали.
    lon_scene, lat_scene = scene.georef.pixel_to_lonlat(cx, cy)
    lon_win, lat_win = georef.pixel_to_lonlat(*georef.center_pixel)
    assert haversine_m(lat_scene, lon_scene, lat_win, lon_win) < 0.71 * scene.mpp

    # И пиксели совпадают с обычным срезом numpy.
    x0, y0 = int(round(cx - (size - 1) / 2)), int(round(cy - (size - 1) / 2))
    np.testing.assert_array_equal(window, scene.image[y0 : y0 + size, x0 : x0 + size])


def test_window_rejects_out_of_bounds(scene):
    with pytest.raises(ValueError, match="не помещается"):
        scene.window(50.0, 50.0, 512)


# --- генерация примера ------------------------------------------------------


def test_sample_ground_truth_is_self_consistent(scene):
    """Истинный центр в пикселях окна и истинные lat/lon — одна и та же точка."""
    sample = generate_sample(
        scene,
        default_camera(256),
        SampleSpec(yaw_deg=30.0, center_offset_px=(40.0, -25.0), prior_offset_m=10.0,
                   prior_offset_bearing_deg=90.0),
        reference_size=640,
    )
    lon, lat = sample.reference_georef.pixel_to_lonlat(*sample.true_center_ref_px)
    assert haversine_m(sample.true_lat, sample.true_lon, lat, lon) < 1e-3


def test_prior_offset_is_realized_as_requested(scene):
    """Заявленный промах приора совпадает с фактическим расстоянием до истины."""
    sample = generate_sample(
        scene,
        default_camera(256),
        SampleSpec(prior_offset_m=25.0, prior_offset_bearing_deg=90.0),
        reference_size=640,
    )
    actual = haversine_m(sample.true_lat, sample.true_lon, sample.prior.lat, sample.prior.lon)
    # Допуск — расхождение конвенций радиусов (0.11%) плюс округление окна до пикселя.
    assert actual == pytest.approx(25.0, abs=0.5)
    # Азимут 90° — строго на восток.
    assert sample.prior.lon > sample.true_lon
    assert sample.prior.lat == pytest.approx(sample.true_lat, abs=1e-6)


def test_true_scale_follows_altitude_ratio(scene):
    camera = default_camera(256)
    base = generate_sample(scene, camera, SampleSpec(), reference_size=640)
    higher = generate_sample(scene, camera, SampleSpec(altitude_ratio=1.2), reference_size=640)

    assert higher.true_altitude_m == pytest.approx(1.2 * base.true_altitude_m)
    assert higher.true_scale == pytest.approx(1.2 * base.true_scale)


def test_sample_query_has_requested_size(scene):
    camera = default_camera(256)
    sample = generate_sample(scene, camera, SampleSpec(yaw_deg=90.0), reference_size=640)
    assert sample.query.shape == (camera.image_height, camera.image_width)
    assert sample.query.dtype == np.uint8


def test_generator_refuses_when_frame_leaves_the_window(scene):
    """Молчаливо достроить кадр отражением краёв нельзя — истина станет выдумкой."""
    with pytest.raises(ValueError, match="не помещается в окно"):
        generate_sample(
            scene, default_camera(256), SampleSpec(altitude_ratio=1.2), reference_size=300
        )


def test_appearance_levels_are_not_supported_yet(scene):
    with pytest.raises(NotImplementedError, match="фазе 2"):
        generate_sample(
            scene, default_camera(256), SampleSpec(appearance_level=2), reference_size=640
        )


def test_iter_specs_builds_cartesian_grid():
    specs = list(iter_specs(yaw_deg=(0.0, 90.0), altitude_ratio=(0.9, 1.1), prior_offset_m=(0.0, 5.0)))
    assert len(specs) == 8
    assert len({(s.yaw_deg, s.altitude_ratio, s.prior_offset_m) for s in specs}) == 8


# --- метрики ----------------------------------------------------------------


def _dummy_sample(scene) -> Sample:
    return generate_sample(scene, default_camera(256), SampleSpec(), reference_size=640)


def test_evaluate_handles_refusal_without_crashing(scene):
    metrics = evaluate(LocalizationResult.failed("нет соответствий"), _dummy_sample(scene))
    assert metrics.localized is False
    assert metrics.status is Status.NOT_LOCALIZED
    assert metrics.reason == "нет соответствий"
    assert np.isnan(metrics.center_error_m)


def test_summary_of_all_failures_is_not_a_crash(scene):
    """Прогон, где всё отказало, обязан дать сводку, а не деление на ноль."""
    summary = run_grid(
        scene,
        default_camera(256),
        [SampleSpec()],
        reference_size=640,
        # Заведомо невозможное требование к числу инлайеров даст отказ.
        matcher=_NullMatcher(),
    )
    assert summary.n_localized == 0
    assert summary.success_rate == 0.0
    assert np.isnan(summary.median_center_error_m)
    assert "локализовано:           0" in summary.format_report()


class _NullMatcher:
    """Матчер, который ничего не находит — проверяет ветку отказа сквозь стенд."""

    def match(self, query_gray, ref_gray):
        from aero_geoloc.matcher import Correspondences

        return Correspondences.empty()
