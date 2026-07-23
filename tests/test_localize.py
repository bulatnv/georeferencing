"""Сквозные тесты фазы 1 и её критерий приёмки (``docs/PLAN.md``).

Критерий: на чистой синтетике медианная ошибка центра — доли пикселя, ошибка
поворота < 0.5°, масштаба < 1%, пайплайн стабильно сходится без ложняков.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np
import pytest

from aero_geoloc.localize import localize_against_reference, normalize_gray
from aero_geoloc.matcher import AKAZEMatcher, SIFTMatcher
from aero_geoloc.testbench import (
    SampleSpec,
    default_camera,
    evaluate,
    generate_sample,
    iter_specs,
    make_synthetic_scene,
    run_grid,
)
from aero_geoloc.types import Status


@pytest.fixture(scope="module")
def scene():
    return make_synthetic_scene(2048, seed=0)


@pytest.fixture(scope="module")
def camera():
    return default_camera(512)


@pytest.fixture(scope="module")
def sample(scene, camera):
    return generate_sample(
        scene, camera, SampleSpec(yaw_deg=137.0, altitude_ratio=1.1, prior_offset_m=15.0)
    )


def run(sample, **kwargs):
    return localize_against_reference(
        sample.query, sample.camera, sample.prior, sample.reference,
        sample.reference_georef, **kwargs
    )


# --- нормализация входа -----------------------------------------------------


def test_normalize_gray_converts_colour_and_passes_through_gray():
    gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
    np.testing.assert_array_equal(normalize_gray(gray), gray)

    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    np.testing.assert_allclose(normalize_gray(bgr), gray, atol=1)
    bgra = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGRA)
    np.testing.assert_allclose(normalize_gray(bgra), gray, atol=1)


def test_normalize_gray_rescales_non_uint8():
    out = normalize_gray(np.linspace(0.0, 1.0, 256, dtype=np.float32).reshape(16, 16))
    assert out.dtype == np.uint8
    assert (out.min(), out.max()) == (0, 255)


def test_normalize_gray_clahe_changes_contrast():
    rng = np.random.default_rng(0)
    img = (rng.normal(128, 12, (128, 128))).clip(0, 255).astype(np.uint8)
    assert normalize_gray(img, clahe=True).std() > img.std()


def test_normalize_gray_rejects_empty_and_odd_channel_counts():
    with pytest.raises(ValueError, match="пустое"):
        normalize_gray(np.empty((0, 0), np.uint8))
    with pytest.raises(ValueError, match="каналами"):
        normalize_gray(np.zeros((8, 8, 5), np.uint8))


# --- согласованность входов -------------------------------------------------

def test_rejects_frame_that_disagrees_with_camera(sample):
    wrong = dataclasses.replace(sample.camera, image_width=sample.camera.image_width + 8)
    with pytest.raises(ValueError, match="не совпадает"):
        localize_against_reference(
            sample.query, wrong, sample.prior, sample.reference, sample.reference_georef
        )


def test_rejects_reference_that_disagrees_with_georef(sample):
    with pytest.raises(ValueError, match="не совпадает"):
        localize_against_reference(
            sample.query, sample.camera, sample.prior,
            sample.reference[:-4, :-4], sample.reference_georef,
        )


# --- одиночная локализация --------------------------------------------------


def test_single_sample_is_localized_accurately(sample):
    result = run(sample)
    metrics = evaluate(result, sample)

    assert result.status is Status.LOCALIZED
    assert metrics.center_error_px < 1.5
    assert metrics.heading_error_deg < 0.5
    assert abs(metrics.scale_error_pct) < 1.0


def test_result_carries_full_payload(sample):
    result = run(sample)

    assert result.transform.shape == (2, 3)
    assert len(result.footprint_lonlat) == 4
    assert 0.0 <= result.heading_deg < 360.0
    assert result.altitude_est_m == pytest.approx(sample.true_altitude_m, rel=0.01)
    # Фаза 1 не калибрует доверие и обязана признавать это в диагностике.
    assert result.diagnostics["confidence_calibrated"] is False
    for key in ("n_inliers", "inlier_ratio", "reprojection_rmse_px", "scale", "rotation_deg"):
        assert key in result.diagnostics


def test_footprint_encloses_the_estimated_centre(sample):
    result = run(sample)
    lons = [p[0] for p in result.footprint_lonlat]
    lats = [p[1] for p in result.footprint_lonlat]
    assert min(lons) < result.center_lon < max(lons)
    assert min(lats) < result.center_lat < max(lats)


def test_heading_is_recovered_without_trusting_yaw(scene, camera):
    """Стратегия «не доверяем yaw»: поворот находится матчером сам."""
    s = generate_sample(scene, camera, SampleSpec(yaw_deg=213.0))
    result = run(s, trust_yaw=False)
    assert result.status is Status.LOCALIZED
    assert evaluate(result, s).heading_error_deg < 0.5


def test_swapping_the_matcher_changes_nothing_above_it(sample):
    """Сменное ядро: AKAZE вместо SIFT — та же обвязка, тот же результат."""
    sift = evaluate(run(sample, matcher=SIFTMatcher()), sample)
    akaze = evaluate(run(sample, matcher=AKAZEMatcher()), sample)

    assert sift.localized and akaze.localized
    assert akaze.center_error_px < 3.0
    assert abs(sift.center_error_m - akaze.center_error_m) < 1.0


# --- отказы -----------------------------------------------------------------


def test_prior_position_acts_as_a_constraint(scene, camera):
    """Решение вне диска ±3σ приора отбрасывается, а не выдаётся как есть."""
    s = generate_sample(scene, camera, SampleSpec(prior_offset_m=60.0))
    tight = dataclasses.replace(s, prior=dataclasses.replace(s.prior, sigma_m=5.0))

    assert run(s).status is Status.LOCALIZED
    result = run(tight)
    assert result.status is Status.NOT_LOCALIZED
    assert result.diagnostics["reason"] == "решение вне диска приора"
    assert result.center_lat is None


def test_wrong_altitude_prior_rejects_via_scale_bounds(sample):
    """Приор высоты тоже ограничение: абсурдный масштаб не проходит."""
    absurd = dataclasses.replace(
        sample, prior=dataclasses.replace(sample.prior, altitude_m=3000.0, altitude_sigma_m=10.0)
    )
    result = run(absurd)
    assert result.status is Status.NOT_LOCALIZED
    assert result.diagnostics["reason"] == "нет устойчивой модели подобия"


def test_featureless_frame_is_refused(scene, camera):
    """Однородный кадр — честный отказ, а не уверенно-неверная точка."""
    s = generate_sample(scene, camera, SampleSpec())
    blank = dataclasses.replace(s, query=np.full_like(s.query, 128))

    result = run(blank)
    assert result.status is Status.NOT_LOCALIZED
    assert result.diagnostics["reason"] == "слишком мало соответствий"
    assert result.confidence == 0.0


# --- критерий приёмки фазы 1 ------------------------------------------------


def test_phase1_acceptance_criteria(scene, camera):
    """Сетка возмущений: yaw по кругу × масштаб 0.8–1.2 × промах приора."""
    specs = list(
        iter_specs(
            yaw_deg=(0.0, 45.0, 137.0, 225.0, 315.0),
            altitude_ratio=(0.8, 1.0, 1.2),
            prior_offset_m=(0.0, 20.0),
        )
    )
    summary = run_grid(scene, camera, specs)

    assert summary.n_samples == 30
    assert summary.success_rate == 1.0, summary.format_report()
    # «Доли пикселя» подложки — потолок здесь задаёт точность локализации
    # ключевых точек SIFT; субпиксельный refinement приедет в фазе 2.
    assert summary.median_center_error_px < 1.0, summary.format_report()
    assert summary.max_center_error_m < 1.0, summary.format_report()
    assert summary.max_heading_error_deg < 0.5, summary.format_report()
    assert summary.max_abs_scale_error_pct < 1.0, summary.format_report()
