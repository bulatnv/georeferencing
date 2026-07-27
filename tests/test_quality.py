"""Тесты оценки качества: ковариация, эллипс, NCC, статус — и её калибровка.

Ключевой тест — Монте-Карло калибровка ковариации центра
(:func:`test_center_covariance_coverage_is_calibrated`): под гауссовым шумом
локализации точек доля истинных центров внутри 1σ/2σ-эллипса обязана совпадать с
номиналом (39% / 86% для 2D). Это критерий приёмки шага (``docs/TESTING.md``).
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from aero_geoloc.matcher import Correspondences
from aero_geoloc.pose import PoseEstimate, SimilarityTransform, estimate_similarity
from aero_geoloc.quality import aligned_ncc, assess, center_covariance, error_ellipse
from aero_geoloc.testbench import make_synthetic_scene
from aero_geoloc.types import Status

CENTER = (255.5, 255.5)


def make_corr(transform, *, n, noise_px, rng):
    pts_q = rng.uniform(0.0, 512.0, size=(n, 2))
    pts_r = transform.apply(pts_q) + rng.normal(0.0, noise_px, size=(n, 2))
    return Correspondences(
        pts_q.astype(np.float32), pts_r.astype(np.float32), np.ones(n, np.float32)
    )


# --- калибровка ковариации (главный тест) -----------------------------------


def test_center_covariance_coverage_is_calibrated():
    """Монте-Карло: покрытие 1σ/2σ-эллипса совпадает с номиналом chi²(2)."""
    truth = SimilarityTransform.from_params(1.1, 137.0, 300.0, -120.0)
    true_center = truth.apply(np.array(CENTER))
    rng = np.random.default_rng(0)
    center = np.array(CENTER)

    mahalanobis2 = []
    for _ in range(800):
        corr = make_corr(truth, n=80, noise_px=1.0, rng=rng)
        pose = estimate_similarity(corr, ransac_threshold_px=100.0)  # все точки — инлайеры
        if pose is None:
            continue
        cov = center_covariance(corr.pts_q, corr.pts_r, pose.transform, CENTER)
        err = pose.transform.apply(center) - true_center
        mahalanobis2.append(float(err @ np.linalg.inv(cov) @ err))

    d2 = np.array(mahalanobis2)
    # Для 2D-гаусса d² ~ chi²(2): P(d²≤1)=0.393 (1σ), P(d²≤4)=0.865 (2σ).
    assert np.mean(d2 <= 1.0) == pytest.approx(0.393, abs=0.05)
    assert np.mean(d2 <= 4.0) == pytest.approx(0.865, abs=0.05)
    assert d2.mean() == pytest.approx(2.0, abs=0.2)  # E[chi²(2)] = 2


def test_center_covariance_scales_with_noise_squared():
    rng = np.random.default_rng(1)
    truth = SimilarityTransform.from_params(1.0, 0.0, 0.0, 0.0)
    pts_q = rng.uniform(0.0, 512.0, size=(60, 2))
    noise = rng.normal(0.0, 1.0, size=(60, 2))
    c1 = Correspondences(pts_q.astype(np.float32), (truth.apply(pts_q) + noise).astype(np.float32),
                         np.ones(60, np.float32))
    c2 = Correspondences(pts_q.astype(np.float32), (truth.apply(pts_q) + 3.0 * noise).astype(np.float32),
                         np.ones(60, np.float32))
    cov1 = center_covariance(c1.pts_q, c1.pts_r, truth, CENTER)
    cov2 = center_covariance(c2.pts_q, c2.pts_r, truth, CENTER)
    assert np.trace(cov2) / np.trace(cov1) == pytest.approx(9.0, rel=1e-6)


def test_center_covariance_mpp_scales_to_meters():
    rng = np.random.default_rng(2)
    truth = SimilarityTransform.from_params(1.0, 10.0, 5.0, -5.0)
    corr = make_corr(truth, n=50, noise_px=1.0, rng=rng)
    cov_px = center_covariance(corr.pts_q, corr.pts_r, truth, CENTER, mpp=1.0)
    cov_m = center_covariance(corr.pts_q, corr.pts_r, truth, CENTER, mpp=0.5)
    assert np.allclose(cov_m, cov_px * 0.25)


def test_center_covariance_returns_none_for_too_few_points():
    truth = SimilarityTransform.from_params(1.0, 0.0, 0.0, 0.0)
    rng = np.random.default_rng(3)
    corr = make_corr(truth, n=2, noise_px=1.0, rng=rng)  # 2N−4 = 0 степеней свободы
    assert center_covariance(corr.pts_q, corr.pts_r, truth, CENTER) is None


# --- эллипс ------------------------------------------------------------------


def test_error_ellipse_axis_aligned():
    major, minor, angle = error_ellipse(np.diag([9.0, 4.0]))
    assert major == pytest.approx(3.0)
    assert minor == pytest.approx(2.0)
    assert angle == pytest.approx(0.0)  # большая ось вдоль X


def test_error_ellipse_major_along_y():
    _, _, angle = error_ellipse(np.diag([4.0, 9.0]))
    assert abs(angle) == pytest.approx(90.0)


# --- фотометрический NCC -----------------------------------------------------


def test_aligned_ncc_high_when_aligned_low_when_shifted():
    ref = make_synthetic_scene(512, seed=4).image
    t = SimilarityTransform.from_params(1.0, 0.0, 30.0, 20.0)  # ref = query + (30, 20)
    query = cv2.warpAffine(
        ref, t.matrix.astype(np.float32), (256, 256),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
    )
    good = aligned_ncc(query, ref, t)
    bad = aligned_ncc(query, ref, SimilarityTransform.from_params(1.0, 0.0, 150.0, 120.0))
    assert good > 0.95
    assert bad < good


# --- статус ------------------------------------------------------------------


def _pose(n, noise_px, seed):
    truth = SimilarityTransform.from_params(1.0, 20.0, 100.0, -50.0)
    rng = np.random.default_rng(seed)
    corr = make_corr(truth, n=n, noise_px=noise_px, rng=rng)
    pose = estimate_similarity(corr, ransac_threshold_px=100.0)
    return pose, corr


def test_assess_localized_on_tight_solution():
    pose, corr = _pose(n=80, noise_px=0.5, seed=5)
    q = assess(pose, corr, CENTER, mpp=0.3, photometric_ncc=0.9)
    assert q.status is Status.LOCALIZED
    assert q.error_ellipse_m[0] < 3.0
    assert q.covariance_m2.shape == (2, 2)
    assert 0.0 <= q.confidence <= 1.0


def test_assess_low_confidence_on_large_ellipse():
    # Сильный шум точек → большая ковариация → эллипс выходит за порог.
    pose, corr = _pose(n=40, noise_px=30.0, seed=6)
    q = assess(pose, corr, CENTER, mpp=1.0, photometric_ncc=0.9, max_semi_major_m=3.0)
    assert q.error_ellipse_m[0] > 3.0
    assert q.status is Status.LOW_CONFIDENCE


def test_assess_low_confidence_on_poor_photometric_match():
    pose, corr = _pose(n=80, noise_px=0.5, seed=7)
    q = assess(pose, corr, CENTER, mpp=0.3, photometric_ncc=0.1, min_ncc=0.3)
    assert q.status is Status.LOW_CONFIDENCE  # геометрия хороша, но яркости не сходятся


def test_assess_conjunction_needs_enough_inliers():
    """Связка: тугой матч с хорошим NCC, но инлайеров меньше порога → не LOCALIZED."""
    pose, corr = _pose(n=40, noise_px=0.5, seed=11)  # тугой эллипс, NCC ок
    assert assess(pose, corr, CENTER, mpp=0.3, photometric_ncc=0.9).status is Status.LOCALIZED
    strict = assess(pose, corr, CENTER, mpp=0.3, photometric_ncc=0.9, min_inliers_hard=50)
    assert strict.status is Status.LOW_CONFIDENCE  # инлайеров < порога — связка не проходит


def test_assess_conjunction_needs_ncc_above_calibrated():
    """Связка: отличная геометрия, но NCC ниже калиброванного 0.12 → не LOCALIZED."""
    pose, corr = _pose(n=40, noise_px=0.5, seed=12)  # много инлайеров, тугой эллипс
    assert assess(pose, corr, CENTER, mpp=0.3, photometric_ncc=0.05).status is Status.LOW_CONFIDENCE
    assert assess(pose, corr, CENTER, mpp=0.3, photometric_ncc=0.20).status is Status.LOCALIZED


def test_assess_systematic_floor_inflates_ellipse():
    """Пол абсолютной точности подложки поднимает эллипс не ниже своего уровня."""
    pose, corr = _pose(n=80, noise_px=0.5, seed=8)
    tight = assess(pose, corr, CENTER, mpp=0.3, photometric_ncc=0.9)
    floored = assess(pose, corr, CENTER, mpp=0.3, photometric_ncc=0.9, systematic_floor_m=2.0)
    assert tight.error_ellipse_m[0] < floored.error_ellipse_m[0]
    # Пол 2 м добавляет 2²=4 к дисперсии каждой оси → полуось не меньше ~2 м.
    assert floored.error_ellipse_m[0] >= 2.0
    assert floored.error_ellipse_m[1] >= 2.0
