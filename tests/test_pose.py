"""Тесты робастной оценки позы (фаза 1, ``docs/PLAN.md``).

Соответствия здесь синтезируются напрямую из известного преобразования — без
изображений и матчера. Так проверяется именно pose: восстановление 4 DoF,
устойчивость к выбросам и работа приоров **как ограничений**.
"""

from __future__ import annotations

import numpy as np
import pytest

from aero_geoloc.matcher import Correspondences
from aero_geoloc.pose import SimilarityTransform, estimate_similarity


def make_corr(
    transform: SimilarityTransform,
    *,
    n: int = 120,
    outlier_fraction: float = 0.0,
    noise_px: float = 0.0,
    seed: int = 0,
) -> Correspondences:
    """Соответствия по известному преобразованию, опционально с мусором."""
    rng = np.random.default_rng(seed)
    pts_q = rng.uniform(0.0, 512.0, size=(n, 2))
    pts_r = transform.apply(pts_q)
    if noise_px:
        pts_r = pts_r + rng.normal(0.0, noise_px, size=pts_r.shape)

    n_outliers = int(round(n * outlier_fraction))
    if n_outliers:
        pts_r[:n_outliers] = rng.uniform(0.0, 1024.0, size=(n_outliers, 2))

    return Correspondences(
        pts_q=pts_q.astype(np.float32),
        pts_r=pts_r.astype(np.float32),
        conf=np.full(n, 0.5, dtype=np.float32),
    )


# --- SimilarityTransform ----------------------------------------------------


@pytest.mark.parametrize("rotation_deg", [-179.0, -90.0, 0.0, 45.0, 137.0, 179.0])
@pytest.mark.parametrize("scale", [0.5, 1.0, 2.3])
def test_transform_parameter_roundtrip(scale, rotation_deg):
    t = SimilarityTransform.from_params(scale, rotation_deg, 12.0, -34.0)
    assert t.scale == pytest.approx(scale, rel=1e-12)
    assert t.rotation_deg == pytest.approx(rotation_deg, abs=1e-12)
    assert t.translation == pytest.approx((12.0, -34.0))


def test_rotation_is_reported_in_half_open_range():
    """Ровно 180° возвращается как −180°: диапазон полуоткрытый, это конвенция."""
    assert SimilarityTransform.from_params(1.0, 180.0, 0.0, 0.0).rotation_deg == pytest.approx(
        -180.0
    )
    assert SimilarityTransform.from_params(1.0, 370.0, 0.0, 0.0).rotation_deg == pytest.approx(
        10.0, abs=1e-12
    )


def test_transform_rejects_bad_input():
    with pytest.raises(ValueError, match="2×3"):
        SimilarityTransform(np.eye(3))
    with pytest.raises(ValueError, match="scale"):
        SimilarityTransform.from_params(0.0, 0.0, 0.0, 0.0)


def test_transform_apply_single_point_and_array():
    t = SimilarityTransform.from_params(2.0, 90.0, 10.0, 20.0)
    # Поворот на 90°: (1, 0) → (0, 1), с масштабом 2 и сдвигом → (10, 22).
    np.testing.assert_allclose(t.apply(np.array([1.0, 0.0])), [10.0, 22.0], atol=1e-12)
    out = t.apply(np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out[1], [8.0, 20.0], atol=1e-12)


def test_transform_inverse_roundtrip():
    t = SimilarityTransform.from_params(1.7, 137.0, -55.0, 42.0)
    pts = np.array([[0.0, 0.0], [100.0, -30.0], [511.0, 511.0]])
    np.testing.assert_allclose(t.inverse().apply(t.apply(pts)), pts, atol=1e-9)
    assert t.inverse().scale == pytest.approx(1.0 / 1.7)
    assert t.inverse().rotation_deg == pytest.approx(-137.0)


def test_transform_inverse_rejects_degenerate():
    with pytest.raises(ValueError, match="вырожденное"):
        SimilarityTransform(np.zeros((2, 3))).inverse()


# --- восстановление модели --------------------------------------------------


@pytest.mark.parametrize("rotation_deg", [0.0, 37.0, 137.0, -95.0])
@pytest.mark.parametrize("scale", [0.8, 1.0, 1.2])
def test_recovers_exact_transform_from_clean_correspondences(scale, rotation_deg):
    truth = SimilarityTransform.from_params(scale, rotation_deg, 123.0, -45.0)
    pose = estimate_similarity(make_corr(truth))

    assert pose is not None
    assert pose.transform.scale == pytest.approx(scale, rel=1e-6)
    assert pose.transform.rotation_deg == pytest.approx(rotation_deg, abs=1e-4)
    assert pose.transform.translation == pytest.approx((123.0, -45.0), abs=1e-3)
    assert pose.reprojection_rmse_px == pytest.approx(0.0, abs=1e-3)
    assert pose.inlier_ratio == pytest.approx(1.0)


@pytest.mark.parametrize("outlier_fraction", [0.2, 0.4, 0.6])
def test_survives_outliers(outlier_fraction):
    """Мусор от матчера — норма; RANSAC обязан его вычистить."""
    truth = SimilarityTransform.from_params(1.05, 63.0, 40.0, 80.0)
    pose = estimate_similarity(
        make_corr(truth, n=300, outlier_fraction=outlier_fraction, seed=3),
        ransac_threshold_px=2.0,
    )

    assert pose is not None
    assert pose.transform.scale == pytest.approx(1.05, rel=1e-3)
    assert pose.transform.rotation_deg == pytest.approx(63.0, abs=0.1)
    assert pose.inlier_ratio == pytest.approx(1.0 - outlier_fraction, abs=0.1)


def test_noise_shows_up_in_reprojection_rmse():
    truth = SimilarityTransform.from_params(1.0, 20.0, 5.0, 5.0)
    clean = estimate_similarity(make_corr(truth, seed=4))
    noisy = estimate_similarity(make_corr(truth, noise_px=1.0, seed=4), ransac_threshold_px=5.0)

    assert clean is not None and noisy is not None
    assert noisy.reprojection_rmse_px > clean.reprojection_rmse_px
    assert noisy.reprojection_rmse_px == pytest.approx(1.4, abs=0.5)  # ~√2·σ на двух осях


def test_inlier_mask_matches_counts():
    truth = SimilarityTransform.from_params(1.0, 0.0, 0.0, 0.0)
    pose = estimate_similarity(make_corr(truth, n=200, outlier_fraction=0.25, seed=5))

    assert pose is not None
    assert pose.inlier_mask.dtype == bool
    assert pose.inlier_mask.size == 200
    assert pose.n_inliers == int(np.count_nonzero(pose.inlier_mask))
    assert pose.diagnostics()["n_correspondences"] == 200


# --- отказы -----------------------------------------------------------------


def test_too_few_correspondences_returns_none():
    truth = SimilarityTransform.from_params(1.0, 0.0, 0.0, 0.0)
    assert estimate_similarity(make_corr(truth, n=2)) is None
    assert estimate_similarity(Correspondences.empty()) is None


def test_min_inliers_gate():
    truth = SimilarityTransform.from_params(1.0, 10.0, 0.0, 0.0)
    corr = make_corr(truth, n=40, seed=6)
    assert estimate_similarity(corr, min_inliers=10) is not None
    assert estimate_similarity(corr, min_inliers=100) is None


def test_scale_bounds_reject_implausible_solution():
    """Приор высоты как ограничение: масштаб вне допуска — решения нет."""
    truth = SimilarityTransform.from_params(2.0, 0.0, 0.0, 0.0)
    corr = make_corr(truth, seed=7)
    assert estimate_similarity(corr, scale_bounds=(1.8, 2.2)) is not None
    assert estimate_similarity(corr, scale_bounds=(0.8, 1.2)) is None


def test_rotation_constraint_rejects_and_accepts():
    truth = SimilarityTransform.from_params(1.0, 40.0, 0.0, 0.0)
    corr = make_corr(truth, seed=8)
    assert estimate_similarity(corr, expected_rotation_deg=45.0, rotation_tolerance_deg=10.0)
    assert (
        estimate_similarity(corr, expected_rotation_deg=90.0, rotation_tolerance_deg=10.0) is None
    )


def test_rotation_constraint_handles_wraparound():
    """355° и 5° различаются на 10°, а не на 350° — иначе кадры у севера отвергались бы."""
    truth = SimilarityTransform.from_params(1.0, 5.0, 0.0, 0.0)
    pose = estimate_similarity(
        make_corr(truth, seed=9), expected_rotation_deg=355.0, rotation_tolerance_deg=15.0
    )
    assert pose is not None


def test_with_transform_keeps_inliers_and_recomputes_rmse():
    """Подмена преобразования (после refinement): инлайеры те же, RMSE пересчитан."""
    truth = SimilarityTransform.from_params(1.0, 30.0, 10.0, -5.0)
    corr = make_corr(truth, outlier_fraction=0.2, noise_px=0.3, seed=11)
    pose = estimate_similarity(corr, ransac_threshold_px=2.0)
    assert pose is not None and pose.reprojection_rmse_px < 1.0

    # Сдвинутое на 3 px преобразование обязано дать больший RMSE на тех же инлайерах.
    tx, ty = pose.transform.translation
    shifted = SimilarityTransform.from_params(pose.transform.scale, pose.transform.rotation_deg, tx + 3.0, ty)
    updated = pose.with_transform(shifted, corr)

    np.testing.assert_array_equal(updated.inlier_mask, pose.inlier_mask)
    assert updated.n_inliers == pose.n_inliers
    assert updated.reprojection_rmse_px == pytest.approx(3.0, abs=0.2)
