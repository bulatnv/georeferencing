"""Геометрические сигналы позы: аналитические свойства, а не «как получилось».

Смысл этих сигналов (E2 из `docs/ROADMAP.md`) — судить о позе, **не глядя на
яркости**: тогда сезон на них не влияет по построению. Тесты проверяют именно те
свойства, ради которых они введены.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aero_geoloc.matcher import Correspondences
from aero_geoloc.pose import (
    UNIFORM_SPREAD,
    SimilarityTransform,
    bootstrap_center_scatter_px,
    fit_similarity,
    inlier_spread,
)

FRAME = (640, 480)
DIAGONAL = math.hypot(*FRAME)


def corr_from(pts_q: np.ndarray, transform: SimilarityTransform,
              noise: float = 0.0, seed: int = 0) -> Correspondences:
    """Соответствия по известному преобразованию плюс шум наблюдения.

    ВАЖНО про ``seed``: он обязан отличаться от того, которым порождены сами
    точки. ``rng.normal(loc, scale)`` — это ``loc + scale·g`` из одного и того же
    потока, поэтому одинаковый seed даёт шум, ПРОПОРЦИОНАЛЬНЫЙ смещениям точек от
    центра. Такой «шум» — это ровно изменение масштаба, то есть подобие: невязок
    нет, бутстрэп идеально устойчив, и тест на неустойчивость проходит наоборот.
    Ошибка стоила отладки, поэтому seed шума здесь смещён явно.
    """
    rng = np.random.default_rng(seed + 9000)
    pts_r = transform.apply(pts_q)
    if noise:
        pts_r = pts_r + rng.normal(0.0, noise, pts_r.shape)
    return Correspondences(pts_q=pts_q.astype(np.float32), pts_r=pts_r.astype(np.float32),
                           conf=np.ones(len(pts_q), np.float32))


# --- fit_similarity ---------------------------------------------------------

def test_fit_recovers_known_transform_exactly():
    truth = SimilarityTransform.from_params(1.7, 23.0, 120.0, -40.0)
    pts = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 80.0], [0.0, 80.0]])
    fitted = fit_similarity(pts, truth.apply(pts))
    assert fitted.scale == pytest.approx(1.7, rel=1e-9)
    assert fitted.rotation_deg == pytest.approx(23.0, abs=1e-9)
    assert fitted.translation == pytest.approx((120.0, -40.0), abs=1e-8)


def test_fit_needs_two_distinct_points():
    same = np.array([[10.0, 10.0], [10.0, 10.0], [10.0, 10.0]])
    assert fit_similarity(same, same) is None
    assert fit_similarity(np.array([[1.0, 1.0]]), np.array([[2.0, 2.0]])) is None


# --- inlier_spread ----------------------------------------------------------

def test_uniform_points_match_the_reference_constant():
    """Константа UNIFORM_SPREAD должна воспроизводиться выборкой, а не быть на веру."""
    rng = np.random.default_rng(3)
    side = 500.0
    pts = rng.uniform(0.0, side, (4000, 2))
    assert inlier_spread(pts, side * math.sqrt(2.0)) == pytest.approx(UNIFORM_SPREAD, abs=0.01)


def test_clustered_inliers_score_far_below_uniform():
    """Тридцать точек в одном углу — подозрительны независимо от их числа."""
    rng = np.random.default_rng(1)
    clump = rng.normal([60.0, 60.0], 8.0, (30, 2))
    spread = rng.uniform([0, 0], FRAME, (30, 2))
    assert inlier_spread(clump, DIAGONAL) < 0.05
    assert inlier_spread(spread, DIAGONAL) > 0.25


def test_spread_does_not_depend_on_point_count():
    """Отличие от «покрытия сетки»: мера обязана мерить геометрию, а не счётчик.

    Покрытие сетки 4×4 при восьми инлайерах не может превысить 0.5 — и потому
    меряет во многом то же самое, что порог по числу инлайеров.
    """
    rng = np.random.default_rng(5)
    # Восемь точек сами по себе шумят, поэтому сравниваются СРЕДНИЕ по выборкам:
    # проверяется отсутствие систематической зависимости от числа точек, а не
    # совпадение единичных реализаций.
    few = np.mean([inlier_spread(rng.uniform([0, 0], FRAME, (8, 2)), DIAGONAL)
                   for _ in range(60)])
    many = np.mean([inlier_spread(rng.uniform([0, 0], FRAME, (200, 2)), DIAGONAL)
                    for _ in range(10)])
    assert few == pytest.approx(many, abs=0.02)


def test_degenerate_inputs_are_zero():
    assert inlier_spread(np.zeros((1, 2)), DIAGONAL) == 0.0
    assert inlier_spread(np.zeros((5, 2)), 0.0) == 0.0


# --- bootstrap_center_scatter_px --------------------------------------------

CENTRE = ((FRAME[0] - 1) / 2.0, (FRAME[1] - 1) / 2.0)
TRUTH = SimilarityTransform.from_params(1.0, 12.0, 300.0, 250.0)


def test_consistent_pose_is_stable_under_resampling():
    """Верная поза опирается на много независимых точек: выборка её не шатает."""
    rng = np.random.default_rng(2)
    pts = rng.uniform([0, 0], FRAME, (60, 2))
    corr = corr_from(pts, TRUTH, noise=0.5, seed=2)
    mask = np.ones(len(pts), bool)
    assert bootstrap_center_scatter_px(corr, mask, CENTRE) < 1.0


def test_inconsistent_correspondences_make_the_centre_wander():
    """Поза на случайных совпадениях: центр гуляет, хотя «инлайеров» столько же.

    Это и есть искомое свойство — сигнал ловит несогласованность там, где
    счётчик инлайеров показывает благополучие, а фотометрия молчит.
    """
    rng = np.random.default_rng(4)
    pts_q = rng.uniform([0, 0], FRAME, (60, 2)).astype(np.float32)
    pts_r = rng.uniform([0, 0], FRAME, (60, 2)).astype(np.float32)
    corr = Correspondences(pts_q=pts_q, pts_r=pts_r, conf=np.ones(60, np.float32))
    mask = np.ones(60, bool)
    assert bootstrap_center_scatter_px(corr, mask, CENTRE) > 20.0


def test_clustered_inliers_extrapolate_badly_to_the_centre():
    """Куча точек в углу задаёт масштаб и поворот плохо — центр уезжает.

    Тут сигнал видит то, чего не видит ни RMSE (невязки на самих точках малы),
    ни счётчик инлайеров (их много).
    """
    rng = np.random.default_rng(6)
    clump = rng.normal([40.0, 40.0], 6.0, (40, 2))
    corr = corr_from(clump, TRUTH, noise=0.7, seed=6)
    mask = np.ones(40, bool)
    spread_corr = corr_from(rng.uniform([0, 0], FRAME, (40, 2)), TRUTH, noise=0.7, seed=7)
    assert (bootstrap_center_scatter_px(corr, mask, CENTRE)
            > 5.0 * bootstrap_center_scatter_px(spread_corr, mask, CENTRE))


def test_too_few_inliers_report_infinity():
    """По трём точкам разброс не оценить — честнее сказать «не знаю»."""
    pts = np.array([[0.0, 0.0], [10.0, 5.0], [20.0, 0.0]])
    corr = corr_from(pts, TRUTH)
    assert bootstrap_center_scatter_px(corr, np.ones(3, bool), CENTRE) == float("inf")


def test_scatter_is_reproducible():
    """Сигнал участвует в калибровке порогов — он обязан быть детерминированным."""
    rng = np.random.default_rng(8)
    pts = rng.uniform([0, 0], FRAME, (40, 2))
    corr = corr_from(pts, TRUTH, noise=0.6, seed=8)
    mask = np.ones(40, bool)
    assert (bootstrap_center_scatter_px(corr, mask, CENTRE)
            == bootstrap_center_scatter_px(corr, mask, CENTRE))
