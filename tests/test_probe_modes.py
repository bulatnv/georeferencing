"""Аналитические эталоны для probe_modes (RESEARCH_F §3, И2).

Метрики argmax_hit_frac/warp_hit_frac считаются одним кодом для всех ядер —
значит этот код обязан быть проверен против ручной геометрии, а не «как
получилось». Особо: anchor_grid зеркалит формулу ``cls_to_flow_refine`` из
``romatch`` — тест фиксирует порядок осей (x, y) и row-major развёртку якорей,
чтобы рассинхрон с пакетом не прошёл молча.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from probe_modes import (  # noqa: E402
    anchor_grid,
    argmax_targets,
    central_crop,
    expected_target_norm,
    hit_frac,
    pool_to_grid,
    sample_field_at_patches,
    top_frac_mask,
)


# --- expected_target_norm ----------------------------------------------------

def test_identity_pair_expects_own_patch_centres():
    """Тождество (s=1, стороны равны): ожидание — центры патчей B,
    та же сетка -1 + (2i+1)/n, что у romatch в match()."""
    exp = expected_target_norm(4, 4, 1.0, 800, 800)
    lin = -1 + (2 * np.arange(4) + 1) / 4
    assert np.allclose(exp[0, :, 0], lin)      # x по столбцам
    assert np.allclose(exp[:, 0, 1], lin)      # y по строкам
    assert np.allclose(exp[2, 3], [lin[3], lin[2]])


def test_scale_mismatch_maps_through_centre():
    """Э2.3: подложка вдвое грубее (s=0.5, окно вдвое меньше в px) — центры
    патчей отображаются в те же нормированные позиции: геометрия подобия
    вокруг общего центра сохраняет относительные положения."""
    exp = expected_target_norm(4, 4, 0.5, 800, 400)
    ident = expected_target_norm(4, 4, 1.0, 800, 800)
    assert np.allclose(exp, ident, atol=1e-9)


def test_scale_mismatch_asymmetric_window():
    """Окно не в масштабе (s=0.5, но окно того же px-размера): патчи A должны
    съехать к центру B ровно вдвое по нормированной оси."""
    exp = expected_target_norm(2, 2, 0.5, 800, 800)
    ident = expected_target_norm(2, 2, 1.0, 800, 800)
    assert np.allclose(exp, ident * 0.5, atol=1e-3)


# --- hit_frac ----------------------------------------------------------------

def test_hit_frac_counts_within_tolerance():
    expected = expected_target_norm(4, 4, 1.0, 800, 800)
    pred = expected.copy()
    pred[0, 0] += 10.0                      # один патч мимо
    assert hit_frac(pred, expected, tol_norm=2 / 4) == pytest.approx(15 / 16)


def test_hit_frac_tolerance_is_chebyshev_one_patch():
    expected = np.zeros((1, 1, 2))
    exactly_one_patch = np.array([[[2 / 8, 2 / 8]]])
    just_over = np.array([[[2 / 8 + 1e-6, 0.0]]])
    assert hit_frac(exactly_one_patch, expected, 2 / 8) == 1.0
    assert hit_frac(just_over, expected, 2 / 8) == 0.0


def test_hit_frac_weighted_subset():
    expected = np.zeros((2, 2, 2))
    pred = np.zeros((2, 2, 2))
    pred[1, 1] = 5.0                        # промах в невзвешенном патче
    mask = np.array([[True, True], [True, False]])
    assert hit_frac(pred, expected, 0.1, mask) == 1.0
    assert hit_frac(pred, expected, 0.1) == pytest.approx(3 / 4)


# --- anchor_grid: зеркало cls_to_flow_refine ---------------------------------

def test_anchor_grid_matches_romatch_formula():
    """G = meshgrid(linspace(-1+1/res, 1-1/res, res), indexing='ij'),
    stack([G[1], G[0]]) → (x, y), row-major: k = row * res + col."""
    g = anchor_grid(16)                     # res = 4
    lin = np.linspace(-1 + 1 / 4, 1 - 1 / 4, 4)
    assert g.shape == (16, 2)
    assert np.allclose(g[0], [lin[0], lin[0]])
    assert np.allclose(g[1], [lin[1], lin[0]])   # сосед по k — шаг по x
    assert np.allclose(g[4], [lin[0], lin[1]])   # шаг на res — шаг по y
    assert np.allclose(g[15], [lin[3], lin[3]])


def test_anchor_grid_rejects_non_square():
    with pytest.raises(ValueError):
        anchor_grid(17)


# --- argmax_targets ----------------------------------------------------------

def test_argmax_targets_patch_to_patch():
    """4D S (v2): максимум в патче B (по строке m) даёт центр этого патча."""
    s = np.zeros((2, 2, 4, 4))
    s[0, 0, 2, 3] = 1.0                     # патч A(0,0) → B(row=2, col=3)
    t = argmax_targets(s)
    lin = -1 + (2 * np.arange(4) + 1) / 4
    assert np.allclose(t[0, 0], [lin[3], lin[2]])


def test_argmax_targets_anchor_mode_matches_grid():
    """3D S (v1): argmax по якорям — координата из anchor_grid."""
    s = np.zeros((1, 1, 16))
    s[0, 0, 6] = 1.0
    t = argmax_targets(s)
    assert np.allclose(t[0, 0], anchor_grid(16)[6])


def test_identity_diagonal_scores_full_hit():
    """Диагональная S — идеальное сходство: argmax_hit_frac = 1 на тождестве."""
    n = 6
    s = np.zeros((n, n, n, n))
    for i in range(n):
        for j in range(n):
            s[i, j, i, j] = 1.0
    exp = expected_target_norm(n, n, 1.0, 800, 800)
    assert hit_frac(argmax_targets(s), exp, 2 / n) == 1.0


# --- вспомогательные ---------------------------------------------------------

def test_top_frac_mask_selects_highest():
    score = np.arange(100, dtype=float).reshape(10, 10)
    mask = top_frac_mask(score, 0.10)
    assert mask.sum() == 10 and mask.reshape(-1)[-10:].all()


def test_pool_to_grid_averages_blocks():
    field = np.zeros((8, 8))
    field[:4, :4] = 1.0
    pooled = pool_to_grid(field, 2, 2)
    assert np.allclose(pooled, [[1.0, 0.0], [0.0, 0.0]])


def test_sample_field_at_patches_takes_centres():
    warp = np.zeros((8, 8, 2))
    warp[2, 6] = (0.5, -0.5)               # центр патча (0, 1) сетки 2x2 → px (2, 6)
    at = sample_field_at_patches(warp, 2, 2)
    assert np.allclose(at[0, 1], [0.5, -0.5])


def test_central_crop_keeps_centre():
    img = np.arange(36).reshape(6, 6)
    c = central_crop(img, 2)
    assert c.shape == (2, 2) and c[0, 0] == 14


def test_central_crop_noop_when_smaller():
    img = np.zeros((4, 4))
    assert central_crop(img, 10).shape == (4, 4)
