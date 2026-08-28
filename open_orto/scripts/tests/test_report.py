"""Тесты отчётной части: бины гистограмм не теряют значения.

Дефект, который эти тесты закрывают: границы бинов были прописаны руками под
конкретный диапазон параметров, и после смены высот (250–400 → 175–300)
гистограмма показывала почти пустоту — значения просто не попадали ни в
один бин, и ошибка была видна только глазами.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from report import auto_bins, hist_svg  # noqa: E402


def _covered(values, bins):
    v = np.asarray(values, dtype=float)
    return sum(int(((v >= lo) & (v < hi)).sum()) for lo, hi in bins)


@pytest.mark.parametrize("lo,hi", [(175.0, 300.0), (250.0, 400.0), (0.0, 10.0),
                                   (-25.0, 25.0), (0.85, 1.2), (0.6, 1.0)])
def test_auto_bins_cover_every_value(lo, hi):
    """Ни одно значение не остаётся вне шкалы — при любом диапазоне данных."""
    v = np.linspace(lo, hi, 200)
    bins = auto_bins(v)
    assert _covered(v, bins) == len(v)


def test_auto_bins_follow_the_data_not_the_old_range():
    """Шкала строится по фактическим данным: высоты 175–300 не могут попасть
    в бины, начинающиеся с 250."""
    v = np.random.default_rng(0).uniform(176, 300, 300)
    bins = auto_bins(v)
    assert bins[0][0] <= 176.0
    assert bins[-1][1] >= 300.0
    assert _covered(v, bins) == len(v)


def test_auto_bins_reasonable_count():
    v = np.random.default_rng(1).uniform(0, 1, 100)
    assert 3 <= len(auto_bins(v)) <= 10


def test_auto_bins_handle_constant_and_empty():
    assert len(auto_bins(np.full(20, 3.0))) >= 1
    assert _covered(np.full(20, 3.0), auto_bins(np.full(20, 3.0))) == 20
    assert auto_bins(np.array([])) == [(0.0, 1.0)]


def test_hist_marks_values_outside_the_scale():
    """Если бины заданы вручную и не покрывают данные — отчёт обязан сказать
    об этом прямо, а не рисовать пустоту."""
    v = np.array([10.0, 20.0, 300.0, 400.0])
    svg_manual = hist_svg(v, [(0, 50), (50, 100)], "тест")
    assert "ВНЕ ШКАЛЫ: 2" in svg_manual
    svg_auto = hist_svg(v, None, "тест")
    assert "ВНЕ ШКАЛЫ" not in svg_auto
    assert "n=4" in svg_auto
