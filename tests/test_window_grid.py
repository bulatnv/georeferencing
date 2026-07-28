"""Окно точного уровня и плотность сетки: политика, проверенная замером.

Здесь закрепляется не формула, а **результат эксперимента**. Теория говорит, что
перекрытие сетки должно выводиться из запаса окна: окно строится вокруг центра
клетки, значит сетка обязана быть плотнее, чем позволяет запас. Теория проверена
целиком (2026-07-29) и отвергнута: расчётные перекрытия дали 10/16 вместо 11/16 и
потеряли `Volgograd3`, потому что формула не знает про ``top_k`` — при o = 0.84
первые 15 кандидатов оказываются одним местом в 15 копиях.

Тесты пиннят измеренные значения и **направление** ошибки, а не вывод формулы.
"""

from __future__ import annotations

import pytest

from aero_geoloc.localize import (
    MAX_FINE_WINDOW_PX,
    fine_margin_m,
    required_cell_overlap,
)

#: Геометрия реальных кейсов: отпечаток, разрешение подложки, разрешение индекса,
#: измеренно безопасное перекрытие (то, при котором набор даёт 11/16).
CASES = {
    "00049": (124.0, 0.187, 0.35, 0.5),
    "Ufa": (232.0, 0.172, 0.68, 0.5),
    "Saratov": (517.0, 0.197, 1.58, 0.75),
}


def worst_cell_offset_m(footprint_m: float, overlap: float) -> float:
    """Худшее расстояние от центра ближайшей клетки до произвольной точки."""
    return 0.707 * (1.0 - overlap) * footprint_m


@pytest.mark.parametrize("name", sorted(CASES))
def test_overlap_matches_the_measured_policy(name):
    """Значения зафиксированы замером на наборе, а не выведены — менять только с ним."""
    footprint, mpp_fine, _, expected = CASES[name]
    assert required_cell_overlap(footprint, mpp_fine) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("name", sorted(CASES))
def test_theory_would_demand_a_denser_grid_than_measurement_allows(name):
    """Расхождение теории и замера — факт, и он должен быть виден в тестах.

    Если подставить настоящее разрешение индекса, формула требует ~0.84. Так и
    было сделано, набор прогнан целиком — и результат ухудшился. Тест фиксирует
    само расхождение, чтобы следующий читатель не «починил» политику заново.
    """
    footprint, mpp_fine, coarse, measured = CASES[name]
    theoretical = required_cell_overlap(footprint, mpp_fine, coarse_mpp=coarse)
    assert theoretical > measured
    # Теория при этом действительно накрывает окно, а измеренная политика — нет.
    margin = fine_margin_m(footprint, coarse, mpp_fine)
    assert worst_cell_offset_m(footprint, theoretical) <= margin
    assert worst_cell_offset_m(footprint, measured) > margin


def test_window_and_margin_come_from_one_function():
    """Запас окна считается в единственном месте — иначе места разъезжаются.

    Раньше формула жила и в ``_fine_pass``, и, в виде прикидки, внутри
    ``required_cell_overlap``; они расходились втрое. Теперь второе — осознанная
    политика поверх первого, а не вторая независимая формула.
    """
    footprint, mpp_fine, coarse, _ = CASES["Saratov"]
    margin = fine_margin_m(footprint, coarse, mpp_fine)
    assert margin == pytest.approx(max(6.0 * coarse, 0.15 * footprint))
    window_px = (footprint + 2.0 * margin) / mpp_fine
    assert window_px <= MAX_FINE_WINDOW_PX


@pytest.mark.parametrize("limit", [1200, 2000, 4000])
def test_margin_never_exceeds_the_window_limit(limit):
    """Потолок окна жёсткий: запас не может сделать окно шире предела ядра."""
    for footprint, mpp_fine, coarse, _ in CASES.values():
        margin = fine_margin_m(footprint, coarse, mpp_fine, max_window_px=limit)
        window_px = (footprint + 2.0 * margin) / mpp_fine
        assert window_px <= limit + 1e-6 or margin == 0.0


def test_tighter_window_limit_demands_denser_grid():
    """Понизить потолок окна в одиночку нельзя — сетка обязана уплотниться следом."""
    footprint, mpp_fine, _, _ = CASES["Saratov"]
    wide = required_cell_overlap(footprint, mpp_fine, max_window_px=MAX_FINE_WINDOW_PX)
    tight = required_cell_overlap(footprint, mpp_fine, max_window_px=1500)
    assert tight >= wide


def test_overlap_stays_within_sane_bounds():
    """Ниже 0.5 не опускаемся (прямоугольность кадра), выше 0.85 — не имеет смысла."""
    for footprint, mpp_fine, coarse, _ in CASES.values():
        for limit in (800, 2000, 4000, 20000):
            for cm in (None, coarse):
                o = required_cell_overlap(footprint, mpp_fine, coarse_mpp=cm, max_window_px=limit)
                assert 0.5 <= o <= 0.85


def test_degenerate_geometry_is_rejected():
    with pytest.raises(ValueError, match="должны быть > 0"):
        fine_margin_m(0.0, 1.0, 0.2)
    with pytest.raises(ValueError, match="должны быть > 0"):
        required_cell_overlap(200.0, 0.0)
