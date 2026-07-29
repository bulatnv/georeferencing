"""Геометрия карты района: свойства, на которых держатся оба потребителя.

Модуль вынесен из харнесса оценки, чтобы инструмент владельца не копировал
логику (`docs/TOOL_PLAN.md`, T1). Копия неизбежно разъезжается — в этом проекте
так уже было с запасом окна и перекрытием сетки, и это стоило кросс-сезонных
кадров. Тесты закрепляют то, ради чего модуль и существует.
"""

from __future__ import annotations

import dataclasses

import pytest

from aero_geoloc.camera import Camera
from aero_geoloc.region import (
    DEFAULT_CELL_PX,
    estimated_build_seconds,
    human_time,
    plan_region,
)
from aero_geoloc.types import Prior


def make(gsd_m: float = 0.05, altitude_m: float = 300.0, yaw: float = 67.0) -> tuple:
    camera = Camera.from_gsd(5472, 3648, gsd_m=gsd_m, altitude_m=altitude_m)
    prior = Prior(lat=54.81, lon=56.09, sigma_m=1500.0, altitude_m=altitude_m,
                  altitude_sigma_m=50.0, yaw_deg=yaw)
    return camera, prior


def plan(**kwargs):
    camera, prior = make(**{k: v for k, v in kwargs.items() if k in ("gsd_m", "altitude_m")})
    rest = {k: v for k, v in kwargs.items() if k not in ("gsd_m", "altitude_m")}
    rest.setdefault("radius_m", 1500.0)
    rest.setdefault("max_zoom", 19)
    rest.setdefault("fine_zoom", 19)
    rest.setdefault("trust_yaw", True)
    return plan_region(camera, prior, **rest)


# --- клетка под кадр --------------------------------------------------------

def test_cell_matches_the_frame_footprint():
    """Клетка обязана быть ≈ отпечатку: иначе эмбеддинги несопоставимы по масштабу."""
    p = plan()
    assert p.cell_px * p.mpp_index == pytest.approx(p.footprint_m, rel=0.02)


@pytest.mark.parametrize("gsd_m", [0.02, 0.05, 0.12])
def test_zoom_adapts_to_the_footprint(gsd_m):
    """Зум подбирается ПОД КАДР, а не фиксируется.

    Отпечаток гуляет в разы между кадрами; фиксированный зум либо раздул бы
    число клеток, либо потребовал бы качать тайлы гигабайтами.
    """
    p = plan(gsd_m=gsd_m)
    assert 0.5 * DEFAULT_CELL_PX <= p.cell_px <= 2.0 * DEFAULT_CELL_PX


def test_bigger_footprint_gets_coarser_zoom():
    """Отпечаток ∝ высоте при фиксированном FOV — именно так меняется реальный кадр.

    (Через ``Camera.from_gsd`` высоту менять бесполезно: GSD задан, и отпечаток
    от неё не зависит — эта тонкость стоила одного неверного теста.)
    """
    prior_lo = Prior(lat=54.81, lon=56.09, sigma_m=1500.0, altitude_m=150.0,
                     altitude_sigma_m=50.0, yaw_deg=0.0)
    prior_hi = dataclasses.replace(prior_lo, altitude_m=600.0)
    camera = Camera(5472, 3648, fov_deg=73.0)
    common = dict(radius_m=1500.0, max_zoom=19, fine_zoom=19, trust_yaw=True)
    low = plan_region(camera, prior_lo, **common)
    high = plan_region(camera, prior_hi, **common)
    assert high.footprint_m > 3.5 * low.footprint_m
    assert high.region.zoom < low.region.zoom


# --- цена, видимая ДО сборки ------------------------------------------------

def test_unknown_heading_costs_eightfold():
    """Незнание курса — не мелочь: индекс аугментируется поворотами."""
    known, unknown = plan(trust_yaw=True), plan(trust_yaw=False)
    assert len(known.rotations_deg) == 1
    assert len(unknown.rotations_deg) == 8
    assert unknown.cells == 8 * known.cells


def test_estimate_and_wording_are_usable():
    """Оценка нужна не для точности, а чтобы владелец знал: минуты или секунды."""
    p = plan(trust_yaw=False)
    assert estimated_build_seconds(p) > 0
    assert human_time(45) == "45 с"
    assert human_time(600) == "10 мин"
    assert human_time(5400) == "1 ч 30 мин"
    text = p.describe()
    assert "клеток" in text and "перекрытие" in text and "поворотов" in text


# --- ключ кэша --------------------------------------------------------------

CHANGES = [
    ("radius_m", 2000.0),
    ("cell_px_target", 500),
    ("overlap", 0.8),
    ("pca_dim", 512),
    ("trust_yaw", False),
]


@pytest.mark.parametrize("field,value", CHANGES)
def test_cache_key_covers_everything_that_changes_content(field, value):
    """ГЛАВНЫЙ тест модуля.

    Перекрытие однажды забыли включить в ключ — и прогон с другой плотностью
    сетки молча взял старую карту, «доказав», что изменение не помогло. Тихое
    переиспользование хуже лишней пересборки, поэтому любое из этих полей обязано
    менять имя файла.
    """
    base = plan()
    other = plan(**{field: value})
    assert base.path != other.path, f"{field} не попал в ключ кэша"


def test_same_geometry_reuses_one_map():
    """Снимки одной серии делят карту: ключ по геометрии, а не по имени кадра."""
    assert plan().path == plan().path


def test_cache_dir_and_prefix_are_respected(tmp_path):
    p = plan(cache_dir=tmp_path, prefix="tool")
    assert p.path.parent == tmp_path and p.path.name.startswith("tool_")
    assert not p.cached


# --- связка с окном точного уровня ------------------------------------------

def test_overlap_follows_the_window_policy():
    """Перекрытие не выдумывается здесь — оно приходит из политики окна."""
    from aero_geoloc.localize import required_cell_overlap

    p = plan()
    assert p.overlap == pytest.approx(required_cell_overlap(p.footprint_m, 0.1721), abs=0.3)


def test_explicit_overlap_wins():
    assert plan(overlap=0.9).overlap == pytest.approx(0.9)


def test_degenerate_radius_is_rejected():
    with pytest.raises(ValueError, match="radius_m"):
        plan(radius_m=0.0)
