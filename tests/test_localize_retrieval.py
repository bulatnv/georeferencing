"""Тесты интеграции retrieval в грубый уровень `localize` (фаза 3, шаг 3).

Проверяют, что путь через индекс сохраняет точность фазы 2 (точный уровень тот
же), честно отказывает по низкой уникальности и корректно отражён в диагностике.
"""

from __future__ import annotations

import numpy as np
import pytest

from aero_geoloc.geo import haversine_m
from aero_geoloc.localize import localize
from aero_geoloc.retrieval import AveragePoolEncoder, TerrainIndex
from aero_geoloc.testbench import (
    SampleSpec,
    SceneBasemap,
    default_camera,
    generate_sample,
    make_homogeneous_scene,
    make_synthetic_scene,
)

CELL = 512


@pytest.fixture(scope="module")
def scene():
    return make_synthetic_scene(3072, seed=0)


@pytest.fixture(scope="module")
def camera():
    return default_camera(512)


@pytest.fixture(scope="module")
def index(scene):
    return TerrainIndex(AveragePoolEncoder(24)).build(
        SceneBasemap(scene), scene.georef, cell_size_px=CELL, overlap=0.5, rotations_deg=(0.0,)
    )


def test_localize_via_index_preserves_accuracy(scene, camera, index):
    """Грубый уровень через retrieval → точный уровень тот же → субпиксель сохранён."""
    bm = SceneBasemap(scene)
    ok = 0
    errs = []
    rng = np.random.default_rng(0)
    for ox, oy in [(0, 0), (300, -200), (-250, 300), (200, 200)]:
        yaw = float(rng.uniform(0, 360))
        s = generate_sample(scene, camera, SampleSpec(yaw_deg=yaw, center_offset_px=(ox, oy)),
                            prior_sigma_m=60.0, reference_size=1024)
        r = localize(s.query, camera, s.prior, bm, index=index)
        if r.is_localized:
            ok += 1
            errs.append(haversine_m(s.true_lat, s.true_lon, r.center_lat, r.center_lon))
            assert r.diagnostics["retrieval"] is True
    assert ok >= 3  # retrieval находит место
    assert np.median(errs) < 0.1  # субметровая точность сохранена (доли см)


def test_localize_index_diagnostics_expose_retrieval_signals(scene, camera, index):
    bm = SceneBasemap(scene)
    s = generate_sample(scene, camera, SampleSpec(yaw_deg=30.0), prior_sigma_m=60.0, reference_size=1024)
    r = localize(s.query, camera, s.prior, bm, index=index)
    assert r.diagnostics["retrieval"] is True
    assert "retrieval_uniqueness" in r.diagnostics
    assert r.diagnostics["retrieval_returned"] >= 1


def test_localize_index_refuses_on_low_uniqueness(scene, camera, index):
    """Порог уникальности выше достижимого → честный отказ до матчинга."""
    bm = SceneBasemap(scene)
    s = generate_sample(scene, camera, SampleSpec(yaw_deg=30.0), prior_sigma_m=60.0, reference_size=1024)
    r = localize(s.query, camera, s.prior, bm, index=index, min_uniqueness=0.99)
    assert not r.is_localized
    assert "уникальность" in r.diagnostics.get("reason", "")


def test_localize_index_refuses_on_homogeneous_terrain(camera):
    """Самоподобная местность: и retrieval, и матчер должны отказать, не угадывать."""
    homo = make_homogeneous_scene(3072, seed=0)
    bm = SceneBasemap(homo)
    hidx = TerrainIndex(AveragePoolEncoder(24)).build(
        bm, homo.georef, cell_size_px=CELL, overlap=0.5
    )
    s = generate_sample(homo, camera, SampleSpec(yaw_deg=45.0), prior_sigma_m=60.0, reference_size=1024)
    # Даже без порога уникальности точный уровень не соберёт модель на бедной текстуре.
    r = localize(s.query, camera, s.prior, bm, index=hidx, min_uniqueness=0.0)
    assert not r.is_localized


def test_retrieval_candidates_carry_cell_rotation(scene, camera):
    """Кандидат несёт угол клетки — из него восстанавливается неизвестный курс.

    При неизвестном yaw индекс аугментируют повёрнутыми копиями; совпавшая клетка
    знает свой угол ``R`` (``rotate(клетка, R) ≈ кадр``), и это единственная оценка
    курса, которая есть у системы. Без неё Этаж 2 получил бы кадр под произвольным
    поворотом (см. `_retrieval_candidates`).
    """
    from aero_geoloc.localize import _retrieval_candidates, normalize_gray

    rotations = (0.0, 90.0, 180.0, 270.0)
    idx = TerrainIndex(AveragePoolEncoder(24)).build(
        SceneBasemap(scene), scene.georef, cell_size_px=CELL, overlap=0.5, rotations_deg=rotations
    )
    s = generate_sample(scene, camera, SampleSpec(yaw_deg=90.0), prior_sigma_m=400.0,
                        reference_size=1024)
    candidates, diag = _retrieval_candidates(
        idx, normalize_gray(s.query), s.prior, top_k=8, trust_yaw=False,
        min_uniqueness=0.0, gate_sigma=3.0,
    )
    assert candidates, diag
    assert all(len(c) == 4 for c in candidates)  # (lon, lat, coarse_mpp, rotation)
    assert {c[3] for c in candidates} <= set(rotations)


def test_prerotate_flag_governs_use_of_cell_rotation(scene, camera):
    """Флаг ``prerotate`` остаётся хозяином: для инвариантного матчера поворота нет.

    Иначе изменение ради не-инвариантных ядер втихую меняло бы путь SIFT.
    """
    bm = SceneBasemap(scene)
    idx = TerrainIndex(AveragePoolEncoder(24)).build(
        bm, scene.georef, cell_size_px=CELL, overlap=0.5, rotations_deg=(0.0, 90.0)
    )
    s = generate_sample(scene, camera, SampleSpec(yaw_deg=90.0), prior_sigma_m=400.0,
                        reference_size=1024)
    seen: list[float] = []
    import importlib

    # Не `import aero_geoloc.localize as loc`: в пакете имя `localize`
    # переэкспортировано как ФУНКЦИЯ и затеняет одноимённый модуль.
    loc = importlib.import_module("aero_geoloc.localize")
    original = loc._fine_with_scale_loop

    def spy(*args, **kwargs):
        seen.append(kwargs.get("prerotate_deg"))
        return original(*args, **kwargs)

    loc._fine_with_scale_loop = spy
    try:
        loc.localize(s.query, camera, s.prior, bm, index=idx, trust_yaw=False, prerotate=False)
        assert set(seen) == {0.0}  # инвариантный матчер — кадр не крутим
        seen.clear()
        loc.localize(s.query, camera, s.prior, bm, index=idx, trust_yaw=False, prerotate=True)
        assert seen and set(seen) <= {0.0, -90.0}  # углы взяты у клеток (−R)
    finally:
        loc._fine_with_scale_loop = original


def test_localize_window_path_unchanged_without_index(scene, camera):
    """Без индекса — прежний путь по окну (регресс поведения фазы 2)."""
    bm = SceneBasemap(scene)
    s = generate_sample(scene, camera, SampleSpec(yaw_deg=137.0), prior_sigma_m=60.0, reference_size=1024)
    r = localize(s.query, camera, s.prior, bm)  # index=None
    assert r.diagnostics["retrieval"] is False
    assert r.is_localized
