"""Сборка поставки OrthoLoC: классы разметки и отбор контрольной оси.

Проверяются решения, от которых зависит обучение: чем определяется класс пары
(и почему неизмеренная привязка не считается подтверждённой) и как
прореживается лёгкая ось — по квоте на сцену, а не случайной выборкой по
корпусу, иначе часть территорий осталась бы без контроля забывания.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_ortholoc_dataset as build  # noqa: E402


def test_sample_and_variant_are_read_from_the_name():
    assert build.sample_of("pair_L01_R0012_asis.npz") == "L01_R0012"
    assert build.sample_of("pair_L02_xDOP0156_rect_w.npz") == "L02_xDOP0156"
    assert build.variant_of("L01_R0012") == "R"
    assert build.variant_of("L02_xDOP0156") == "xDOP"
    assert build.variant_of("L03_xDOPDSM0007") == "xDOPDSM"


def test_rect_pairs_are_exact_regardless_of_scene():
    """У rect обе стороны на одной сетке — расходиться нечему."""
    kind, cls, sigma, src = build.classify("rect", "R", "L01", {"L01": 4.0})
    assert (kind, cls, src) == (build.KIND_RECT, "exact", "аналитическая")
    assert sigma == build.SIGMA_RECT


def test_foreign_orthophoto_carries_the_measured_binding_shift():
    """Расхождение источников попадает прямо в ожидаемую ошибку разметки."""
    shifts = {"L20": 2.73}
    kind, cls, sigma, src = build.classify("asis", "xDOP", "L20", shifts)
    assert (kind, cls, src) == (build.KIND_XDOP, "registered", "измерено")
    assert sigma == pytest.approx(2.73)


def test_unmeasured_binding_is_downgraded_not_assumed_good():
    """Сцена без замера привязки — approx, а не registered с оценкой."""
    kind, cls, sigma, src = build.classify("asis", "xDOP", "L99", {"L20": 2.73})
    assert (kind, cls, src) == (build.KIND_XDOP, "approx", "оценка")
    assert sigma == build.SIGMA_FALLBACK
    assert build.WEIGHTS["approx"] < build.WEIGHTS["registered"]


def test_own_orthophoto_gets_the_upper_bound_estimate():
    kind, cls, sigma, src = build.classify("asis", "R", "L01", {})
    assert (kind, cls) == (build.KIND_DOP, "registered")
    assert sigma == build.SIGMA_OWN_DOP
    assert src == "оценка", "оценку нельзя выдавать за измерение"


def test_control_axis_is_thinned_per_scene_not_globally():
    """Квота берётся с каждой сцены: покрытие территорий важнее объёма."""
    rows = ([{"pair": f"a{i}.npz", "mode": "asis", "scene": "L01"} for i in range(80)]
            + [{"pair": f"a{i}.npz", "mode": "asis", "scene": "L02"} for i in range(80, 160)]
            + [{"pair": f"r{i}.npz", "mode": "rect", "scene": "L01"} for i in range(60)]
            + [{"pair": f"r{i}.npz", "mode": "rect", "scene": "L02"} for i in range(60, 63)])

    keep = build.pick_rect(rows, 0.15, np.random.default_rng(0))
    by_scene = {s: sum(1 for r in rows if r["mode"] == "rect"
                       and r["scene"] == s and r["pair"] in keep)
                for s in ("L01", "L02")}

    # цель — 15 % от 160 боевых, то есть 24 пары на две сцены: по 12 с каждой,
    # но у L02 их всего 3 — берутся все, и сцена не остаётся без контроля
    assert by_scene["L01"] == 12
    assert by_scene["L02"] == 3
    assert len(keep) == 15


def test_thinning_keeps_at_least_one_pair_per_scene():
    """Даже при крошечной доле каждая сцена сохраняет контрольную пару."""
    rows = ([{"pair": f"a{i}.npz", "mode": "asis", "scene": "L01"} for i in range(10)]
            + [{"pair": f"r{i}.npz", "mode": "rect", "scene": s}
               for i, s in enumerate(("L01", "L02", "L03"))])

    keep = build.pick_rect(rows, 0.001, np.random.default_rng(0))
    assert len(keep) == 3
