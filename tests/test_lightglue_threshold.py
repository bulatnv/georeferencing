"""Четыре ручки LightGlue: именованные константы, проводка и обратное чтение.

Тесты по спецификации ``docs/RESEARCH_A_LIGHTGLUE_RECHECK.md`` (Г1/Г2/Г5).
Особенность этой линии: наш внешний ``min_score = 0.0`` — no-op, и это усыпляет;
настоящие пороги сидят внутри пакета, включая прунинг, который удаляет точки из
вычисления безвозвратно. Числа — из ``default_conf`` пакета cvg/LightGlue, не
«как получилось».
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from aero_geoloc.matcher import (
    LIGHTGLUE_DEPTH_CONFIDENCE,
    LIGHTGLUE_FILTER_THRESHOLD,
    LIGHTGLUE_WIDTH_CONFIDENCE,
    LOFTR_MIN_CONF,
    ROMAV2_MIN_OVERLAP,
    SUPERPOINT_DETECTION_THRESHOLD,
    LightGlueMatcher,
)


def _default(name):
    return inspect.signature(LightGlueMatcher.__init__).parameters[name].default


# --- Г1: константы — числа пакета, дефолты идут от них ------------------------

def test_constants_are_the_package_numbers():
    assert LIGHTGLUE_FILTER_THRESHOLD == 0.1
    assert LIGHTGLUE_DEPTH_CONFIDENCE == 0.95
    assert LIGHTGLUE_WIDTH_CONFIDENCE == 0.99
    assert SUPERPOINT_DETECTION_THRESHOLD == 0.0005


def test_defaults_come_from_the_named_constants():
    assert _default("filter_threshold") == LIGHTGLUE_FILTER_THRESHOLD
    assert _default("depth_confidence") == LIGHTGLUE_DEPTH_CONFIDENCE
    assert _default("width_confidence") == LIGHTGLUE_WIDTH_CONFIDENCE
    assert _default("detection_threshold") == SUPERPOINT_DETECTION_THRESHOLD


def test_kernel_thresholds_do_not_alias_each_other():
    """Пороги трёх линий — разные величины разных калибровок; совпадение значений
    допустимо только случайное, ссылок друг на друга быть не должно."""
    assert LIGHTGLUE_FILTER_THRESHOLD not in (LOFTR_MIN_CONF, ROMAV2_MIN_OVERLAP)


def test_external_min_score_stays_a_noop_by_default():
    """0.0 — исторический no-op; менять его молча нельзя, история LoFTR учит,
    чем кончается «внешний фильтр на всякий случай»."""
    assert _default("min_score") == 0.0


# --- Г2: обратное чтение фактических порогов ----------------------------------

def _fake_confs(**over):
    matcher_conf = SimpleNamespace(filter_threshold=0.1, depth_confidence=0.95,
                                   width_confidence=0.99)
    extractor_conf = SimpleNamespace(detection_threshold=0.0005)
    for k, v in over.items():
        if k == "detection_threshold":
            setattr(extractor_conf, k, v)
        else:
            setattr(matcher_conf, k, v)
    return matcher_conf, extractor_conf


def test_effective_values_are_read_back_not_assumed():
    m = LightGlueMatcher()
    eff = m._read_effective(*_fake_confs(filter_threshold=0.0, width_confidence=-1))
    assert eff["filter_threshold"] == 0.0
    assert eff["width_confidence"] == -1.0
    assert eff["detection_threshold"] == 0.0005


def test_missing_matcher_key_raises_not_noops():
    m = LightGlueMatcher()
    conf = SimpleNamespace(filter_threshold=0.1, depth_confidence=0.95)  # нет width
    with pytest.raises(RuntimeError, match="width_confidence"):
        m._read_effective(conf, SimpleNamespace(detection_threshold=0.0005))


def test_missing_detector_key_raises_not_noops():
    m = LightGlueMatcher()
    mc, _ = _fake_confs()
    with pytest.raises(RuntimeError, match="detection_threshold"):
        m._read_effective(mc, SimpleNamespace())
