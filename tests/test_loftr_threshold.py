"""Пороги LoFTR: именованные константы, проводка coarse_thr и evidence при нуле.

Тесты существуют из-за спецификации ``docs/RESEARCH_A_LOFTR_RECHECK.md``
(Л1/Л2/Л4/Л5): порог dual-softmax — свойство обучения, и константа одного ядра
не должна молча применяться к другому; внутренний порог kornia обязан быть
видимым и проверяемо доезжать до модели; сводка сигнала не должна теряться
ровно в том случае, ради которого заводилась. Числа — из kornia-конфига (0.2)
и из истории проекта, а не «как получилось».
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from aero_geoloc.matcher import (
    LOFTR_MIN_CONF,
    ROMAV2_MIN_OVERLAP,
    Correspondences,
    LoFTRMatcher,
)


def _default(cls, name):
    return inspect.signature(cls.__init__).parameters[name].default


# --- Л1: константы разведены --------------------------------------------------

def test_loftr_threshold_is_a_named_constant():
    assert _default(LoFTRMatcher, "min_conf") == LOFTR_MIN_CONF


def test_loftr_and_romav2_thresholds_are_separate():
    """Ловушка «порог одного ядра у другого» уже стоила двух неверных выводов —
    константы не должны ссылаться друг на друга."""
    assert LOFTR_MIN_CONF != ROMAV2_MIN_OVERLAP


def test_coarse_thr_default_leaves_kornia_alone():
    """None = не трогать дефолт пакета: исторические замеры сняты именно так."""
    assert _default(LoFTRMatcher, "coarse_thr") is None


# --- Л2: проводка порога проверяема без весов --------------------------------

class _FakeCoarse:
    def __init__(self, thr=0.2):
        self.thr = thr


class _FakeModel:
    def __init__(self):
        self.coarse_matching = _FakeCoarse()


def test_coarse_thr_actually_reaches_the_model():
    m = LoFTRMatcher(coarse_thr=0.0)
    fake = _FakeModel()
    m._apply_coarse_thr(fake)
    assert fake.coarse_matching.thr == 0.0
    assert m.effective_coarse_thr == 0.0


def test_untouched_coarse_thr_is_still_recorded():
    """Фактический порог пишется и когда мы его не меняли — kornia-дефолт 0.2."""
    m = LoFTRMatcher()
    m._apply_coarse_thr(_FakeModel())
    assert m.effective_coarse_thr == 0.2


def test_missing_thr_attribute_raises_not_noops():
    """Присваивание атрибута nn.Module всегда успешно — молчаливый no-op дал бы
    «мы сняли порог» при неснятом пороге. Смена версии kornia обязана падать."""
    m = LoFTRMatcher(coarse_thr=0.0)

    class NoThr:
        pass

    with pytest.raises(RuntimeError, match="coarse_matching.thr"):
        m._apply_coarse_thr(NoThr())


# --- Л4: evidence переживает ноль пар ----------------------------------------

def test_empty_correspondences_keep_evidence():
    ev = {"n_model_out": 0, "conf_max": float("nan")}
    corr = Correspondences.empty(ev)
    assert len(corr) == 0
    assert corr.evidence["n_model_out"] == 0


def test_empty_without_evidence_stays_backward_compatible():
    corr = Correspondences.empty()
    assert len(corr) == 0 and corr.evidence == {}


def test_empty_copies_the_dict():
    """Чужой словарь не должен становиться разделяемым состоянием."""
    ev = {"a": 1}
    corr = Correspondences.empty(ev)
    ev["a"] = 2
    assert corr.evidence["a"] == 1


def test_take_still_carries_evidence():
    corr = Correspondences(
        np.zeros((3, 2), np.float32), np.zeros((3, 2), np.float32),
        np.array([0.1, 0.6, 0.9], np.float32), evidence={"n_model_out": 3})
    sub = corr.take(corr.conf > 0.5)
    assert len(sub) == 2 and sub.evidence["n_model_out"] == 3
