"""Пороги overlap у RoMa v1 и v2 — разные величины, и константы обязаны быть разными.

Тест существует из-за конкретной ловушки (``docs/RESEARCH_A_ROMAV2_RECHECK.md``):
``min_conf = 0.5``, унаследованный v2 от v1, обрезал распределение ``overlap``,
живущее у v2 в сотых, — и модель была записана как «0 поз на всех кейсах», не
будучи измеренной вовсе. У v1 тот же порог безобиден: его ``conf`` после
``sample`` — константа 1.0, и фильтр — no-op.

Числа здесь — из статьи (arXiv:2511.15706, §4.1, ур. 6), а не «как получилось»:
принцип тестирования из ``CLAUDE.md``.
"""

from __future__ import annotations

import inspect

from aero_geoloc.matcher import ROMAV2_MIN_OVERLAP, RoMaMatcher, RoMaV2Matcher


def _default(cls, name):
    return inspect.signature(cls.__init__).parameters[name].default


def test_romav2_threshold_is_the_papers_number():
    """Ур. 6 статьи: p̂ = max(1[p > 0.05], p). Авторский порог — 0.05."""
    assert ROMAV2_MIN_OVERLAP == 0.05


def test_v2_default_is_the_paper_constant_not_v1_inheritance():
    """Сама ловушка: 0.5 у v2 глушит всё. Дефолт обязан идти от константы статьи."""
    assert _default(RoMaV2Matcher, "min_conf") == ROMAV2_MIN_OVERLAP
    assert _default(RoMaV2Matcher, "min_conf") != 0.5


def test_v1_and_v2_thresholds_are_separate_constants():
    """У v1 conf — константа 1.0 и его 0.5 — no-op; наследование в любую сторону
    молча меняет смысл фильтра. Пороги обязаны различаться и не ссылаться друг
    на друга."""
    assert _default(RoMaMatcher, "min_conf") == 0.5          # документированный no-op
    assert _default(RoMaMatcher, "min_conf") != _default(RoMaV2Matcher, "min_conf")


def test_model_threshold_defaults_to_the_paper_flattening():
    """Ур. 6 — это не только фильтр, но и уплощение весов сэмплирования в модели.
    По умолчанию включено авторское значение; None остаётся доступным для
    измерений сырого поля."""
    assert _default(RoMaV2Matcher, "model_threshold") == ROMAV2_MIN_OVERLAP


def test_pair_precision_capture_is_opt_in():
    """Попарные Σ⁻¹ — только по явной просьбе эксперимента: (N,2,2)-массивам не
    место в evidence и диагностике штатного тракта."""
    assert _default(RoMaV2Matcher, "keep_pair_precision") is False
