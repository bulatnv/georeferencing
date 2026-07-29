"""Свидетельства уровня пары: доходят до связки качества и не теряются по дороге.

Зачем они. Не всё, что знает матчер, ложится на отдельные точки. Плотное ядро
оценивает уверенность по ВСЕМУ полю, и доля уверенной площади — свойство пары
картинок, а не какой-то выборки точек. Такому знанию некуда деться в
``conf``, поэтому у :class:`Correspondences` есть ``evidence``.

Отдельно проверяется, что свидетельства **переживают преобразования по дороге**:
именно на этом они однажды и пропали — предповорот кадра пересобирал
``Correspondences`` и терял их ровно у тех кейсов, где известен курс.
"""

from __future__ import annotations

import numpy as np
import pytest

from aero_geoloc.localize import _match_prerotated
from aero_geoloc.matcher import Correspondences, ResizedMatcher


def corr(n: int = 4, evidence: dict | None = None) -> Correspondences:
    return Correspondences(
        pts_q=np.arange(2 * n, dtype=np.float32).reshape(n, 2),
        pts_r=np.arange(2 * n, dtype=np.float32).reshape(n, 2) + 5.0,
        conf=np.ones(n, np.float32),
        evidence=evidence if evidence is not None else {"certainty_mean": 0.31},
    )


class Fixed:
    """Матчер, всегда отдающий одно и то же — проверяем именно перенос."""

    def __init__(self, out: Correspondences) -> None:
        self.out = out

    def match(self, query_gray, ref_gray) -> Correspondences:
        return self.out


def test_evidence_is_empty_by_default():
    """Разреженным ядрам нечего сюда класть, и это штатный случай."""
    plain = Correspondences(np.zeros((2, 2), np.float32), np.zeros((2, 2), np.float32),
                            np.ones(2, np.float32))
    assert plain.evidence == {}


def test_filtering_points_keeps_pair_level_evidence():
    """Свидетельства не про точки: отбор точек их не меняет."""
    kept = corr(4).take(np.array([True, False, True, False]))
    assert len(kept) == 2
    assert kept.evidence == {"certainty_mean": 0.31}


def test_prerotation_keeps_evidence():
    """Тот самый случай, где они пропадали: предповорот меняет координаты, не знание."""
    out = _match_prerotated(Fixed(corr()), np.zeros((64, 64), np.uint8),
                            np.zeros((64, 64), np.uint8), 30.0)
    assert out.evidence == {"certainty_mean": 0.31}


def test_prerotation_by_zero_is_a_plain_match():
    out = _match_prerotated(Fixed(corr()), np.zeros((64, 64), np.uint8),
                            np.zeros((64, 64), np.uint8), 0.0)
    assert out.evidence == {"certainty_mean": 0.31}


def test_resizing_wrapper_keeps_evidence():
    """Обёртка рабочего разрешения тоже пересобирает соответствия."""
    wrapped = ResizedMatcher(Fixed(corr()), max_side=32, pad_to=1)
    out = wrapped.match(np.zeros((64, 64), np.uint8), np.zeros((64, 64), np.uint8))
    assert out.evidence == {"certainty_mean": 0.31}


def test_evidence_copies_are_independent():
    """Перенос делается копией: чужой словарь не должен мутировать наш."""
    source = corr()
    kept = source.take(np.array([True, True, False, False]))
    kept.evidence["certainty_mean"] = 0.99
    assert source.evidence["certainty_mean"] == pytest.approx(0.31)


def test_assess_exposes_evidence_in_signals():
    """Связка обязана показать свидетельства в диагностике, даже если не решает по ним."""
    from aero_geoloc.pose import PoseEstimate, SimilarityTransform
    from aero_geoloc.quality import assess

    rng = np.random.default_rng(0)
    pts_q = rng.uniform(0, 100, (30, 2)).astype(np.float32)
    c = Correspondences(pts_q, pts_q + 0.1, np.ones(30, np.float32),
                        evidence={"certainty_mean": 0.42, "certainty_cover": 0.07})
    pose = PoseEstimate(SimilarityTransform.from_params(1.0, 0.0, 0.1, 0.1),
                        np.ones(30, bool), 0.1)
    signals = assess(pose, c, (50.0, 50.0), mpp=0.3, photometric=0.6).signals
    assert signals["certainty_mean"] == pytest.approx(0.42)
    assert signals["certainty_cover"] == pytest.approx(0.07)
