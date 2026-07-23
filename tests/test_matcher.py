"""Тесты сменного ядра матчинга (фаза 1, ``docs/PLAN.md``).

Главное, что здесь проверяется, — не качество SIFT, а то, что интерфейс
:class:`Matcher` действительно сменный: обе реализации проходят один и тот же
протокол, и ничего кроме :class:`Correspondences` наружу не торчит.
"""

from __future__ import annotations

import importlib.util

import cv2
import numpy as np
import pytest

from aero_geoloc.matcher import (
    AKAZEMatcher,
    Correspondences,
    LightGlueMatcher,
    LoFTRMatcher,
    Matcher,
    SIFTMatcher,
    create_matcher,
)

_HAS_TORCH = importlib.util.find_spec("torch") is not None
from aero_geoloc.testbench import make_synthetic_scene

MATCHER_NAMES = ["sift", "akaze"]


@pytest.fixture(scope="module")
def texture() -> np.ndarray:
    """Текстурная картинка, на которой классика уверенно находит точки.

    Берём ту же процедурную сцену, что и стенд: она специально сделана богатой
    на структуру всех масштабов, и тест не должен мерить бедность самодельного
    шума вместо работы матчера.
    """
    return make_synthetic_scene(512, seed=7).image


# --- Correspondences --------------------------------------------------------


def test_empty_correspondences():
    corr = Correspondences.empty()
    assert len(corr) == 0
    assert corr.pts_q.shape == (0, 2)
    assert corr.conf.shape == (0,)


def test_correspondences_validates_shapes():
    pts = np.zeros((5, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="одной формы"):
        Correspondences(pts_q=pts, pts_r=np.zeros((4, 2), np.float32), conf=np.zeros(5, np.float32))
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        Correspondences(
            pts_q=np.zeros((5, 3), np.float32),
            pts_r=np.zeros((5, 3), np.float32),
            conf=np.zeros(5, np.float32),
        )
    with pytest.raises(ValueError, match=r"conf"):
        Correspondences(pts_q=pts, pts_r=pts, conf=np.zeros(4, np.float32))


def test_correspondences_take_keeps_alignment():
    pts_q = np.arange(10, dtype=np.float32).reshape(5, 2)
    pts_r = pts_q + 100.0
    conf = np.linspace(0.0, 1.0, 5, dtype=np.float32)
    corr = Correspondences(pts_q=pts_q, pts_r=pts_r, conf=conf)

    subset = corr.take(np.array([True, False, True, False, True]))
    assert len(subset) == 3
    np.testing.assert_allclose(subset.pts_r - subset.pts_q, 100.0)
    np.testing.assert_allclose(subset.conf, conf[[0, 2, 4]])


# --- протокол ---------------------------------------------------------------


@pytest.mark.parametrize("name", MATCHER_NAMES)
def test_implementations_satisfy_protocol(name):
    matcher = create_matcher(name)
    assert isinstance(matcher, Matcher)


def test_create_matcher_rejects_unknown_name():
    with pytest.raises(ValueError, match="неизвестный матчер"):
        create_matcher("no_such_matcher")


def test_create_matcher_passes_through_kwargs():
    matcher = create_matcher("sift", ratio=0.6)
    assert matcher.ratio == 0.6


def test_ratio_is_validated():
    with pytest.raises(ValueError, match="ratio"):
        SIFTMatcher(ratio=0.0)
    with pytest.raises(ValueError, match="ratio"):
        SIFTMatcher(ratio=1.5)


# --- поведение на изображениях ----------------------------------------------


@pytest.mark.parametrize("name", MATCHER_NAMES)
def test_self_match_is_identity(name, texture):
    """Картинка сама с собой: соответствия должны совпадать точка-в-точку."""
    corr = create_matcher(name).match(texture, texture)
    assert len(corr) > 50
    np.testing.assert_allclose(corr.pts_q, corr.pts_r, atol=1e-4)
    assert np.all((corr.conf >= 0.0) & (corr.conf <= 1.0))


@pytest.mark.parametrize("name", MATCHER_NAMES)
def test_match_recovers_known_shift(name, texture):
    """Сдвиг окна подложки восстанавливается по медиане невязок."""
    dx, dy = 37, -21
    shifted = np.roll(np.roll(texture, dy, axis=0), dx, axis=1)
    corr = create_matcher(name).match(texture, shifted)

    assert len(corr) > 30
    delta = corr.pts_r - corr.pts_q
    assert np.median(delta[:, 0]) == pytest.approx(dx, abs=1.0)
    assert np.median(delta[:, 1]) == pytest.approx(dy, abs=1.0)


@pytest.mark.parametrize("name", MATCHER_NAMES)
def test_blank_images_yield_no_matches(name):
    """На однородной картинке точек нет — это штатный пустой результат, не падение."""
    blank = np.full((200, 200), 128, dtype=np.uint8)
    assert len(create_matcher(name).match(blank, blank)) == 0


def test_stricter_ratio_yields_fewer_matches(texture):
    rng = np.random.default_rng(1)
    noisy = np.clip(texture.astype(np.int16) + rng.integers(-12, 12, texture.shape), 0, 255).astype(
        np.uint8
    )
    loose = len(SIFTMatcher(ratio=0.9).match(texture, noisy))
    strict = len(SIFTMatcher(ratio=0.5).match(texture, noisy))
    assert strict < loose


def test_max_matches_caps_output_and_keeps_best(texture):
    rng = np.random.default_rng(2)
    noisy = np.clip(texture.astype(np.int16) + rng.integers(-8, 8, texture.shape), 0, 255).astype(
        np.uint8
    )
    capped = SIFTMatcher(max_matches=25).match(texture, noisy)
    full = SIFTMatcher(max_matches=None).match(texture, noisy)
    assert len(capped) == 25
    # Оставлены именно самые уверенные пары.
    assert capped.conf.min() >= np.sort(full.conf)[-25]


@pytest.mark.parametrize("name", MATCHER_NAMES)
def test_rejects_malformed_input(name, texture):
    matcher = create_matcher(name)
    with pytest.raises(ValueError, match="grayscale"):
        matcher.match(cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR), texture)
    with pytest.raises(ValueError, match="uint8"):
        matcher.match(texture.astype(np.float32), texture)
    with pytest.raises(ValueError, match="пустое"):
        matcher.match(np.empty((0, 0), dtype=np.uint8), texture)


def test_akaze_is_available():
    """AKAZE переехал в xfeatures2d в OpenCV 5 — шим обязан это скрывать."""
    assert isinstance(AKAZEMatcher(), Matcher)


# --- обучаемые матчеры фазы 4 (за тем же интерфейсом, gated по torch) --------


@pytest.mark.parametrize("name,cls", [("lightglue", LightGlueMatcher), ("loftr", LoFTRMatcher)])
def test_learned_matcher_constructs_and_registers_without_torch(name, cls):
    """Конструктор и реестр работают без torch — тяжёлое ядро грузится лениво."""
    m = create_matcher(name)
    assert isinstance(m, cls)
    assert isinstance(m, Matcher)  # удовлетворяет протоколу


@pytest.mark.skipif(_HAS_TORCH, reason="torch установлен — боевой путь проверяется отдельно")
@pytest.mark.parametrize("name", ["lightglue", "loftr"])
def test_learned_matcher_without_torch_gives_clear_error(name, texture):
    with pytest.raises(RuntimeError, match="требует"):
        create_matcher(name).match(texture, texture)


@pytest.mark.skipif(not _HAS_TORCH, reason="нужен torch (+ веса)")
def test_lightglue_matches_are_well_formed(texture):
    corr = LightGlueMatcher().match(texture, texture)
    assert isinstance(corr, Correspondences)
    assert corr.pts_q.shape == corr.pts_r.shape
