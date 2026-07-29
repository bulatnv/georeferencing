"""Обёртка рабочего разрешения: проверяем ровно ту ловушку, что уже срабатывала.

Дважды на одном LoFTR (`docs/JOURNAL.md`) неверный вход давал неверный вывод о
самом матчере. Тесты фиксируют оба свойства, которых тогда не хватило: общий
коэффициент для обеих картинок и возврат координат в ИСХОДНЫЕ пиксели.
"""

from __future__ import annotations

import numpy as np
import pytest

from aero_geoloc.matcher import Correspondences, ResizedMatcher, create_matcher


class SpyMatcher:
    """Запоминает, что ему подали, и отдаёт заранее заданные соответствия."""

    def __init__(self, corr: Correspondences | None = None) -> None:
        self.seen: list[tuple[int, int]] = []
        self.corr = corr or Correspondences.empty()

    def match(self, query_gray, ref_gray) -> Correspondences:
        self.seen = [query_gray.shape[:2], ref_gray.shape[:2]]
        return self.corr


def test_both_images_share_one_scale_factor():
    """Суть ловушки: кадр и окно покрывают разную площадь земли.

    Приведение каждого к 640 по отдельности вносит расхождение масштабов (у нас
    было 1.5×) и рушит модель подобия. Коэффициент обязан быть общим, поэтому
    отношение сторон между картинками сохраняется.
    """
    spy = SpyMatcher()
    ResizedMatcher(spy, max_side=640, pad_to=1).match(
        np.zeros((500, 500), np.uint8), np.zeros((2000, 2000), np.uint8))
    (qh, qw), (rh, rw) = spy.seen
    assert (rh, rw) == (640, 640)
    assert (qh, qw) == (160, 160)          # 500 * 640/2000 — тот же коэффициент
    assert rh / qh == pytest.approx(2000 / 500)


def test_points_come_back_in_original_pixels():
    """Всё выше матчера работает в пикселях входа и не должно знать про ресайз."""
    corr = Correspondences(pts_q=np.array([[0.0, 0.0], [100.0, 50.0]], np.float32),
                           pts_r=np.array([[10.0, 20.0], [110.0, 70.0]], np.float32),
                           conf=np.ones(2, np.float32))
    wrapped = ResizedMatcher(SpyMatcher(corr), max_side=500, pad_to=1)
    out = wrapped.match(np.zeros((1000, 1000), np.uint8), np.zeros((1000, 1000), np.uint8))
    assert out.pts_q == pytest.approx(np.array([[0.0, 0.0], [200.0, 100.0]]))
    assert out.pts_r == pytest.approx(np.array([[20.0, 40.0], [220.0, 140.0]]))
    assert out.conf == pytest.approx(np.ones(2))


def test_small_images_pass_through_untouched():
    """Увеличивать нечего и незачем — вход меньше рабочего размера идёт как есть."""
    spy = SpyMatcher()
    ResizedMatcher(spy, max_side=640, pad_to=1).match(
        np.zeros((100, 120), np.uint8), np.zeros((200, 150), np.uint8))
    assert spy.seen == [(100, 120), (200, 150)]


def test_padding_keeps_multiple_of_eight_and_does_not_shift_points():
    """LoFTR внутри делит на 8. Дополнение справа/снизу не сдвигает координаты."""
    corr = Correspondences(pts_q=np.array([[3.0, 4.0]], np.float32),
                           pts_r=np.array([[5.0, 6.0]], np.float32),
                           conf=np.ones(1, np.float32))
    spy = SpyMatcher(corr)
    out = ResizedMatcher(spy, max_side=0, pad_to=8).match(
        np.zeros((101, 103), np.uint8), np.zeros((99, 97), np.uint8))
    assert all(h % 8 == 0 and w % 8 == 0 for h, w in spy.seen)
    assert out.pts_q == pytest.approx(np.array([[3.0, 4.0]]))


def test_wrapping_is_one_argument_of_the_factory():
    """Смена ядра остаётся одной строкой конфигурации — вместе с рабочим размером."""
    plain = create_matcher("sift")
    wrapped = create_matcher("sift", max_side=640)
    assert not isinstance(plain, ResizedMatcher)
    assert isinstance(wrapped, ResizedMatcher)
    assert wrapped.max_side == 640


def test_wrapped_sift_still_finds_a_known_shift():
    """Сквозная проверка на настоящем матчере: обёртка не ломает поиск."""
    rng = np.random.default_rng(0)
    scene = rng.integers(0, 255, (900, 900), dtype=np.uint8)
    scene = np.repeat(np.repeat(scene[::3, ::3], 3, 0), 3, 1)   # крупная текстура
    query, ref = scene[100:700, 100:700], scene[130:730, 150:750]
    corr = create_matcher("sift", max_side=400).match(query, ref)
    assert len(corr) > 20
    shift = np.median(corr.pts_q - corr.pts_r, axis=0)
    assert shift == pytest.approx(np.array([50.0, 30.0]), abs=6.0)
