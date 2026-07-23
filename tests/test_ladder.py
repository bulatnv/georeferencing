"""Тесты лестницы возмущений внешнего вида (шаг 5 фазы 2, ``docs/TESTING.md``).

Проверяют, что возмущения корректны и детерминированы, что «точка перелома»
разделяет матчеры **измеримо** (это и есть ответ на SIFT vs AKAZE), и что на
однородной местности система честно отказывает, а не выдаёт уверенную ошибку.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from aero_geoloc.geo import haversine_m
from aero_geoloc.matcher import AKAZEMatcher, SIFTMatcher
from aero_geoloc.localize import localize_against_reference
from aero_geoloc.testbench import (
    APPEARANCE_LEVELS,
    SampleSpec,
    apply_appearance,
    default_camera,
    generate_sample,
    make_homogeneous_scene,
    make_synthetic_scene,
)


@pytest.fixture(scope="module")
def scene():
    return make_synthetic_scene(2048, seed=0)


@pytest.fixture(scope="module")
def camera():
    return default_camera(512)


def _success_rate(scene, camera, matcher, level, strength, yaws):
    ok = 0
    for yaw in yaws:
        spec = SampleSpec(yaw_deg=float(yaw), appearance_level=level, appearance_strength=strength)
        s = generate_sample(scene, camera, spec, reference_size=1280)
        r = localize_against_reference(
            s.query, camera, s.prior, s.reference, s.reference_georef, matcher=matcher, refine=True
        )
        ok += r.is_localized
    return ok / len(yaws)


# --- корректность возмущений -------------------------------------------------


def test_apply_appearance_preserves_shape_and_dtype():
    img = make_synthetic_scene(256, seed=1).image
    rng = np.random.default_rng(0)
    for level in APPEARANCE_LEVELS:
        out = apply_appearance(img, level, 1.0, np.random.default_rng(0))
        assert out.shape == img.shape and out.dtype == np.uint8
    # L0 и нулевая сила — тождество.
    assert np.array_equal(apply_appearance(img, 0, 1.0, rng), img)
    assert np.array_equal(apply_appearance(img, 2, 0.0, rng), img)


def test_apply_appearance_is_deterministic_and_changes_image():
    img = make_synthetic_scene(256, seed=2).image
    a = apply_appearance(img, 2, 1.0, np.random.default_rng(7))
    b = apply_appearance(img, 2, 1.0, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)  # одинаковый rng → одинаковый результат
    assert not np.array_equal(a, img)  # возмущение реально что-то меняет


def test_all_appearance_levels_generate_valid_samples(scene, camera):
    for level in APPEARANCE_LEVELS:
        s = generate_sample(scene, camera, SampleSpec(appearance_level=level), reference_size=1280)
        assert s.query.shape == (512, 512)


# --- точка перелома: SIFT vs AKAZE (ответ измерением) ------------------------


def test_akaze_beats_sift_on_blur_noise(scene, camera):
    """L2 (блюр/шум/JPEG): AKAZE заметно устойчивее SIFT — измеренный факт."""
    yaws = np.arange(0.0, 360.0, 45.0)
    sift = _success_rate(scene, camera, SIFTMatcher(), level=2, strength=1.0, yaws=yaws)
    akaze = _success_rate(scene, camera, AKAZEMatcher(), level=2, strength=1.0, yaws=yaws)
    assert akaze > 0.75  # AKAZE держит L2
    assert sift < 0.5  # SIFT уже сыплется
    assert akaze > sift


def test_sift_beats_akaze_on_spectral_shift(scene, camera):
    """L3 (спектральный сдвиг): SIFT не хуже AKAZE — провалы комплементарны."""
    yaws = np.arange(0.0, 360.0, 45.0)
    sift = _success_rate(scene, camera, SIFTMatcher(), level=3, strength=0.5, yaws=yaws)
    akaze = _success_rate(scene, camera, AKAZEMatcher(), level=3, strength=0.5, yaws=yaws)
    assert sift >= akaze
    assert akaze < 0.25  # бинарные дескрипторы разворот градиентов почти не переживают


def test_local_object_changes_still_localize_via_ransac(scene, camera):
    """L4 (подрисованные/убранные объекты): RANSAC отбрасывает изменённые зоны."""
    yaws = np.arange(0.0, 360.0, 45.0)
    rate = _success_rate(scene, camera, SIFTMatcher(), level=4, strength=1.0, yaws=yaws)
    assert rate == 1.0  # локальные изменения не мешают — их гасит робастная оценка


# --- регресс на однородной сцене --------------------------------------------


def test_homogeneous_scene_refuses_not_guesses(camera):
    """Бедная текстурой местность → честный NOT_LOCALIZED, а не уверенная ошибка."""
    scene = make_homogeneous_scene(2048, seed=0)
    for yaw in np.arange(0.0, 360.0, 60.0):
        s = generate_sample(scene, camera, SampleSpec(yaw_deg=float(yaw)), reference_size=1280)
        for matcher in (SIFTMatcher(), AKAZEMatcher()):
            r = localize_against_reference(
                s.query, camera, s.prior, s.reference, s.reference_georef, matcher=matcher, refine=True
            )
            assert not r.is_localized
