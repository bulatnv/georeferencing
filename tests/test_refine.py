"""Тесты субпиксельного ECC-refinement (стадия 5, ``docs/PIPELINE.md``).

В отличие от ``test_pose.py`` здесь нужны изображения: refinement фотометрический.
Примеры берём из синтетического стенда — у них есть ground truth, поэтому можно
проверить не «стало ли иначе», а «стало ли ближе к истине».

Главная проверка — решение №1 из ``docs/STATUS.md``: ECC сдвигает медиану ошибки
центра с ~0.5 px (потолок локализации ключевых точек) в субпиксель.
"""

from __future__ import annotations

import numpy as np
import pytest

from aero_geoloc.localize import localize_against_reference, normalize_gray
from aero_geoloc.matcher import SIFTMatcher
from aero_geoloc.pose import estimate_similarity, refine_ecc
from aero_geoloc.testbench import SampleSpec, default_camera, generate_sample, make_synthetic_scene


def _sample(yaw_deg=137.0, altitude_ratio=1.1, seed=0):
    scene = make_synthetic_scene(2048, seed=seed)
    camera = default_camera(512)
    return generate_sample(scene, camera, SampleSpec(yaw_deg=yaw_deg, altitude_ratio=altitude_ratio))


def _ransac_pose(sample):
    """RANSAC-модель без refinement — стартовая точка для ECC."""
    q = normalize_gray(sample.query)
    r = normalize_gray(sample.reference)
    corr = SIFTMatcher().match(q, r)
    mpp = sample.reference_georef.mpp
    expected = sample.camera.gsd(sample.prior.altitude_m) / mpp
    lo, hi = sample.prior.scale_bounds
    pose = estimate_similarity(
        corr,
        scale_bounds=(expected * lo, expected * hi),
        expected_rotation_deg=sample.prior.yaw_deg,
    )
    assert pose is not None
    return pose, q, r


def _center_error_px(transform, sample):
    """Ошибка центра кадра в пикселях подложки относительно истины."""
    cx, cy = sample.camera.principal_point()
    est = transform.apply(np.array([cx, cy]))
    true = np.asarray(sample.true_center_ref_px)
    return float(np.linalg.norm(est - true))


# --- ядро refine_ecc --------------------------------------------------------


def test_refine_ecc_reaches_subpixel_from_ransac():
    sample = _sample()
    pose, q, r = _ransac_pose(sample)
    err_before = _center_error_px(pose.transform, sample)

    refined = refine_ecc(q, r, pose.transform)
    assert refined is not None
    err_after = _center_error_px(refined, sample)

    # RANSAC садится на локализацию точек (~0.5 px), ECC уводит в субпиксель.
    assert err_before > 0.2
    assert err_after < 0.1
    assert err_after < err_before


def test_refine_ecc_output_is_pure_similarity():
    """Проекция на подобие: в матрице нет шира — ровно структура [[a,-b],[b,a]]."""
    sample = _sample()
    pose, q, r = _ransac_pose(sample)
    refined = refine_ecc(q, r, pose.transform)
    assert refined is not None
    m = refined.matrix
    assert m[0, 0] == pytest.approx(m[1, 1], abs=1e-12)
    assert m[1, 0] == pytest.approx(-m[0, 1], abs=1e-12)


def test_refine_ecc_rejects_far_seed():
    """Refinement — полировка, а не поиск: сильно смещённый старт отбрасывается."""
    sample = _sample()
    pose, q, r = _ransac_pose(sample)
    tx, ty = pose.transform.translation
    bad = type(pose.transform).from_params(
        pose.transform.scale, pose.transform.rotation_deg, tx + 40.0, ty - 40.0
    )
    assert refine_ecc(q, r, bad) is None


def test_refine_ecc_none_on_flat_image():
    """Без градиентов ECC не сходится — честный None, а не исключение наружу."""
    sample = _sample()
    pose, _, _ = _ransac_pose(sample)
    flat_q = np.full((512, 512), 127, dtype=np.uint8)
    flat_r = np.full((1280, 1280), 127, dtype=np.uint8)
    assert refine_ecc(flat_q, flat_r, pose.transform) is None


# --- интеграция через localize ---------------------------------------------


def test_localize_refine_flag_improves_center_and_sets_diagnostic():
    sample = _sample()
    common = dict(
        camera=sample.camera,
        prior=sample.prior,
        reference=sample.reference,
        reference_georef=sample.reference_georef,
    )
    base = localize_against_reference(sample.query, **common, refine=False)
    refined = localize_against_reference(sample.query, **common, refine=True)

    assert base.is_localized and refined.is_localized
    assert base.diagnostics["refined_ecc"] is False
    assert refined.diagnostics["refined_ecc"] is True

    from aero_geoloc.geo import haversine_m

    err_base = haversine_m(sample.true_lat, sample.true_lon, base.center_lat, base.center_lon)
    err_refined = haversine_m(
        sample.true_lat, sample.true_lon, refined.center_lat, refined.center_lon
    )
    assert err_refined < err_base
    assert err_refined / sample.reference_georef.mpp < 0.1  # субпиксель


def test_localize_default_does_not_refine():
    """По умолчанию refinement выключен — поведение фазы 1 не меняется молча."""
    sample = _sample()
    result = localize_against_reference(
        sample.query,
        sample.camera,
        sample.prior,
        sample.reference,
        sample.reference_georef,
    )
    assert result.diagnostics["refined_ecc"] is False
