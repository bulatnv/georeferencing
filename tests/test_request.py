"""Сборка входа инструмента: три способа задать одно и то же.

Главное, что здесь проверяется, — не «работает», а **не обманывает**:
происхождение каждого числа доходит до отчёта, GPS снимка не подставляется в
приор молча, а нехватка данных даёт сообщение с готовой командой, а не
трассировку (`docs/TOOL_PLAN.md`, T2).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from aero_geoloc.request import ASSUMED_ALTITUDE_M, InputError, build_request


@pytest.fixture
def image(tmp_path):
    """Снимок без метаданных: 4000×3000, как кадр со срезанным EXIF."""
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), np.full((3000, 4000, 3), 128, np.uint8))
    return path


COORDS = dict(lat=54.81, lon=56.09, sigma_m=1500.0)


# --- три способа задать масштаб ---------------------------------------------

def test_explicit_gsd_defines_the_footprint(image):
    req = build_request(image, gsd_m=0.065, yaw_deg=30.0, **COORDS)
    assert req.gsd_source == "задан явно"
    assert req.footprint_m[0] == pytest.approx(0.065 * 4000, rel=1e-6)


def test_altitude_and_fov_give_the_same_geometry(image):
    """Три источника — одна геометрия, иначе владелец получит разные ответы на одно."""
    by_gsd = build_request(image, gsd_m=0.05, altitude_m=300.0, yaw_deg=0.0, **COORDS)
    focal_px = 300.0 / 0.05
    fov = 2 * np.degrees(np.arctan((4000 / 2) / focal_px))
    by_fov = build_request(image, altitude_m=300.0, fov_deg=float(fov), yaw_deg=0.0, **COORDS)
    assert by_fov.gsd_m == pytest.approx(by_gsd.gsd_m, rel=1e-3)
    assert by_fov.footprint_m[0] == pytest.approx(by_gsd.footprint_m[0], rel=1e-3)


def test_focal_and_sensor_are_accepted(image):
    req = build_request(image, altitude_m=300.0, focal_mm=8.8, sensor_width_mm=13.2,
                        yaw_deg=0.0, **COORDS)
    assert req.gsd_source == "из высоты и параметров камеры"
    assert req.gsd_m == pytest.approx(300.0 * 13.2 / (8.8 * 4000), rel=1e-6)


def test_altitude_without_camera_params_is_an_error(image):
    with pytest.raises(InputError, match="--fov"):
        build_request(image, altitude_m=300.0, yaw_deg=0.0, **COORDS)


# --- честность происхождения ------------------------------------------------

def test_missing_altitude_is_assumed_and_reported(image):
    """Допущение о высоте не влияет на геометрию, но о нём обязаны сказать."""
    req = build_request(image, gsd_m=0.065, yaw_deg=0.0, **COORDS)
    assert req.prior.altitude_m == pytest.approx(ASSUMED_ALTITUDE_M)
    assert any("высота не задана" in n for n in req.notes)


def test_conflicting_scale_inputs_are_reported_not_silently_resolved(image):
    req = build_request(image, gsd_m=0.065, altitude_m=300.0, fov_deg=73.0,
                        yaw_deg=0.0, **COORDS)
    assert req.gsd_source == "задан явно"
    assert any("и GSD, и параметры камеры" in n for n in req.notes)


def test_unknown_heading_warns_about_the_price(image):
    """Цена незнания курса — восьмикратная сборка карты; это надо знать заранее."""
    req = build_request(image, gsd_m=0.065, **COORDS)
    assert req.trust_yaw is False
    assert any("в 8 раз" in n for n in req.notes)


def test_describe_shows_the_footprint_for_eyeballing(image):
    """Неверный GSD даёт не отказ, а уверенно-неверный масштаб.

    Единственная дешёвая защита — показать владельцу, сколько метров покрывает
    кадр, ДО запуска.
    """
    text = build_request(image, gsd_m=0.065, yaw_deg=0.0, **COORDS).describe()
    assert "покрывает 260×195 м" in text
    assert "приор" in text and "±1500 м" in text


# --- ошибки пользователя ----------------------------------------------------

def test_no_scale_source_lists_every_way_to_give_it(image):
    with pytest.raises(InputError) as exc:
        build_request(image, **COORDS)
    text = str(exc.value)
    for hint in ("--gsd", "--altitude", "--focal-mm", "--from-exif"):
        assert hint in text


def test_missing_prior_names_the_flags(image):
    with pytest.raises(InputError, match="--lat"):
        build_request(image, gsd_m=0.065, sigma_m=1000.0, yaw_deg=0.0)


def test_missing_sigma_is_an_error(image):
    with pytest.raises(InputError, match="погрешность"):
        build_request(image, gsd_m=0.065, lat=54.8, lon=56.1, yaw_deg=0.0)


def test_absent_file_is_a_user_error(tmp_path):
    with pytest.raises(InputError, match="нет файла"):
        build_request(tmp_path / "нет.jpg", gsd_m=0.065, yaw_deg=0.0, **COORDS)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_nonpositive_values_are_rejected(image, bad):
    with pytest.raises(InputError):
        build_request(image, gsd_m=bad, yaw_deg=0.0, **COORDS)
    with pytest.raises(InputError):
        build_request(image, gsd_m=0.05, yaw_deg=0.0, lat=54.8, lon=56.1, sigma_m=bad)


def test_from_exif_without_metadata_explains_what_to_do(image):
    """Снимок без EXIF + --from-exif: сообщение обязано предложить альтернативу."""
    with pytest.raises(InputError, match="Задайте --lat/--lon"):
        build_request(image, from_exif=True, sigma_m=1000.0)
