"""Тесты модели камеры (фаза 0, ``docs/PLAN.md``).

Проверки идут против ручного расчёта и против таблицы высот из
``docs/PIPELINE.md`` (стадия 0) — чтобы код и документация не разъезжались.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aero_geoloc.camera import Camera

# Камера из README (широкоугольная): f_px = 16 · 1024 / 23.5 = 697.02.
WIDE = Camera(image_width=1024, image_height=1024, focal_mm=16.0, sensor_width_mm=23.5)

# Камера, воспроизводящая таблицу высот из PIPELINE.md: f_px = 30 · 1024 / 10.24 = 3000.
PIPELINE_CAM = Camera(image_width=1024, image_height=1024, focal_mm=30.0, sensor_width_mm=10.24)


# --- фокус и интринсики -----------------------------------------------------


def test_focal_px_from_physical_parameters():
    assert WIDE.focal_px() == pytest.approx(16.0 * 1024 / 23.5)
    assert PIPELINE_CAM.focal_px() == pytest.approx(3000.0)


def test_focal_px_from_fov():
    """f_px = (W/2) / tan(FOV/2) — вторая формула из PIPELINE.md."""
    cam = Camera(image_width=1024, image_height=1024, fov_deg=90.0)
    assert cam.focal_px() == pytest.approx(512.0)  # tan(45°) = 1


def test_fov_and_physical_definitions_agree():
    """Камера, собранная по FOV первой камеры, даёт тот же фокус."""
    cam = Camera(image_width=WIDE.image_width, image_height=WIDE.image_height,
                 fov_deg=WIDE.hfov_deg())
    assert cam.focal_px() == pytest.approx(WIDE.focal_px(), rel=1e-12)


def test_physical_parameters_win_over_fov():
    cam = Camera(image_width=1024, image_height=1024, focal_mm=30.0,
                 sensor_width_mm=10.24, fov_deg=90.0)
    assert cam.focal_px() == pytest.approx(3000.0)


def test_hfov_vfov_for_non_square_sensor():
    cam = Camera(image_width=1024, image_height=768, fov_deg=60.0)
    assert cam.hfov_deg() == pytest.approx(60.0, rel=1e-12)
    expected_v = 2.0 * math.degrees(math.atan(384.0 / cam.focal_px()))
    assert cam.vfov_deg() == pytest.approx(expected_v, rel=1e-12)
    assert cam.vfov_deg() < cam.hfov_deg()


def test_principal_point_uses_pixel_center_convention():
    cam = Camera(image_width=1024, image_height=768, fov_deg=60.0)
    assert cam.principal_point() == (511.5, 383.5)


def test_K_structure():
    k = PIPELINE_CAM.K()
    assert k.shape == (3, 3)
    assert k[0, 0] == pytest.approx(3000.0)
    assert k[1, 1] == pytest.approx(3000.0)
    assert k[0, 2] == pytest.approx(511.5)
    assert k[1, 2] == pytest.approx(511.5)
    assert k[0, 1] == 0.0 and k[1, 0] == 0.0
    np.testing.assert_allclose(k[2], [0.0, 0.0, 1.0])


def test_K_inv_is_true_inverse():
    for cam in (WIDE, PIPELINE_CAM, Camera(1024, 768, fov_deg=45.0)):
        np.testing.assert_allclose(cam.K() @ cam.K_inv(), np.eye(3), atol=1e-12)
        np.testing.assert_allclose(cam.K_inv(), np.linalg.inv(cam.K()), atol=1e-12)


# --- GSD и footprint --------------------------------------------------------


def test_gsd_manual_calculation():
    """GSD = H / f_px, посчитано вручную."""
    assert PIPELINE_CAM.gsd(600.0) == pytest.approx(600.0 / 3000.0)
    assert WIDE.gsd(600.0) == pytest.approx(600.0 * 23.5 / (16.0 * 1024))


def test_gsd_is_linear_in_altitude():
    assert PIPELINE_CAM.gsd(1200.0) == pytest.approx(2.0 * PIPELINE_CAM.gsd(600.0))


@pytest.mark.parametrize(
    "altitude_m,footprint_m,gsd_m",
    [
        # Таблица «Стадия 0» из docs/PIPELINE.md, W = 1024, f_px = 3000.
        (600.0, 205.0, 0.20),
        (250.0, 85.0, 0.083),
        (150.0, 51.0, 0.050),
        (100.0, 34.0, 0.033),
    ],
)
def test_reproduces_pipeline_altitude_table(altitude_m, footprint_m, gsd_m):
    assert PIPELINE_CAM.gsd(altitude_m) == pytest.approx(gsd_m, abs=0.001)
    width_m, height_m = PIPELINE_CAM.footprint_m(altitude_m)
    assert width_m == pytest.approx(footprint_m, abs=0.5)
    assert height_m == pytest.approx(footprint_m, abs=0.5)


def test_footprint_is_rectangular_for_non_square_frame():
    cam = Camera(image_width=1024, image_height=768, focal_mm=30.0, sensor_width_mm=10.24)
    width_m, height_m = cam.footprint_m(600.0)
    assert width_m == pytest.approx(0.2 * 1024)
    assert height_m == pytest.approx(0.2 * 768)


def test_altitude_for_gsd_is_inverse_of_gsd():
    for altitude in (100.0, 250.0, 600.0, 800.0):
        assert PIPELINE_CAM.altitude_for_gsd(PIPELINE_CAM.gsd(altitude)) == pytest.approx(
            altitude, rel=1e-12
        )


def test_gsd_rejects_nonpositive_altitude():
    with pytest.raises(ValueError):
        PIPELINE_CAM.gsd(0.0)
    with pytest.raises(ValueError):
        PIPELINE_CAM.gsd(-10.0)
    with pytest.raises(ValueError):
        PIPELINE_CAM.altitude_for_gsd(0.0)


# --- ректификация наклона ---------------------------------------------------


def test_tilt_homography_is_identity_at_zero_angles():
    h = PIPELINE_CAM.tilt_rectification_homography(0.0, 0.0)
    np.testing.assert_allclose(h, np.eye(3), atol=1e-12)


def _apply(h: np.ndarray, x: float, y: float) -> tuple[float, float]:
    p = h @ np.array([x, y, 1.0])
    return (p[0] / p[2], p[1] / p[2])


@pytest.mark.parametrize("pitch_deg", [-10.0, -3.0, 3.0, 10.0])
def test_pure_pitch_shifts_center_along_y(pitch_deg):
    """Центр уезжает в (cx, cy − f·tan(pitch)) — знаковая конвенция из docstring."""
    cam = PIPELINE_CAM
    cx, cy = cam.principal_point()
    x, y = _apply(cam.tilt_rectification_homography(pitch_deg, 0.0), cx, cy)
    assert x == pytest.approx(cx, abs=1e-9)
    assert y == pytest.approx(cy - cam.focal_px() * math.tan(math.radians(pitch_deg)), rel=1e-12)


@pytest.mark.parametrize("roll_deg", [-10.0, -3.0, 3.0, 10.0])
def test_pure_roll_shifts_center_along_x(roll_deg):
    """Центр уезжает в (cx + f·tan(roll), cy)."""
    cam = PIPELINE_CAM
    cx, cy = cam.principal_point()
    x, y = _apply(cam.tilt_rectification_homography(0.0, roll_deg), cx, cy)
    assert y == pytest.approx(cy, abs=1e-9)
    assert x == pytest.approx(cx + cam.focal_px() * math.tan(math.radians(roll_deg)), rel=1e-12)


@pytest.mark.parametrize("angle", [2.0, 7.5, 15.0])
def test_opposite_tilts_cancel(angle):
    """H(−θ)·H(θ) = I для каждого угла по отдельности (повороты вокруг одной оси)."""
    cam = PIPELINE_CAM
    for pitch, roll in ((angle, 0.0), (0.0, angle)):
        h = cam.tilt_rectification_homography(pitch, roll)
        h_back = cam.tilt_rectification_homography(-pitch, -roll)
        composed = h_back @ h
        np.testing.assert_allclose(composed / composed[2, 2], np.eye(3), atol=1e-9)


def test_tilt_homography_is_normalized_and_invertible():
    h = PIPELINE_CAM.tilt_rectification_homography(6.0, -4.0)
    assert h[2, 2] == pytest.approx(1.0)
    assert abs(np.linalg.det(h)) > 1e-6


def test_small_tilt_is_close_to_pure_translation():
    """При малых углах ректификация ≈ сдвиг: 1° на f=3000 даёт ~52 px."""
    cam = PIPELINE_CAM
    cx, cy = cam.principal_point()
    h = cam.tilt_rectification_homography(1.0, 0.0)
    dy_center = _apply(h, cx, cy)[1] - cy
    dy_corner = _apply(h, 0.0, 0.0)[1] - 0.0
    assert dy_center == pytest.approx(-52.36, abs=0.05)
    assert dy_corner == pytest.approx(dy_center, rel=0.2)


# --- валидация --------------------------------------------------------------


def test_camera_requires_focal_definition():
    with pytest.raises(ValueError, match="focal_mm"):
        Camera(image_width=1024, image_height=1024)
    with pytest.raises(ValueError):
        Camera(image_width=1024, image_height=1024, focal_mm=16.0)  # без sensor_width_mm


@pytest.mark.parametrize(
    "kwargs",
    [
        {"image_width": 0, "image_height": 1024, "fov_deg": 60.0},
        {"image_width": 1024, "image_height": -1, "fov_deg": 60.0},
        {"image_width": 1024, "image_height": 1024, "focal_mm": -16.0, "sensor_width_mm": 23.5},
        {"image_width": 1024, "image_height": 1024, "focal_mm": 16.0, "sensor_width_mm": 0.0},
        {"image_width": 1024, "image_height": 1024, "fov_deg": 0.0},
        {"image_width": 1024, "image_height": 1024, "fov_deg": 180.0},
    ],
)
def test_camera_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        Camera(**kwargs)


def test_camera_is_immutable():
    with pytest.raises(Exception):
        WIDE.focal_mm = 20.0  # type: ignore[misc]
