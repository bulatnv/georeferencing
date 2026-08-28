"""V-тесты геометрии виртуальной камеры (AGENT_TASK_DATASET_BASEMAP §6).

Стенд раньше кода: каждый тест закрепляет конвенцию, ошибка в которой даёт
молча неверную разметку.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geom import (  # noqa: E402
    camera_rotation,
    footprint_corners,
    fov_deg,
    ground_to_pixels,
    intrinsics,
    nadir_gsd,
    pixels_to_ground,
    rect_overlap_frac,
    tilt_offset_m,
    valid_mask,
)

W, H, F = 1024, 576, 735.0
K = intrinsics(W, H, F)


# --- V2: тождество надира ------------------------------------------------------

def test_nadir_identity_scale_and_orientation():
    """Надир: наземные координаты = пиксельные × GSD, север вверх."""
    height = 300.0
    gsd = nadir_gsd(height, F)
    R = camera_rotation(0.0, 0.0, 0.0)
    cam = (1000.0, 2000.0)
    px = np.array([(W - 1) / 2, (W - 1) / 2 + 100, (W - 1) / 2])
    py = np.array([(H - 1) / 2, (H - 1) / 2, (H - 1) / 2 - 50])
    gx, gy = pixels_to_ground(px, py, K, R, cam, height)
    assert gx[0] == pytest.approx(cam[0]) and gy[0] == pytest.approx(cam[1])
    assert gx[1] - gx[0] == pytest.approx(100 * gsd)      # вправо по кадру = +X
    assert gy[2] - gy[0] == pytest.approx(50 * gsd)       # вверх по кадру = +Y (север)


def test_fov_matches_battle_camera():
    """Камера повторяет тестовый набор: FOV ≈ 69.7°, а не «на глаз»."""
    assert fov_deg(W, F) == pytest.approx(69.7, abs=0.15)


# --- V3: чистый yaw ------------------------------------------------------------

def test_pure_yaw_rotates_footprint_without_scaling():
    height = 300.0
    cam = (500.0, 700.0)
    c0 = footprint_corners(W, H, K, camera_rotation(0.0, 0, 0), cam, height)
    c1 = footprint_corners(W, H, K, camera_rotation(30.0, 0, 0), cam, height)
    r0 = np.linalg.norm(c0 - np.array(cam), axis=1)
    r1 = np.linalg.norm(c1 - np.array(cam), axis=1)
    assert np.allclose(r0, r1)                              # масштаб не изменился
    ang = np.degrees(np.arctan2(*(c1 - np.array(cam)).T[::-1])) - \
        np.degrees(np.arctan2(*(c0 - np.array(cam)).T[::-1]))
    ang = (ang + 180) % 360 - 180
    assert np.allclose(ang, ang[0], atol=1e-6)              # все углы повернулись равно
    assert abs(ang[0]) == pytest.approx(30.0, abs=1e-6)


def test_yaw_sign_is_clockwise():
    """Курс растёт по часовой: при yaw=90° верх кадра смотрит на восток."""
    height, cam = 300.0, (0.0, 0.0)
    gx, gy = pixels_to_ground((W - 1) / 2, (H - 1) / 2 - 100,
                              K, camera_rotation(90.0, 0, 0), cam, height)
    assert gx > 0 and abs(gy) < 1e-9


# --- V4: направление наклона ---------------------------------------------------

@pytest.mark.parametrize("az,exp", [(0.0, (0, 1)), (90.0, (1, 0)),
                                    (180.0, (0, -1)), (270.0, (-1, 0))])
def test_tilt_moves_centre_towards_azimuth(az, exp):
    """Центр кадра уходит по азимуту ровно на H·tan(tilt) — все четыре стороны."""
    height, tilt, cam = 300.0, 8.0, (100.0, 200.0)
    R = camera_rotation(0.0, tilt, az)
    gx, gy = pixels_to_ground((W - 1) / 2, (H - 1) / 2, K, R, cam, height)
    d = tilt_offset_m(height, tilt)
    assert gx - cam[0] == pytest.approx(exp[0] * d, abs=1e-6)
    assert gy - cam[1] == pytest.approx(exp[1] * d, abs=1e-6)


def test_tilt_offset_formula():
    assert tilt_offset_m(300.0, 10.0) == pytest.approx(300 * math.tan(math.radians(10)))


# --- V5: наклон — гомография ---------------------------------------------------

def test_tilted_projection_is_homography():
    """Четыре угла задают всё поле: сверка с гомографией по углам."""
    import cv2
    height, cam = 320.0, (50.0, -70.0)
    R = camera_rotation(17.0, 9.0, 210.0)
    src = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    dst = footprint_corners(W, H, K, R, cam, height).astype(np.float32)
    Hm = cv2.getPerspectiveTransform(src, dst)
    px, py = np.meshgrid(np.linspace(0, W - 1, 7), np.linspace(0, H - 1, 5))
    gx, gy = pixels_to_ground(px, py, K, R, cam, height)
    pts = np.stack([px.ravel(), py.ravel()], axis=-1).astype(np.float32)[None]
    proj = cv2.perspectiveTransform(pts, Hm)[0]
    assert np.allclose(proj[:, 0], gx.ravel(), atol=1e-6)
    assert np.allclose(proj[:, 1], gy.ravel(), atol=1e-6)


# --- V6: круговой тест ---------------------------------------------------------

def test_round_trip_ground_pixel_ground():
    height, cam = 380.0, (12345.0, 67890.0)
    R = camera_rotation(-23.0, 7.5, 42.0)
    px = np.linspace(0, W - 1, 9)
    py = np.linspace(0, H - 1, 9)
    gx, gy = pixels_to_ground(px, py, K, R, cam, height)
    bx, by = ground_to_pixels(gx, gy, K, R, cam, height)
    assert np.allclose(bx, px, atol=1e-6) and np.allclose(by, py, atol=1e-6)


def test_points_behind_camera_are_nan():
    """Луч, уходящий вверх (наклон > 90° от надира), соответствия не имеет."""
    R = camera_rotation(0.0, 89.0, 0.0)
    gx, _ = pixels_to_ground(0.0, 0.0, K, R, (0.0, 0.0), 300.0)
    assert np.isnan(gx)


# --- V7: перекрытие ------------------------------------------------------------

def test_overlap_matches_analytic_rectangles():
    """Надирный след — прямоугольник: доля внутри бокса считается аналитически."""
    height = 300.0
    gsd = nadir_gsd(height, F)
    R = camera_rotation(0.0, 0.0, 0.0)
    poly = footprint_corners(W, H, K, R, (0.0, 0.0), height)
    half_w, half_h = (W - 1) / 2 * gsd, (H - 1) / 2 * gsd
    assert rect_overlap_frac(poly, (-half_w, -half_h, half_w, half_h)) == pytest.approx(1.0)
    # бокс срезает ровно половину следа по X
    assert rect_overlap_frac(poly, (0.0, -half_h, half_w, half_h)) == pytest.approx(0.5, abs=1e-6)
    assert rect_overlap_frac(poly, (10 * half_w, 0, 11 * half_w, half_h)) == 0.0


def test_overlap_of_rotated_footprint_is_between_zero_and_one():
    poly = footprint_corners(W, H, K, camera_rotation(35.0, 6.0, 120.0), (0.0, 0.0), 300.0)
    frac = rect_overlap_frac(poly, (-100.0, -100.0, 100.0, 100.0))
    assert 0.0 < frac < 1.0


# --- маска валидности ----------------------------------------------------------

def test_valid_mask_rejects_black_white_and_pure_fill():
    img = np.zeros((1, 4, 3), np.uint8)
    img[0, 0] = (120, 130, 125)     # нормальный пиксель
    img[0, 1] = (0, 0, 0)           # чёрная дыра
    img[0, 2] = (255, 255, 255)     # белая дыра
    img[0, 3] = (255, 0, 0)         # служебная красная заливка
    assert valid_mask(img)[0].tolist() == [True, False, False, False]
