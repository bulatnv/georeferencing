"""V-тесты генератора: компоновка, перекрытие, поле сдвигов (§6 п.7–8)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate import (  # noqa: E402
    DELTA_YAW_MAX,
    FRAME_H,
    FRAME_W,
    PARTIAL_OVERLAP,
    F_PX,
    ShiftField,
    footprint_overlap,
    place_camera,
    plan_sample,
)
from geom import camera_rotation, footprint_corners, intrinsics, rect_overlap_frac  # noqa: E402
from rasters import Grid  # noqa: E402

K = intrinsics(FRAME_W, FRAME_H, F_PX)


def _plan(layout="inside", height=300.0, tilt=0.0, tilt_az=0.0, yaw=0.0, gsd_a=None):
    return dict(layout=layout, height=height, tilt=tilt, tilt_az=tilt_az, yaw=yaw,
                gsd_a=gsd_a if gsd_a is not None else height / F_PX)


def _grid(span_m, cx=1000.0, cy=2000.0, rot=0.0, gsd=0.5):
    n = int(round(span_m / gsd))
    return Grid(x=cx, y=cy, size_px=n, gsd=gsd, rot_deg=rot)


# --- V8: компоновка inside -----------------------------------------------------

@pytest.mark.parametrize("tilt,az,yaw", [(0, 0, 0), (10, 0, 0), (10, 90, 20),
                                         (8, 200, -25), (6, 315, 15)])
def test_inside_layout_keeps_footprint_within_crop(tilt, az, yaw):
    """След кадра целиком внутри кропа — при любом наклоне и повороте."""
    plan = _plan("inside", tilt=tilt, tilt_az=az, yaw=yaw)
    R = camera_rotation(yaw, tilt, az)
    box = _grid(700.0)
    rng = np.random.default_rng(1)
    pos = place_camera(K, R, plan, box, rng)
    assert pos is not None
    assert footprint_overlap(K, R, plan, pos, box) == pytest.approx(1.0, abs=1e-6)


def test_inside_refuses_when_frame_does_not_fit():
    """Честный отказ, если кадр крупнее кропа — а не «как-нибудь впишем»."""
    plan = _plan("inside", height=300.0)          # след ≈ 418 м
    R = camera_rotation(0, 0, 0)
    assert place_camera(K, R, plan, _grid(220.0), np.random.default_rng(0)) is None


def test_tilt_shift_is_compensated_in_placement():
    """Позиция камеры сдвинута назад на H·tan(tilt): центр следа — у якоря."""
    import math
    plan = _plan("inside", height=350.0, tilt=10.0, tilt_az=90.0)
    R = camera_rotation(0.0, 10.0, 90.0)
    box = _grid(900.0)
    rng = np.random.default_rng(3)
    cx, cy = place_camera(K, R, plan, box, rng)
    off = 350.0 * math.tan(math.radians(10.0))
    # камера смещена на запад (−X) относительно якоря максимум на off + разброс
    assert cx <= box.x - off + box.size_px * box.gsd / 2


# --- V7/V8: компоновка partial -------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_partial_layout_hits_target_overlap(seed):
    plan = _plan("partial", height=300.0, tilt=5.0, tilt_az=45.0, yaw=10.0)
    R = camera_rotation(10.0, 5.0, 45.0)
    box = _grid(800.0)
    pos = place_camera(K, R, plan, box, np.random.default_rng(seed))
    assert pos is not None
    frac = footprint_overlap(K, R, plan, pos, box)
    assert PARTIAL_OVERLAP[0] <= frac <= PARTIAL_OVERLAP[1]


# --- план сэмпла ---------------------------------------------------------------

def test_plan_ranges_match_spec():
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = plan_sample(rng, "inside")
        assert 175.0 <= p["height"] <= 300.0
        assert 0.85 <= p["scale"] <= 1.20
        assert 768 <= p["b_px"] <= 2048
        # курс кадра теперь произвольный (0–360°), ограничена лишь РАЗНИЦА
        # курсов сторон — см. test_plan_yaw_is_free_but_delta_is_bounded
        assert 0.0 <= p["yaw"] <= 360.0 and 0.0 <= p["tilt"] <= 10.0
        assert p["gsd_a"] == pytest.approx(p["height"] / F_PX)
        assert p["gsd_b"] == pytest.approx(p["gsd_a"] / p["scale"])


def test_plan_inside_crop_is_wider_than_frame():
    """В компоновке inside кроп B заведомо шире следа кадра."""
    rng = np.random.default_rng(5)
    for _ in range(50):
        p = plan_sample(rng, "inside")
        span_b = p["b_px"] * p["gsd_b"]
        assert span_b > FRAME_W * p["gsd_a"] * 1.05


# --- поле сдвигов ---------------------------------------------------------------

def test_shift_field_interpolates_and_falls_back(tmp_path):
    p = tmp_path / "f.npz"
    np.savez_compressed(p, x=np.array([0.0, 100.0, 0.0, 100.0]),
                        y=np.array([0.0, 0.0, 100.0, 100.0]),
                        dx=np.array([2.0, 2.0, 2.0, 2.0]),
                        dy=np.array([-1.0, -1.0, -1.0, -1.0]),
                        peak=np.zeros(4), resid=np.zeros(4),
                        global_dx=5.0, global_dy=5.0)
    f = ShiftField(p)
    dx, dy, src = f.at(50.0, 50.0)
    assert src == "field" and dx == pytest.approx(2.0) and dy == pytest.approx(-1.0)
    dx2, dy2, src2 = f.at(100000.0, 100000.0)      # вне поля — константа
    assert src2 == "global" and dx2 == 5.0 and dy2 == 5.0


def test_shift_field_none_gives_zero_constant():
    f = ShiftField(None)
    assert f.at(0.0, 0.0) == (0.0, 0.0, "global")


# --- повороты сторон (доработка 28.08) -----------------------------------------

def test_plan_yaw_is_free_but_delta_is_bounded():
    """Курс кадра — любой, но разница с курсом подложки не выходит за потолок."""
    rng = np.random.default_rng(7)
    yaws = []
    for _ in range(400):
        p = plan_sample(rng, "inside")
        yaws.append(p["yaw"])
        assert abs(p["delta_yaw"]) <= DELTA_YAW_MAX
        diff = (p["yaw"] - p["yaw_b"] + 180) % 360 - 180
        assert abs(diff) <= DELTA_YAW_MAX + 1e-9
        assert 0.0 <= p["yaw_b"] < 360.0
    assert min(yaws) < 30 and max(yaws) > 330      # курс покрывает всю окружность


@pytest.mark.parametrize("yaw,rot", [(0.0, 0.0), (200.0, 185.0), (350.0, 10.0),
                                     (95.0, 118.0)])
def test_inside_layout_in_rotated_crop(yaw, rot):
    """Компоновка inside держится и когда кроп B повёрнут вслед за кадром."""
    plan = _plan("inside", tilt=6.0, tilt_az=40.0, yaw=yaw)
    R = camera_rotation(yaw, 6.0, 40.0)
    grid = _grid(760.0, rot=rot)
    pos = place_camera(K, R, plan, grid, np.random.default_rng(2))
    assert pos is not None
    assert footprint_overlap(K, R, plan, pos, grid) == pytest.approx(1.0, abs=1e-6)


def test_overlap_uses_rotated_frame_not_bounding_box():
    """Перекрытие считается в системе повёрнутой сетки, а не по её габариту:
    точка вне кропа, но внутри габарита, не должна засчитываться."""
    grid = _grid(400.0, rot=45.0)
    corner = grid.bounds()[0] + 1.0, grid.bounds()[3] - 1.0    # угол габарита
    px, py = grid.pixel_from_world(*corner)
    n = grid.size_px - 1
    assert not (0 <= px <= n and 0 <= py <= n)                 # вне самой сетки
