"""V-тесты сеточного слоя и замера сдвига (V1 из §6 задания)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rasters import Grid, gradient_map, phase_shift, zoom_for_ground_mpp  # noqa: E402


def _texture(n=256, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.random((n, n)).astype(np.float32)
    import cv2
    return cv2.GaussianBlur(img, (0, 0), 3.0)


# --- V1: знак фазовой корреляции ------------------------------------------------

@pytest.mark.parametrize("dx,dy", [(7, 0), (0, 5), (-6, 4), (11, -9)])
def test_phase_shift_sign_and_magnitude(dx, dy):
    """Синтетический сдвиг np.roll → замер возвращает тот же знак и величину."""
    a = _texture()
    b = np.roll(np.roll(a, dy, axis=0), dx, axis=1)
    sx, sy, peak = phase_shift(a, b)
    assert sx == pytest.approx(dx, abs=0.25)
    assert sy == pytest.approx(dy, abs=0.25)
    assert peak > 0.3


def _fft_shift(img, dx, dy):
    """Точный циклический сдвиг через фазу — честный эталон для субпикселя
    (warpAffine с BORDER_WRAP мажет края интерполяцией и занижает замер)."""
    h, w = img.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    spec = np.fft.fft2(img) * np.exp(-2j * np.pi * (fx * dx + fy * dy))
    return np.real(np.fft.ifft2(spec)).astype(np.float32)


def test_phase_shift_subpixel():
    a = _texture(seed=3)
    b = _fft_shift(a, 2.4, -1.6)
    sx, sy, _ = phase_shift(a, b)
    assert sx == pytest.approx(2.4, abs=0.15) and sy == pytest.approx(-1.6, abs=0.15)


def test_phase_shift_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        phase_shift(np.zeros((8, 8), np.float32), np.zeros((8, 9), np.float32))


# --- градиентная карта ----------------------------------------------------------

def test_gradient_map_zeroes_invalid_and_normalises():
    img = (np.random.default_rng(1).random((64, 64, 3)) * 255).astype(np.uint8)
    valid = np.ones((64, 64), bool)
    valid[:10] = False
    g = gradient_map(img, valid)
    assert g[:10].max() == 0.0
    assert 0.0 <= g.min() and g.max() <= 1.0


# --- сетка ----------------------------------------------------------------------

def test_grid_centres_are_pixel_centred():
    g = Grid(x=100.0, y=200.0, size_px=5, gsd=2.0)
    gx, gy = g.pixel_centres()
    assert gx[0, 0] == pytest.approx(100.0 - 4.0)     # (5-1)/2 * 2
    assert gy[0, 0] == pytest.approx(200.0 + 4.0)
    assert gx[2, 2] == pytest.approx(100.0) and gy[2, 2] == pytest.approx(200.0)
    assert gx[0, 1] - gx[0, 0] == pytest.approx(2.0)
    assert gy[1, 0] - gy[0, 0] == pytest.approx(-2.0)  # Y убывает вниз по строкам


def test_grid_bounds_span_pixel_centres():
    """bounds — габарит по ЦЕНТРАМ угловых пикселей: это и есть диапазон,
    в котором координата попадает внутрь сетки [0, n−1]."""
    g = Grid(x=0.0, y=0.0, size_px=4, gsd=1.0)
    assert g.bounds() == (-1.5, -1.5, 1.5, 1.5)


def test_rotated_grid_bounds_grow_with_rotation():
    """У повёрнутой сетки axis-aligned габарит больше — это габарит, не сетка."""
    g0 = Grid(x=0.0, y=0.0, size_px=100, gsd=1.0, rot_deg=0.0)
    g45 = Grid(x=0.0, y=0.0, size_px=100, gsd=1.0, rot_deg=45.0)
    w0 = g0.bounds()[2] - g0.bounds()[0]
    w45 = g45.bounds()[2] - g45.bounds()[0]
    assert w45 == pytest.approx(w0 * np.sqrt(2), rel=1e-6)


# --- выбор зума ------------------------------------------------------------------

def test_zoom_accounts_for_latitude_stretch():
    """На широте 60° наземный MPP вдвое мельче номинала — зум обязан это учесть."""
    from aero_geoloc.geo import ground_mpp
    z60 = zoom_for_ground_mpp(60.0, 0.30)
    z0 = zoom_for_ground_mpp(0.0, 0.30)
    assert z60 == z0 - 1
    assert ground_mpp(60.0, z60) == pytest.approx(0.30, rel=0.1)


# --- ограничение области поиска -------------------------------------------------

def test_max_shift_limits_search_and_rejects_far_peak():
    """Далёкий ложный пик вне радиуса приора не рассматривается вовсе."""
    a = _texture(seed=11)
    b = np.roll(a, 40, axis=1)               # истинный сдвиг 40 px
    sx_free, _, _ = phase_shift(a, b)
    assert sx_free == pytest.approx(40, abs=0.3)          # без ограничения найден
    sx_lim, sy_lim, _ = phase_shift(a, b, max_shift_px=12)
    assert abs(sx_lim) <= 12.5 and abs(sy_lim) <= 12.5    # вне радиуса не берётся


@pytest.mark.parametrize("dx,dy", [(5, -3), (-7, 2)])
def test_max_shift_keeps_true_peak_inside_radius(dx, dy):
    a = _texture(seed=12)
    b = np.roll(np.roll(a, dy, axis=0), dx, axis=1)
    sx, sy, _ = phase_shift(a, b, max_shift_px=20)
    assert sx == pytest.approx(dx, abs=0.3) and sy == pytest.approx(dy, abs=0.3)


# --- повёрнутая сетка -----------------------------------------------------------

def test_rotated_grid_round_trip():
    """Мир → пиксель → мир возвращает исходную точку при любом повороте."""
    from rasters import Grid
    for rot in (0.0, 25.0, 90.0, 187.0, -40.0):
        g = Grid(x=1000.0, y=-500.0, size_px=64, gsd=0.4, rot_deg=rot)
        gx, gy = g.pixel_centres()
        px, py = g.pixel_from_world(gx, gy)
        j, i = np.meshgrid(np.arange(64.0), np.arange(64.0))
        assert np.allclose(px, j, atol=1e-9) and np.allclose(py, i, atol=1e-9)


def test_rotated_grid_axes_point_to_azimuth():
    """«Верх» сетки смотрит в азимут rot_deg: 90° = восток."""
    from rasters import Grid
    g = Grid(x=0.0, y=0.0, size_px=3, gsd=1.0, rot_deg=90.0)
    right, up = g.axes
    assert up[0] == pytest.approx(1.0) and abs(up[1]) < 1e-12     # вверх → +X (восток)
    assert abs(right[0]) < 1e-12 and right[1] == pytest.approx(-1.0)


def test_rotated_grid_preserves_scale_and_centre():
    from rasters import Grid
    g = Grid(x=10.0, y=20.0, size_px=9, gsd=2.0, rot_deg=33.0)
    gx, gy = g.pixel_centres()
    assert gx[4, 4] == pytest.approx(10.0) and gy[4, 4] == pytest.approx(20.0)
    step = np.hypot(gx[0, 1] - gx[0, 0], gy[0, 1] - gy[0, 0])
    assert step == pytest.approx(2.0)                             # шаг = gsd при любом rot


def test_grid_corners_match_pixel_corners():
    from rasters import Grid
    g = Grid(x=5.0, y=7.0, size_px=16, gsd=0.5, rot_deg=61.0)
    gx, gy = g.pixel_centres()
    c = g.corners_world()
    assert c[0] == pytest.approx([gx[0, 0], gy[0, 0]])
    assert c[1] == pytest.approx([gx[0, -1], gy[0, -1]])
    assert c[2] == pytest.approx([gx[-1, -1], gy[-1, -1]])
    assert c[3] == pytest.approx([gx[-1, 0], gy[-1, 0]])


def test_rotated_grid_bounds_drive_mosaic_size():
    """Габарит повёрнутой сетки — то, по чему заказывается мозаика подложки:
    по стороне её углы вылезали бы за скачанное окно (серые треугольники)."""
    from rasters import Grid
    g = Grid(x=0.0, y=0.0, size_px=1000, gsd=0.5, rot_deg=30.0)
    x0, y0, x1, y1 = g.bounds()
    span = max(x1 - x0, y1 - y0)
    assert span > g.size_px * g.gsd            # габарит больше стороны
    gx, gy = g.pixel_centres()
    assert gx.min() >= x0 - 1e-9 and gx.max() <= x1 + 1e-9
    assert gy.min() >= y0 - 1e-9 and gy.max() <= y1 + 1e-9
