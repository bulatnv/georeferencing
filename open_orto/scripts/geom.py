"""Геометрия виртуальной камеры БПЛА над плоской землёй (z = 0).

Мир — метрическая сетка ортофотоплана (UTM): ось X вправо, Y вверх, Z вверх.
Камера в точке ``C = (X, Y, H)`` смотрит вниз; её ориентация задаётся
``yaw`` (поворот кадра вокруг вертикали, по часовой от направления «север
вверх») и наклоном ``tilt`` в сторону азимута ``tilt_az``.

Все функции здесь чистые: ни файлов, ни сети — их и покрывают V-тесты
(``AGENT_TASK_DATASET_BASEMAP`` §6).

Конвенции проекта:

- пиксель — центр: ``(0, 0)`` есть центр левого верхнего пикселя, главная
  точка кадра ``((W−1)/2, (H−1)/2)``;
- ``yaw`` растёт по часовой стрелке (как курс), ``tilt_az`` — азимут
  направления наклона в той же системе: 0° = на север (+Y), 90° = на восток.
"""

from __future__ import annotations

import math

import numpy as np


def intrinsics(width: int, height: int, f_px: float) -> np.ndarray:
    """Матрица K для кадра ``width × height`` с фокусом ``f_px`` (пиксель-центр)."""
    return np.array([
        [f_px, 0.0, (width - 1) / 2.0],
        [0.0, f_px, (height - 1) / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def fov_deg(width: int, f_px: float) -> float:
    """Горизонтальный угол обзора в градусах."""
    return 2.0 * math.degrees(math.atan(width / (2.0 * f_px)))


def camera_rotation(yaw_deg: float, tilt_deg: float, tilt_az_deg: float) -> np.ndarray:
    """Матрица R: направление луча в системе камеры → направление в мире.

    Строится как «надир, повёрнутый на yaw вокруг вертикали, затем
    наклонённый на tilt в сторону азимута tilt_az». При нулевых углах ось
    камеры смотрит в −Z (вниз), ось x кадра — на восток (+X), ось y кадра —
    на юг (−Y): это north-up кадр.
    """
    yaw = math.radians(yaw_deg)
    tilt = math.radians(tilt_deg)
    az = math.radians(tilt_az_deg)

    # Базис надирной north-up камеры: столбцы — образы осей (x, y, z) камеры.
    base = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ])
    cy, sy = math.cos(yaw), math.sin(yaw)
    # Поворот вокруг вертикали по часовой стрелке (курс).
    rz = np.array([
        [cy, sy, 0.0],
        [-sy, cy, 0.0],
        [0.0, 0.0, 1.0],
    ])
    # Наклон: ось вращения горизонтальна и перпендикулярна азимуту наклона.
    axis = np.array([math.cos(az), -math.sin(az), 0.0])  # ⟂ направлению азимута
    ct, st = math.cos(tilt), math.sin(tilt)
    kx = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    r_tilt = np.eye(3) + st * kx + (1 - ct) * (kx @ kx)
    return r_tilt @ rz @ base


def pixels_to_ground(px, py, K: np.ndarray, R: np.ndarray, cam_xy, height_m: float):
    """Пиксели кадра → точки земли (плоскость z = 0). Возвращает (gx, gy).

    Луч, уходящий вверх или параллельно земле, даёт NaN — такие пиксели
    смотрят «за горизонт» и соответствия не имеют.
    """
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)
    dx = (px - K[0, 2]) / K[0, 0]
    dy = (py - K[1, 2]) / K[1, 1]
    d_cam = np.stack([dx, dy, np.ones_like(dx)], axis=-1)
    d_world = d_cam @ R.T
    dz = d_world[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(dz < -1e-12, -height_m / dz, np.nan)
    gx = cam_xy[0] + t * d_world[..., 0]
    gy = cam_xy[1] + t * d_world[..., 1]
    return gx, gy


def ground_to_pixels(gx, gy, K: np.ndarray, R: np.ndarray, cam_xy, height_m: float):
    """Точки земли → пиксели кадра (обратно к :func:`pixels_to_ground`).

    Точки позади камеры (луч уходит вверх) дают NaN.
    """
    gx = np.asarray(gx, dtype=np.float64)
    gy = np.asarray(gy, dtype=np.float64)
    vec = np.stack([gx - cam_xy[0], gy - cam_xy[1],
                    np.full(np.shape(gx), -height_m, dtype=np.float64)], axis=-1)
    d_cam = vec @ R                      # R ортогональна: R⁻¹ = Rᵀ, vec·R = Rᵀ·vec
    z = d_cam[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        px = np.where(z > 1e-12, K[0, 0] * d_cam[..., 0] / z + K[0, 2], np.nan)
        py = np.where(z > 1e-12, K[1, 1] * d_cam[..., 1] / z + K[1, 2], np.nan)
    return px, py


def footprint_corners(width: int, height: int, K, R, cam_xy, height_m: float):
    """Четыре угла следа кадра на земле в порядке (0,0), (W−1,0), (W−1,H−1), (0,H−1)."""
    px = np.array([0.0, width - 1.0, width - 1.0, 0.0])
    py = np.array([0.0, 0.0, height - 1.0, height - 1.0])
    gx, gy = pixels_to_ground(px, py, K, R, cam_xy, height_m)
    return np.stack([gx, gy], axis=-1)


def tilt_offset_m(height_m: float, tilt_deg: float) -> float:
    """Смещение центра следа от надирной точки при наклоне: ``H·tan(tilt)``."""
    return height_m * math.tan(math.radians(tilt_deg))


def nadir_gsd(height_m: float, f_px: float) -> float:
    """Наземный размер пикселя в центре надирного кадра."""
    return height_m / f_px


def rect_overlap_frac(poly_xy: np.ndarray, box) -> float:
    """Доля площади выпуклого четырёхугольника ``poly_xy``, попавшая в
    прямоугольник ``box = (x_min, y_min, x_max, y_max)`` (клип Сазерленда—Ходжмана).
    """
    x_min, y_min, x_max, y_max = box
    poly = [tuple(p) for p in np.asarray(poly_xy, dtype=np.float64)]

    def clip(subject, inside, intersect):
        out = []
        if not subject:
            return out
        prev = subject[-1]
        for cur in subject:
            if inside(cur):
                if not inside(prev):
                    out.append(intersect(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(intersect(prev, cur))
            prev = cur
        return out

    def lerp(a, b, t):
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

    edges = [
        (lambda p: p[0] >= x_min, lambda a, b: lerp(a, b, (x_min - a[0]) / (b[0] - a[0]))),
        (lambda p: p[0] <= x_max, lambda a, b: lerp(a, b, (x_max - a[0]) / (b[0] - a[0]))),
        (lambda p: p[1] >= y_min, lambda a, b: lerp(a, b, (y_min - a[1]) / (b[1] - a[1]))),
        (lambda p: p[1] <= y_max, lambda a, b: lerp(a, b, (y_max - a[1]) / (b[1] - a[1]))),
    ]
    for inside, inter in edges:
        poly = clip(poly, inside, inter)
    if len(poly) < 3:
        return 0.0
    return abs(_shoelace(poly)) / max(abs(_shoelace([tuple(p) for p in poly_xy])), 1e-12)


def _shoelace(poly) -> float:
    s = 0.0
    for i, (x1, y1) in enumerate(poly):
        x2, y2 = poly[(i + 1) % len(poly)]
        s += x1 * y2 - x2 * y1
    return s / 2.0


def valid_mask(rgb: np.ndarray) -> np.ndarray:
    """Маска валидных пикселей растра (§2.1 задания).

    Отсекает три вида «пустоты» разом: чёрную (максимум канала мал), белую
    (минимум велик) и **служебные заливки чистым цветом** — красные полосы по
    краям блоков съёмки, которые проходят и проверку на чёрное, и на белое,
    но имеют огромный размах между каналами.
    """
    a = rgb.astype(np.int16)
    hi = a.max(axis=-1)
    lo = a.min(axis=-1)
    return (hi > 5) & (lo < 250) & ((hi - lo) < 200)
