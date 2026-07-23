"""Последовательностный режим для низких высот: визуальная одометрия + EKF (фаза 5).

На 100–250 м одиночный кадр из широкого приора — иголка в стоге, и часто плохо
поставлен ([README.md], ограничение 3). Выход ([PLAN.md], фаза 5): накапливать
траекторию **кадр-к-кадру** (visual odometry) и **периодически** привязываться к
карте там, где сцена уникальна, распространяя коррекцию фильтром. Так редкие
уверенные фиксы «якорят» дрейф, а бедные кадры между ними не заставляют угадывать.

Модуль классический (матчер + pose), без тяжёлых зависимостей: VO — это тот же
матчинг двух соседних кадров, а слияние — простой EKF на состоянии
``(east, north, heading)``. Абсолютная привязка приходит извне (колбэком),
поэтому режим не зависит от того, чем именно её получают (окно или retrieval).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .camera import Camera
from .matcher import Matcher, SIFTMatcher
from .pose import estimate_similarity
from .quality import center_covariance

__all__ = [
    "VOStep",
    "estimate_vo",
    "EKFState",
    "TrajectoryEKF",
    "AbsoluteFix",
    "localize_sequence",
]


def _wrap_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class VOStep:
    """Относительное движение между двумя соседними кадрами (в ENU-метрах).

    Attributes:
        delta_east_m, delta_north_m: смещение камеры на земле, метры.
        delta_yaw_deg: изменение курса, градусы.
        scale: отношение масштабов кадров (≈ отношение высот; 1 при постоянной высоте).
        n_inliers: инлайеров в кадр-к-кадру матче.
        position_sigma_m: СКО смещения, метры (шум процесса для EKF).
    """

    delta_east_m: float
    delta_north_m: float
    delta_yaw_deg: float
    scale: float
    n_inliers: int
    position_sigma_m: float


def estimate_vo(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    camera: Camera,
    altitude_m: float,
    prev_heading_deg: float,
    *,
    matcher: Matcher | None = None,
    ransac_threshold_px: float = 3.0,
    min_inliers: int = 10,
) -> VOStep | None:
    """Визуальная одометрия: смещение камеры на земле из матча соседних кадров.

    Матч ``curr → prev`` даёт подобие ``T``. Его поворот — изменение курса
    (``Δyaw = θ_T``), а сдвиг центра кадра ``d = T(c) − c`` (в пикселях)
    переводится в смещение на земле: ``ΔENU = GSD · R(prev_yaw) · d`` (формула
    выведена и проверена на известном движении в ``tests/test_sequence.py``).
    Ось север — это ``−y``, отсюда знак у ``north``.

    Returns:
        :class:`VOStep` либо ``None``, если кадр-к-кадру матч не удался (бедный
        кадр — легитимный исход, наверху он инфлирует ковариацию без движения).
    """
    matcher = matcher if matcher is not None else SIFTMatcher()
    corr = matcher.match(curr_gray, prev_gray)
    if len(corr) < 3:
        return None
    pose = estimate_similarity(corr, ransac_threshold_px=ransac_threshold_px, min_inliers=min_inliers)
    if pose is None:
        return None

    gsd = camera.gsd(altitude_m)
    center = np.array(camera.principal_point())
    shift = pose.transform.apply(center) - center  # сдвиг центра в пикселях кадра

    theta = math.radians(prev_heading_deg)
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    ground = gsd * (rot @ shift)  # метры в ENU-осях (x=east, y=south)

    inlier = pose.inlier_mask
    cov = center_covariance(
        corr.pts_q[inlier], corr.pts_r[inlier], pose.transform, tuple(center), mpp=gsd
    )
    position_sigma_m = math.sqrt(np.trace(cov) / 2.0) if cov is not None else gsd

    return VOStep(
        delta_east_m=float(ground[0]),
        delta_north_m=float(-ground[1]),
        delta_yaw_deg=float(pose.transform.rotation_deg),
        scale=float(pose.transform.scale),
        n_inliers=pose.n_inliers,
        position_sigma_m=float(position_sigma_m),
    )


@dataclass
class EKFState:
    """Оценка позы в траектории: положение ENU (метры), курс (°) и ковариация 3×3."""

    east_m: float
    north_m: float
    heading_deg: float
    covariance: np.ndarray = field(default_factory=lambda: np.eye(3))

    @property
    def position_sigma_m(self) -> float:
        return math.sqrt(0.5 * (self.covariance[0, 0] + self.covariance[1, 1]))


@dataclass(frozen=True)
class AbsoluteFix:
    """Абсолютная привязка к карте: измерение положения/курса с неопределённостью."""

    east_m: float
    north_m: float
    heading_deg: float
    position_sigma_m: float
    heading_sigma_deg: float


class TrajectoryEKF:
    """Простой EKF на состоянии ``(east, north, heading)``.

    Модель движения аддитивна: VO уже даёт приращения в ENU (используя текущий
    курс), поэтому предсказание — сложение, а якобиан единичный. Обновление —
    линейное (измеряем состояние напрямую, ``H = I``), с корректным
    заворачиванием невязки по курсу. Упрощение (нет кросс-корреляции ошибки курса
    и положения) допустимо для завода режима на стенде.
    """

    def __init__(self, state: EKFState) -> None:
        self.state = state

    def predict(self, vo: VOStep, *, heading_process_sigma_deg: float = 0.5) -> None:
        s = self.state
        s.east_m += vo.delta_east_m
        s.north_m += vo.delta_north_m
        s.heading_deg = _wrap_deg(s.heading_deg + vo.delta_yaw_deg)
        process = np.diag(
            [vo.position_sigma_m**2, vo.position_sigma_m**2, heading_process_sigma_deg**2]
        )
        s.covariance = s.covariance + process

    def predict_missing(self, *, position_drift_sigma_m: float, heading_drift_sigma_deg: float = 1.0) -> None:
        """Бедный кадр: движение неизвестно — не двигаемся, но инфлируем ковариацию."""
        self.state.covariance = self.state.covariance + np.diag(
            [position_drift_sigma_m**2, position_drift_sigma_m**2, heading_drift_sigma_deg**2]
        )

    def update(self, fix: AbsoluteFix) -> None:
        s = self.state
        z = np.array([fix.east_m, fix.north_m, fix.heading_deg])
        x = np.array([s.east_m, s.north_m, s.heading_deg])
        innovation = z - x
        innovation[2] = _wrap_deg(innovation[2])
        r_cov = np.diag([fix.position_sigma_m**2, fix.position_sigma_m**2, fix.heading_sigma_deg**2])
        s_cov = s.covariance + r_cov
        gain = s.covariance @ np.linalg.inv(s_cov)
        x_new = x + gain @ innovation
        s.east_m, s.north_m = float(x_new[0]), float(x_new[1])
        s.heading_deg = _wrap_deg(float(x_new[2]))
        s.covariance = (np.eye(3) - gain) @ s.covariance


#: Колбэк абсолютной привязки: (индекс кадра, текущая оценка) → фикс или None.
AbsoluteFixFn = Callable[[int, EKFState], "AbsoluteFix | None"]


def localize_sequence(
    frames_gray: Sequence[np.ndarray],
    camera: Camera,
    init: EKFState,
    *,
    altitude_m: float,
    matcher: Matcher | None = None,
    absolute_fix_fn: AbsoluteFixFn | None = None,
    ransac_threshold_px: float = 3.0,
    min_inliers: int = 10,
) -> list[EKFState]:
    """Прогнать последовательность кадров: VO-дедрекон + периодические фиксы через EKF.

    Args:
        frames_gray: кадры траектории по порядку (grayscale).
        init: начальная оценка (например, из первой уверенной абсолютной привязки).
        altitude_m: высота (для GSD в VO).
        absolute_fix_fn: колбэк, дающий абсолютный фикс на кадре ``i`` (или ``None``,
            если сцена там не уникальна) — так «якоря» ставятся только где надёжно.

    Returns:
        Список :class:`EKFState` — оценка на каждый кадр (сглаженная траектория).
    """
    matcher = matcher if matcher is not None else SIFTMatcher()
    ekf = TrajectoryEKF(
        EKFState(init.east_m, init.north_m, init.heading_deg, np.array(init.covariance, dtype=float))
    )
    gsd = camera.gsd(altitude_m)
    states = [EKFState(ekf.state.east_m, ekf.state.north_m, ekf.state.heading_deg, ekf.state.covariance.copy())]

    for i in range(1, len(frames_gray)):
        vo = estimate_vo(
            frames_gray[i - 1], frames_gray[i], camera, altitude_m, ekf.state.heading_deg,
            matcher=matcher, ransac_threshold_px=ransac_threshold_px, min_inliers=min_inliers,
        )
        if vo is not None:
            ekf.predict(vo)
        else:
            # Бедный кадр: дрейф в один footprint-полукадр как консервативная оценка.
            ekf.predict_missing(position_drift_sigma_m=gsd * camera.image_width * 0.5)

        if absolute_fix_fn is not None:
            fix = absolute_fix_fn(i, ekf.state)
            if fix is not None:
                ekf.update(fix)

        states.append(
            EKFState(ekf.state.east_m, ekf.state.north_m, ekf.state.heading_deg, ekf.state.covariance.copy())
        )
    return states
