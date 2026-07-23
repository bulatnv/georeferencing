"""Общие типы данных сервиса: запрос, приор, результат, статус.

Форматы зафиксированы в ``docs/ARCHITECTURE.md`` («Форматы данных»). Модуль
намеренно не зависит ни от чего, кроме :mod:`aero_geoloc.camera`, — на него
опираются все остальные слои.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .camera import Camera

__all__ = ["Camera", "Prior", "LocalizationRequest", "Status", "LocalizationResult"]


@dataclass(frozen=True)
class Prior:
    """Грубое начальное приближение, приходящее с борта.

    Приор позиции — **ограничение**, а не подсказка (инвариант 3 из
    ``docs/ARCHITECTURE.md``): решение вне диска ±σ отбрасывается.
    Приор высоты задаёт допуск на восстановленный масштаб.

    Attributes:
        lat, lon: предполагаемый центр кадра, градусы.
        sigma_m: σ ошибки позиции, метры (штатно до 5000).
        altitude_m: приблизительная высота над землёй, метры (штатно 400–800).
        altitude_sigma_m: σ ошибки высоты, метры → приор на масштаб.
        yaw_deg: курс камеры относительно севера, градусы.
        pitch_deg, roll_deg: углы наклона, градусы (для надира ≈ 0).
    """

    lat: float
    lon: float
    sigma_m: float
    altitude_m: float
    altitude_sigma_m: float
    yaw_deg: float
    pitch_deg: float = 0.0
    roll_deg: float = 0.0

    def __post_init__(self) -> None:
        if not -90.0 <= self.lat <= 90.0:
            raise ValueError(f"lat вне [-90, 90]: {self.lat}")
        if not -180.0 <= self.lon <= 180.0:
            raise ValueError(f"lon вне [-180, 180]: {self.lon}")
        if self.sigma_m <= 0.0:
            raise ValueError(f"sigma_m должен быть > 0, получено {self.sigma_m}")
        if self.altitude_m <= 0.0:
            raise ValueError(f"altitude_m должен быть > 0, получено {self.altitude_m}")
        if self.altitude_sigma_m < 0.0:
            raise ValueError(
                f"altitude_sigma_m должен быть >= 0, получено {self.altitude_sigma_m}"
            )

    @property
    def scale_bounds(self) -> tuple[float, float]:
        """Допустимый диапазон масштаба ``s = H_true / H_prior`` в пределах ±3σ высоты.

        Используется как отбраковка гипотез в ``pose.py``: ``GSD ∝ H``, поэтому
        неопределённость высоты напрямую переводится в неопределённость масштаба.
        """
        delta = 3.0 * self.altitude_sigma_m
        lo = max(self.altitude_m - delta, 1e-6) / self.altitude_m
        hi = (self.altitude_m + delta) / self.altitude_m
        return (lo, hi)


@dataclass
class LocalizationRequest:
    """Полный вход одной локализации."""

    image: np.ndarray | str
    camera: Camera
    prior: Prior
    basemap: str = "esri"


class Status(str, Enum):
    """Исход локализации. Отказ — легитимный результат, а не ошибка."""

    LOCALIZED = "localized"
    LOW_CONFIDENCE = "low_confidence"
    NOT_LOCALIZED = "not_localized"


@dataclass
class LocalizationResult:
    """Результат локализации: распределение, а не точка (инвариант 4).

    При статусе :attr:`Status.NOT_LOCALIZED` все геометрические поля равны
    ``None`` — потребитель обязан проверять статус до чтения координат.

    Attributes:
        status: исход.
        center_lat, center_lon: уточнённый центр кадра, градусы.
        heading_deg: курс кадра относительно севера, градусы.
        altitude_est_m: оценка высоты из восстановленного масштаба.
        footprint_lonlat: 4 угла отпечатка кадра как ``(lon, lat)``.
        transform: 2×3 матрица подобия «кадр → подложка».
        error_ellipse_m: ``(semi_major, semi_minor, angle_deg)``, 1σ.
        covariance_m2: 2×2 ковариация центра в метрах².
        confidence: откалиброванная уверенность 0..1.
        diagnostics: сырьё для стенда и разбора провалов — не выбрасывается.
    """

    status: Status
    center_lat: float | None = None
    center_lon: float | None = None
    heading_deg: float | None = None
    altitude_est_m: float | None = None
    footprint_lonlat: list[tuple[float, float]] | None = None
    transform: np.ndarray | None = None
    error_ellipse_m: tuple[float, float, float] | None = None
    covariance_m2: np.ndarray | None = None
    confidence: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def is_localized(self) -> bool:
        """Есть ли пригодные к использованию координаты (LOCALIZED или LOW_CONFIDENCE)."""
        return self.status is not Status.NOT_LOCALIZED

    @classmethod
    def failed(cls, reason: str, **diagnostics: Any) -> LocalizationResult:
        """Собрать отказ с указанием причины — единая точка для всех веток провала."""
        return cls(
            status=Status.NOT_LOCALIZED,
            confidence=0.0,
            diagnostics={"reason": reason, **diagnostics},
        )
