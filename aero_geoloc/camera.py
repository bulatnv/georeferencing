"""Модель камеры: интринсики, GSD, footprint, ректификация наклона.

Модуль ничего не знает ни про матчинг, ни про координаты — только про
внутреннюю геометрию камеры и её проекцию на плоскую землю (стадия 0
пайплайна, ``docs/PIPELINE.md``).

Соглашение о пиксельных координатах — «центр пикселя», как в
:mod:`aero_geoloc.geo` и OpenCV: главная точка идеальной камеры находится
в ``((W-1)/2, (H-1)/2)``.

Система координат камеры: **x вправо, y вниз, z вдоль оптической оси**
(при надире z смотрит в землю). Это стандартная конвенция OpenCV.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["Camera"]


@dataclass(frozen=True)
class Camera:
    """Параметры камеры кадра.

    Фокус задаётся одним из двух способов:
    ``focal_mm`` + ``sensor_width_mm``, либо ``fov_deg`` (горизонтальный угол
    поля зрения). Пиксели считаются квадратными, дисторсия не моделируется.

    Attributes:
        image_width, image_height: размер кадра в пикселях.
        focal_mm: фокусное расстояние объектива, мм.
        sensor_width_mm: физическая ширина сенсора, мм.
        fov_deg: горизонтальный FOV в градусах — альтернатива focal+sensor.
    """

    image_width: int
    image_height: int
    focal_mm: float | None = None
    sensor_width_mm: float | None = None
    fov_deg: float | None = None

    @classmethod
    def from_gsd(
        cls, image_width: int, image_height: int, gsd_m: float, altitude_m: float
    ) -> Camera:
        """Камера по **известному GSD** — для снимков без метаданных объектива.

        Пайплайну нужен не фокус сам по себе, а GSD: им выбирается зум подложки и
        задаётся ожидаемый масштаб. Когда EXIF срезан (см. ``docs/EVAL_PLAN.md``),
        фокуса нет, но GSD часто известен из паспорта съёмки — тогда камера
        восстанавливается отсюда: ``f_px = altitude_m / gsd_m``.

        ``altitude_m`` здесь — **опорная** высота, а не измеренная: геометрию
        задаёт только отношение ``altitude_m / gsd_m``. Пара подбирается так,
        чтобы ``camera.gsd(altitude_m) == gsd_m`` (это и проверяется тестом), а
        высота дальше живёт в :class:`~aero_geoloc.types.Prior` и уточняется
        восстановленным масштабом.

        Args:
            image_width, image_height: размер кадра в пикселях.
            gsd_m: разрешение кадра на земле, м/пиксель.
            altitude_m: опорная высота, м (та, при которой GSD равен заданному).
        """
        if gsd_m <= 0.0:
            raise ValueError(f"gsd_m должен быть > 0, получено {gsd_m}")
        if altitude_m <= 0.0:
            raise ValueError(f"altitude_m должен быть > 0, получено {altitude_m}")
        focal_px = altitude_m / gsd_m
        # Задаём именно FOV, а не focal+sensor: физическая оптика неизвестна, а
        # угол поля зрения — то, что сохраняется при ресемпле кадра
        # (``drone.frame_at_mpp`` пересобирает камеру как ``Camera(w, h, fov_deg=...)``).
        fov_deg = 2.0 * math.degrees(math.atan((image_width / 2.0) / focal_px))
        return cls(image_width=image_width, image_height=image_height, fov_deg=fov_deg)

    def __post_init__(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError(
                f"размер кадра должен быть > 0, получено {self.image_width}×{self.image_height}"
            )

        has_physical = self.focal_mm is not None and self.sensor_width_mm is not None
        has_fov = self.fov_deg is not None
        if not has_physical and not has_fov:
            raise ValueError(
                "нужно задать либо focal_mm вместе с sensor_width_mm, либо fov_deg"
            )
        if has_physical and (self.focal_mm <= 0 or self.sensor_width_mm <= 0):
            raise ValueError("focal_mm и sensor_width_mm должны быть > 0")
        if has_fov and not 0.0 < self.fov_deg < 180.0:
            raise ValueError(f"fov_deg должен лежать в (0, 180), получено {self.fov_deg}")

    # --- интринсики ---------------------------------------------------------

    def focal_px(self) -> float:
        """Фокусное расстояние в пикселях.

        ``f_px = focal_mm · W / sensor_mm``  либо  ``f_px = (W/2) / tan(FOV/2)``.
        Если заданы оба способа, приоритет у физических параметров.
        """
        if self.focal_mm is not None and self.sensor_width_mm is not None:
            return self.focal_mm * self.image_width / self.sensor_width_mm
        return (self.image_width / 2.0) / math.tan(math.radians(self.fov_deg) / 2.0)

    def principal_point(self) -> tuple[float, float]:
        """Главная точка (центр кадра) в соглашении «центр пикселя»."""
        return ((self.image_width - 1) / 2.0, (self.image_height - 1) / 2.0)

    def K(self) -> np.ndarray:
        """Матрица интринсиков 3×3 (квадратные пиксели, нулевой скос)."""
        f = self.focal_px()
        cx, cy = self.principal_point()
        return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=float)

    def K_inv(self) -> np.ndarray:
        """Обратная матрица интринсиков (аналитически, без ``np.linalg.inv``)."""
        f = self.focal_px()
        cx, cy = self.principal_point()
        return np.array(
            [[1.0 / f, 0.0, -cx / f], [0.0, 1.0 / f, -cy / f], [0.0, 0.0, 1.0]], dtype=float
        )

    def hfov_deg(self) -> float:
        """Горизонтальный угол поля зрения, градусы."""
        return 2.0 * math.degrees(math.atan((self.image_width / 2.0) / self.focal_px()))

    def vfov_deg(self) -> float:
        """Вертикальный угол поля зрения, градусы."""
        return 2.0 * math.degrees(math.atan((self.image_height / 2.0) / self.focal_px()))

    # --- проекция на землю (надир, плоская земля) ---------------------------

    def gsd(self, altitude_m: float) -> float:
        """Разрешение кадра на земле [м/пиксель] при съёмке в надир.

        ``GSD = H / f_px``. Допущение плоской земли и строгого надира;
        деградирует с падением высоты (см. ограничение 5 в README).
        """
        if altitude_m <= 0.0:
            raise ValueError(f"altitude_m должен быть > 0, получено {altitude_m}")
        return altitude_m / self.focal_px()

    def footprint_m(self, altitude_m: float) -> tuple[float, float]:
        """Размер отпечатка кадра на земле [м] как ``(ширина, высота)``."""
        gsd = self.gsd(altitude_m)
        return (gsd * self.image_width, gsd * self.image_height)

    def altitude_for_gsd(self, gsd_m: float) -> float:
        """Высота [м], дающая заданный GSD. Обратная к :meth:`gsd`.

        Используется в цикле переуточнения масштаба: восстановленный масштаб
        даёт GSD, GSD даёт оценку высоты.
        """
        if gsd_m <= 0.0:
            raise ValueError(f"gsd_m должен быть > 0, получено {gsd_m}")
        return gsd_m * self.focal_px()

    # --- ректификация наклона -----------------------------------------------

    def tilt_rectification_homography(
        self, pitch_deg: float, roll_deg: float
    ) -> np.ndarray:
        """Гомография, приводящая наклонный кадр к надир-эквиваленту.

        ``H = K · R · K⁻¹``, где ``R = Rx(pitch) · Ry(roll)`` — поворот,
        переводящий направления лучей из системы снятого кадра в систему
        надирной камеры. Полученная матрица отображает пиксели ИСХОДНОГО
        кадра в пиксели ректифицированного (``cv2.warpPerspective`` с этой
        матрицей и без флага ``WARP_INVERSE_MAP``).

        Знаковая конвенция (проверяется тестами): при ``roll = 0`` центр
        кадра уезжает в ``(cx, cy − f·tan(pitch))``, при ``pitch = 0`` — в
        ``(cx + f·tan(roll), cy)``. То есть положительный pitch поднимает
        содержимое вверх по кадру, положительный roll сдвигает вправо.

        Допущение плоской земли: приближение хорошее при
        ``|pitch|, |roll| ≲ 10°`` (см. ``docs/PIPELINE.md``, стадия 1).
        Высокие объекты этим не исправляются — нужен DEM/DSM.

        Note:
            Выход не перецентрируется и не масштабируется под холст: при
            заметных углах часть содержимого уедет за границы. Подбор
            выходного размера — задача вызывающего кода (фаза 2).
        """
        p = math.radians(pitch_deg)
        r = math.radians(roll_deg)

        rx = np.array(
            [[1.0, 0.0, 0.0], [0.0, math.cos(p), -math.sin(p)], [0.0, math.sin(p), math.cos(p)]]
        )
        ry = np.array(
            [[math.cos(r), 0.0, math.sin(r)], [0.0, 1.0, 0.0], [-math.sin(r), 0.0, math.cos(r)]]
        )

        h = self.K() @ (rx @ ry) @ self.K_inv()
        return h / h[2, 2]
