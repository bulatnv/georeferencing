"""Робастная оценка позы: подобие 4 DoF из соответствий.

Модель — **подобие** (поворот θ, единый масштаб s, сдвиг), инвариант 1 из
``docs/ARCHITECTURE.md``. Для надира это физически правильная модель, а её
ограниченность сама по себе работает фильтром: гипотезы с абсурдным масштабом
или поворотом отбрасываются раньше, чем дойдут до геореференцирования.

Приоры входят сюда **как ограничения, а не как подсказки** (инвариант 3):
решение, не проходящее по масштабу или повороту, отвергается целиком — это
дешевле, чем потом объяснять уверенно-неверную точку.

Субпиксельный refinement (ECC/flow) — фаза 2, здесь только RANSAC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .matcher import Correspondences

__all__ = ["SimilarityTransform", "PoseEstimate", "estimate_similarity"]


def _wrap_deg(angle: float) -> float:
    """Привести угол к полуинтервалу ``[-180, 180)``: ровно 180° даёт −180°."""
    return (angle + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class SimilarityTransform:
    """Преобразование подобия «кадр → подложка», 4 DoF.

    Матрица 2×3 в форме OpenCV: ``[[a, -b, tx], [b, a, ty]]``, где
    ``a = s·cos θ``, ``b = s·sin θ``.
    """

    matrix: np.ndarray

    def __post_init__(self) -> None:
        m = np.asarray(self.matrix, dtype=float)
        if m.shape != (2, 3):
            raise ValueError(f"матрица должна быть 2×3, получено {m.shape}")
        object.__setattr__(self, "matrix", m)

    @classmethod
    def from_params(
        cls, scale: float, rotation_deg: float, tx: float, ty: float
    ) -> SimilarityTransform:
        """Собрать преобразование из параметров."""
        if scale <= 0.0:
            raise ValueError(f"scale должен быть > 0, получено {scale}")
        theta = math.radians(rotation_deg)
        a = scale * math.cos(theta)
        b = scale * math.sin(theta)
        return cls(np.array([[a, -b, tx], [b, a, ty]], dtype=float))

    @property
    def scale(self) -> float:
        """Масштаб ``s = √(a² + b²)`` (формула из docs/PIPELINE.md, стадия 4)."""
        return float(math.hypot(self.matrix[0, 0], self.matrix[1, 0]))

    @property
    def rotation_deg(self) -> float:
        """Поворот ``θ = atan2(b, a)`` в градусах, в полуинтервале ``[-180, 180)``."""
        return _wrap_deg(math.degrees(math.atan2(self.matrix[1, 0], self.matrix[0, 0])))

    @property
    def translation(self) -> tuple[float, float]:
        """Сдвиг ``(tx, ty)`` в пикселях подложки."""
        return (float(self.matrix[0, 2]), float(self.matrix[1, 2]))

    def apply(self, pts: np.ndarray) -> np.ndarray:
        """Применить к точкам ``(N, 2)`` или к одной точке ``(2,)``."""
        p = np.atleast_2d(np.asarray(pts, dtype=float))
        out = p @ self.matrix[:, :2].T + self.matrix[:, 2]
        return out[0] if np.ndim(pts) == 1 else out

    def inverse(self) -> SimilarityTransform:
        """Обратное преобразование «подложка → кадр»."""
        a, b = self.matrix[0, 0], self.matrix[1, 0]
        det = a * a + b * b
        if det <= 0.0:
            raise ValueError("вырожденное преобразование: нулевой масштаб")
        inv_lin = np.array([[a, b], [-b, a]], dtype=float) / det
        return SimilarityTransform(np.hstack([inv_lin, -inv_lin @ self.matrix[:, 2:3]]))


@dataclass(frozen=True)
class PoseEstimate:
    """Результат робастной оценки вместе с диагностикой для quality/стенда.

    Attributes:
        transform: оценённое подобие «кадр → подложка».
        inlier_mask: ``(N,)`` bool — какие из входных соответствий признаны инлайерами.
        reprojection_rmse_px: RMSE невязок на инлайерах, пиксели подложки.
    """

    transform: SimilarityTransform
    inlier_mask: np.ndarray
    reprojection_rmse_px: float

    @property
    def n_inliers(self) -> int:
        return int(np.count_nonzero(self.inlier_mask))

    @property
    def inlier_ratio(self) -> float:
        n = int(self.inlier_mask.size)
        return self.n_inliers / n if n else 0.0

    def diagnostics(self) -> dict[str, float | int]:
        """Сырьё для ``quality.py`` и разбора провалов на стенде."""
        return {
            "n_correspondences": int(self.inlier_mask.size),
            "n_inliers": self.n_inliers,
            "inlier_ratio": self.inlier_ratio,
            "reprojection_rmse_px": self.reprojection_rmse_px,
            "scale": self.transform.scale,
            "rotation_deg": self.transform.rotation_deg,
        }


def _reprojection_rmse(
    transform: SimilarityTransform, corr: Correspondences, mask: np.ndarray
) -> float:
    """RMSE расстояний «предсказание vs наблюдение» на инлайерах."""
    if not np.any(mask):
        return float("inf")
    predicted = transform.apply(corr.pts_q[mask])
    residuals = np.linalg.norm(predicted - corr.pts_r[mask], axis=1)
    return float(np.sqrt(np.mean(residuals**2)))


def estimate_similarity(
    corr: Correspondences,
    *,
    ransac_threshold_px: float = 3.0,
    min_inliers: int = 10,
    scale_bounds: tuple[float, float] | None = None,
    expected_rotation_deg: float | None = None,
    rotation_tolerance_deg: float = 15.0,
    max_iters: int = 5000,
    confidence: float = 0.999,
) -> PoseEstimate | None:
    """Оценить подобие «кадр → подложка», отбросив выбросы.

    Args:
        corr: кандидатные соответствия от матчера (с мусором).
        ransac_threshold_px: порог инлайера, пиксели подложки.
        min_inliers: минимум инлайеров, ниже которого решение не принимается.
        scale_bounds: допустимый диапазон масштаба ``(lo, hi)``. Выводится из
            приора высоты: ``GSD ∝ H``, см. ``Prior.scale_bounds``.
        expected_rotation_deg: ожидаемый поворот из yaw, если ему доверяем.
        rotation_tolerance_deg: допуск на отклонение от ожидаемого поворота.
        max_iters, confidence: параметры RANSAC.

    Returns:
        :class:`PoseEstimate` либо ``None``, если решения нет или оно не прошло
        ограничения. ``None`` — штатный исход, а не ошибка: наверху он
        превращается в честный отказ.
    """
    if len(corr) < 3:  # подобие определяется двумя парами, RANSAC нужен запас
        return None

    matrix, inliers = cv2.estimateAffinePartial2D(
        corr.pts_q.reshape(-1, 1, 2),
        corr.pts_r.reshape(-1, 1, 2),
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold_px,
        maxIters=max_iters,
        confidence=confidence,
        refineIters=10,
    )
    if matrix is None:
        return None

    mask = (
        np.zeros(len(corr), dtype=bool)
        if inliers is None
        else inliers.ravel().astype(bool)
    )
    if int(np.count_nonzero(mask)) < min_inliers:
        return None

    transform = SimilarityTransform(matrix)

    if scale_bounds is not None:
        lo, hi = scale_bounds
        if not lo <= transform.scale <= hi:
            return None

    if expected_rotation_deg is not None:
        if abs(_wrap_deg(transform.rotation_deg - expected_rotation_deg)) > rotation_tolerance_deg:
            return None

    return PoseEstimate(
        transform=transform,
        inlier_mask=mask,
        reprojection_rmse_px=_reprojection_rmse(transform, corr, mask),
    )
