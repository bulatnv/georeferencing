"""Робастная оценка позы: подобие 4 DoF из соответствий.

Модель — **подобие** (поворот θ, единый масштаб s, сдвиг), инвариант 1 из
``docs/ARCHITECTURE.md``. Для надира это физически правильная модель, а её
ограниченность сама по себе работает фильтром: гипотезы с абсурдным масштабом
или поворотом отбрасываются раньше, чем дойдут до геореференцирования.

Приоры входят сюда **как ограничения, а не как подсказки** (инвариант 3):
решение, не проходящее по масштабу или повороту, отвергается целиком — это
дешевле, чем потом объяснять уверенно-неверную точку.

Субпиксельный refinement (:func:`refine_ecc`) — стадия 5 из ``docs/PIPELINE.md``,
главный рычаг точности фазы 2: RANSAC садится на локализацию ключевых точек
(~0.5 px, потолок из решения №1 в ``docs/STATUS.md``), а ECC дотягивает
преобразование **фотометрически по всем пикселям** до субпикселя.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .matcher import Correspondences

__all__ = [
    "SimilarityTransform",
    "PoseEstimate",
    "estimate_similarity",
    "refine_ecc",
    "fit_similarity",
    "inlier_spread",
    "bootstrap_center_scatter_px",
    "UNIFORM_SPREAD",
]


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

    def with_transform(
        self, transform: SimilarityTransform, corr: Correspondences
    ) -> PoseEstimate:
        """Та же оценка с заменённым преобразованием (после refinement).

        Инлайеры остаются прежними — их отбирал RANSAC по геометрии, — а RMSE
        пересчитывается против нового преобразования: после ECC невязки на тех
        же инлайерах меняются, и честнее показать именно их.
        """
        return PoseEstimate(
            transform=transform,
            inlier_mask=self.inlier_mask,
            reprojection_rmse_px=_reprojection_rmse(transform, corr, self.inlier_mask),
        )


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


def _as_ecc_input(image: np.ndarray) -> np.ndarray:
    """Привести изображение к одноканальному float32 — формату, удобному ECC."""
    if image.ndim != 2:
        raise ValueError(f"ECC ждёт grayscale, получено {image.ndim}D {image.shape}")
    return image.astype(np.float32)


def refine_ecc(
    query_gray: np.ndarray,
    ref_gray: np.ndarray,
    transform: SimilarityTransform,
    *,
    max_iters: int = 100,
    eps: float = 1e-6,
    gauss_filt_size: int = 5,
    max_center_shift_px: float = 8.0,
    max_scale_ratio: float = 1.05,
    max_rotation_deg: float = 2.0,
) -> SimilarityTransform | None:
    """Субпиксельный фотометрический refinement подобия через ECC (стадия 5).

    Максимизирует ``cv2.findTransformECC`` корреляцию яркостей кадра и подложки,
    засеявшись RANSAC-моделью. Уточнение идёт в режиме ``MOTION_AFFINE`` (6 DoF),
    после чего результат **проецируется на ближайшее подобие** — так мы получаем
    фотометрическую точность, но не выходим за 4 DoF (инвариант 1): для надира
    шир и разномасштабность по осям физически невозможны, и удерживать их в
    модели нельзя.

    Направление warp совпадает с нашим ``transform`` (кадр → подложка): для ECC
    ``query`` — это template, ``ref`` — input, поэтому найденный warp отображает
    пиксели кадра в пиксели подложки, как и ``transform``.

    Refinement — это **полировка, а не новый поиск**: результат принимается
    только если он близок к исходной модели (сдвиг центра, масштаб, поворот в
    заданных допусках). Большой скачок означает, что ECC ушёл вразнос на
    неудачной текстуре, и такой результат честнее отбросить, вернув ``None`` —
    наверху останется RANSAC-модель.

    Args:
        query_gray, ref_gray: кадр и окно подложки, grayscale.
        transform: стартовое преобразование от :func:`estimate_similarity`.
        max_iters, eps: критерий остановки ECC.
        gauss_filt_size: сглаживание градиентов в ECC (нечётное).
        max_center_shift_px: макс. допустимый сдвиг центра кадра при refinement.
        max_scale_ratio: масштаб после refinement должен лежать в
            ``[1/r, r]`` относительно исходного.
        max_rotation_deg: макс. допустимое изменение поворота, градусы.

    Returns:
        Уточнённое :class:`SimilarityTransform` либо ``None``, если ECC не сошёлся
        или результат не прошёл проверку близости к исходной модели.
    """
    q = _as_ecc_input(query_gray)
    r = _as_ecc_input(ref_gray)
    warp = transform.matrix.astype(np.float32).copy()
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, int(max_iters), float(eps))
    try:
        cv2.findTransformECC(q, r, warp, cv2.MOTION_AFFINE, criteria, None, gauss_filt_size)
    except cv2.error:
        return None
    if not np.all(np.isfinite(warp)):
        return None

    # Проекция аффинной матрицы на ближайшее (в норме Фробениуса) подобие:
    # a = (A00+A11)/2, b = (A10−A01)/2 — это точное решение МНК на подпространстве
    # матриц вида [[a,−b],[b,a]]. Перенос берём как есть.
    a = float((warp[0, 0] + warp[1, 1]) / 2.0)
    b = float((warp[1, 0] - warp[0, 1]) / 2.0)
    refined = SimilarityTransform(np.array([[a, -b, warp[0, 2]], [b, a, warp[1, 2]]], dtype=float))
    if refined.scale <= 0.0:
        return None

    # Сан-гейт близости к исходной модели.
    if not (1.0 / max_scale_ratio <= refined.scale / transform.scale <= max_scale_ratio):
        return None
    if abs(_wrap_deg(refined.rotation_deg - transform.rotation_deg)) > max_rotation_deg:
        return None
    h, w = query_gray.shape[:2]
    center = np.array([(w - 1) / 2.0, (h - 1) / 2.0])
    if float(np.linalg.norm(refined.apply(center) - transform.apply(center))) > max_center_shift_px:
        return None

    return refined


# --- геометрические сигналы согласия (E2 из docs/ROADMAP.md) -----------------
#
# Зачем они. Существующая связка качества опирается на фотометрию (NCC) и на
# счётчик инлайеров. Первое измеренно рушится на смене сезона (0.53-0.69 летом
# против 0.04-0.11 весной при ВЕРНОЙ позе), второе — просто число, ничего не
# говорящее о том, КАК инлайеры лежат. Сигналы ниже смотрят только на геометрию
# соответствий и потому от сезона не зависят в принципе: тридцать точек, сбитых
# в один угол, подозрительны и летом, и весной.

#: Ожидаемый ``inlier_spread`` при РАВНОМЕРНОМ разбросе точек по квадрату.
#:
#: Среднее попарное расстояние для равномерных точек в квадрате со стороной L
#: равно ≈0.5214·L, диагональ равна √2·L, отсюда 0.5214/√2. Константа нужна как
#: точка отсчёта: 0.37 — «разбросаны как попало по кадру», близко к нулю — «сбиты
#: в кучу».
UNIFORM_SPREAD = 0.3687


def fit_similarity(pts_q: np.ndarray, pts_r: np.ndarray) -> SimilarityTransform | None:
    """Подобие по МНК на ВСЕХ поданных парах, без робастности.

    Отличие от :func:`estimate_similarity` принципиальное: там RANSAC отбирает
    инлайеры, здесь модель считается по тому, что дали. Нужно для бутстрэпа, где
    выборка уже состоит из инлайеров и повторный отбор исказил бы разброс.

    Решение замкнутое: в комплексной записи подобие — это умножение на число,
    поэтому МНК даёт ``c = Σ conj(z)·w / Σ |z|²``.
    """
    q = np.asarray(pts_q, dtype=float)
    r = np.asarray(pts_r, dtype=float)
    if q.shape != r.shape or q.ndim != 2 or q.shape[0] < 2:
        return None
    mq, mr = q.mean(axis=0), r.mean(axis=0)
    z = (q[:, 0] - mq[0]) + 1j * (q[:, 1] - mq[1])
    w = (r[:, 0] - mr[0]) + 1j * (r[:, 1] - mr[1])
    denom = float(np.sum(np.abs(z) ** 2))
    if denom <= 1e-12:
        return None                       # все точки в одной позиции — модели нет
    c = complex(np.sum(np.conj(z) * w) / denom)
    a, b = float(c.real), float(c.imag)
    if math.hypot(a, b) <= 0.0:
        return None
    tx = mr[0] - (a * mq[0] - b * mq[1])
    ty = mr[1] - (b * mq[0] + a * mq[1])
    return SimilarityTransform(np.array([[a, -b, tx], [b, a, ty]], dtype=float))


def inlier_spread(pts: np.ndarray, diagonal_px: float) -> float:
    """Насколько широко инлайеры разбросаны по кадру, в долях его диагонали.

    Среднее попарное расстояние, делённое на диагональ кадра. Мера **не зависит
    от числа точек** — этим она отличается от «покрытия сетки», которое при
    восьми инлайерах не может превысить 8/16 и потому мерит счётчик, а не
    геометрию.

    Ориентир — :data:`UNIFORM_SPREAD` ≈ 0.37 (равномерный разброс по квадрату).
    Значение около нуля означает, что вся поза держится на одном пятачке: такая
    выборка задаёт масштаб и поворот крайне плохо, даже если инлайеров много.
    """
    p = np.asarray(pts, dtype=float)
    if p.ndim != 2 or p.shape[0] < 2 or diagonal_px <= 0.0:
        return 0.0
    diff = p[:, None, :] - p[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    n = p.shape[0]
    mean_pairwise = float(distances.sum() / (n * (n - 1)))
    return mean_pairwise / float(diagonal_px)


def bootstrap_center_scatter_px(
    corr: Correspondences, inlier_mask: np.ndarray, centre_px: tuple[float, float],
    *, draws: int = 32, seed: int = 0
) -> float:
    """Насколько «шатается» центр кадра, если пересобрать позу по подвыборке инлайеров.

    Инлайеры берутся с возвращением ``draws`` раз, по каждой выборке считается
    подобие, и смотрится, куда уезжает центр кадра. Возвращается медианное
    отклонение от медианной точки, в пикселях подложки.

    Смысл сигнала — **самосогласованность** позы, и он не смотрит на яркости
    вообще. Верная поза опирается на много независимых точек: выбрось любую
    треть — центр не сдвинется. Ложная держится на случайном совпадении
    нескольких точек, и её центр гуляет. Это ровно то свойство, которое NCC
    измерить не может ни в какой сезон.
    """
    mask = np.asarray(inlier_mask, dtype=bool)
    q, r = corr.pts_q[mask], corr.pts_r[mask]
    n = int(q.shape[0])
    if n < 4:
        return float("inf")               # по трём точкам разброс не оценить
    rng = np.random.default_rng(seed)
    centre = np.asarray(centre_px, dtype=float)
    centres = []
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        transform = fit_similarity(q[idx], r[idx])
        if transform is not None:
            centres.append(transform.apply(centre))
    if len(centres) < draws // 2:
        return float("inf")
    stacked = np.asarray(centres, dtype=float)
    median = np.median(stacked, axis=0)
    return float(np.median(np.linalg.norm(stacked - median, axis=1)))
