"""★ ЭТАЖ 1 — RETRIEVAL ★: «хеширование местности» для грубого поиска «ГДЕ примерно».

Фаза 3 из [PLAN.md](../docs/PLAN.md), детали — [RETRIEVAL.md](../docs/RETRIEVAL.md).
Задача — не точность, а отсечь 99% карты: кадр → эмбеддинг → ANN-поиск по индексу
клеток подложки → **top-K кандидатных клеток** с их георефами и сигналом
уникальности. Точную позу считает Этаж 2 (матчер) только на этих клетках.

Сменное ядро — энкодер
----------------------
Как матчер спрятан за интерфейсом :class:`~aero_geoloc.matcher.Matcher`, так и
энкодер спрятан за :class:`Encoder`. Всё вокруг (индекс, ANN, top-K, сигнал
уникальности, метрика Recall@K, интеграция в `localize`) от его внутренностей не
зависит. Здесь стоит простой :class:`AveragePoolEncoder` — «SIFT retrieval-этажа»:
им заводится и проверяется обвязка на синтетике. Сильный self-supervised энкодер
(DINOv2 / Salad) встанет за тем же интерфейсом позже и поднимет Recall@K на
appearance gap, не меняя ничего вокруг.

ANN сейчас — точный numpy-kNN по косинусу (нормированные векторы → скалярное
произведение). Для больших территорий его заменит FAISS/HNSW; интерфейс
:meth:`TerrainIndex.query` при этом не изменится.

Инвариантность к повороту
-------------------------
DINO/простой энкодер не инвариантны к yaw. Две стратегии ([RETRIEVAL.md]):
*доверяем yaw* → предповорот кадра на −yaw перед запросом (``prerotate_deg``);
*не доверяем* → индекс аугментируется повёрнутыми копиями клеток
(``rotations_deg``), и запрос находит ближайшую по ротации.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import cv2
import numpy as np

from .basemap import BasemapSource
from .geo import Georef, ground_mpp, haversine_m

__all__ = [
    "Encoder",
    "AveragePoolEncoder",
    "Cell",
    "RetrievalResult",
    "TerrainIndex",
    "recall_at_k",
    "should_localize",
    "calibrate_uniqueness_threshold",
]


@runtime_checkable
class Encoder(Protocol):
    """Сменное ядро retrieval-этажа: grayscale-изображение → нормированный вектор."""

    @property
    def dim(self) -> int:
        """Размерность эмбеддинга."""
        ...

    def encode(self, gray: np.ndarray) -> np.ndarray:
        """Закодировать изображение в L2-нормированный вектор ``(dim,)`` float32."""
        ...


class AveragePoolEncoder:
    """Стенд-энкодер: ресайз до ``grid×grid``, вычесть среднее, L2-норма.

    Простое детерминированное ядро для завода обвязки — «SIFT retrieval-этажа».
    Ресайз сам нормализует масштаб клетки, среднее убирает общий уровень яркости,
    L2 делает косинус метрикой близости. К appearance gap хрупок (как классика),
    зато без тяжёлых зависимостей и воспроизводим. DINOv2 встанет за
    :class:`Encoder` позже.
    """

    def __init__(self, grid: int = 24) -> None:
        if grid < 2:
            raise ValueError(f"grid должен быть >= 2, получено {grid}")
        self.grid = grid

    @property
    def dim(self) -> int:
        return self.grid * self.grid

    def encode(self, gray: np.ndarray) -> np.ndarray:
        if gray.ndim != 2:
            raise ValueError(f"энкодер ждёт grayscale, получено {gray.ndim}D {gray.shape}")
        small = cv2.resize(gray, (self.grid, self.grid), interpolation=cv2.INTER_AREA)
        vec = small.astype(np.float32).ravel()
        vec -= vec.mean()
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0.0 else vec


@dataclass(frozen=True)
class Cell:
    """Клетка индекса: её геореференс и ротация, под которой она закодирована.

    Attributes:
        center_lon, center_lat: центр клетки, градусы.
        zoom: уровень подложки, на котором вырезана клетка.
        size_px: сторона клетки в пикселях.
        rotation_deg: поворот, применённый к клетке перед кодированием (аугментация).
    """

    center_lon: float
    center_lat: float
    zoom: int
    size_px: int
    rotation_deg: float = 0.0


@dataclass(frozen=True)
class RetrievalResult:
    """Ответ retrieval: top-K клеток, их близости и сигнал уникальности.

    Attributes:
        cells: top-K клеток по убыванию близости.
        similarities: косинусные близости этих клеток, по убыванию.
        uniqueness: зазор близости между top-1 и лучшим **пространственно
            далёким** конкурентом (см. :meth:`TerrainIndex.query`). Большой →
            место уникально; ~0 → есть далёкий двойник (самоподобие) → повод
            занизить доверие ещё до матчинга ([RETRIEVAL.md]). Сравнивать с
            top-2 в лоб нельзя: при перекрытии клеток top-2 — обычно сосед того
            же места, а не двусмысленность.
    """

    cells: list[Cell]
    similarities: np.ndarray  # косинус, по убыванию
    uniqueness: float = 1.0

    @property
    def best(self) -> Cell | None:
        return self.cells[0] if self.cells else None


def _rotate(image: np.ndarray, deg: float) -> np.ndarray:
    """Повернуть изображение вокруг центра на ``deg`` (для ротационной аугментации)."""
    if deg % 360.0 == 0.0:
        return image
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), deg, 1.0)
    return cv2.warpAffine(
        image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )


class TerrainIndex:
    """Многомасштабный индекс клеток подложки + ANN-запрос кадром.

    Строится офлайн из источника подложки (:class:`BasemapSource`), в рантайме
    отвечает на запрос кадром top-K клетками. Векторы хранятся нормированными,
    поиск — точный kNN по косинусу (numpy); для больших территорий заменяется на
    FAISS/HNSW без изменения интерфейса.
    """

    def __init__(self, encoder: Encoder) -> None:
        self.encoder = encoder
        self._vectors: list[np.ndarray] = []
        self._cells: list[Cell] = []
        self._matrix: np.ndarray | None = None  # (N, dim), собирается лениво

    def __len__(self) -> int:
        return len(self._cells)

    # --- построение ---------------------------------------------------------

    def add(self, cell_gray: np.ndarray, cell: Cell) -> None:
        """Добавить одну клетку: закодировать (с её ротацией) и запомнить с метаданными."""
        self._vectors.append(self.encoder.encode(_rotate(cell_gray, cell.rotation_deg)))
        self._cells.append(cell)
        self._matrix = None

    def build(
        self,
        basemap: BasemapSource,
        region: Georef,
        *,
        cell_size_px: int,
        overlap: float = 0.5,
        rotations_deg: Sequence[float] = (0.0,),
        zoom: int | None = None,
    ) -> TerrainIndex:
        """Нарезать ``region`` на перекрывающиеся клетки и проиндексировать их.

        Клетки идут регулярной сеткой с шагом ``cell_size_px·(1−overlap)``;
        перекрытие ~50% сглаживает эффект «кадр попал между клеток». Каждая клетка
        тянется через ``basemap`` (тот же интерфейс, что у `localize`), поэтому
        индекс строится и по синтетике (``SceneBasemap``), и по реальному региону
        (``TileBasemap``). Для каждой клетки заводится по записи на каждую ротацию
        из ``rotations_deg`` (аугментация под неизвестный yaw).

        Args:
            region: геореференс области индексации (задаёт центр, зум и размер).
            cell_size_px: сторона клетки в пикселях (≈ footprint кадра).
            overlap: доля перекрытия соседних клеток в ``[0, 1)``.
            rotations_deg: углы ротационной аугментации.
            zoom: зум клеток; по умолчанию — зум ``region``.
        """
        if not 0.0 <= overlap < 1.0:
            raise ValueError(f"overlap должен лежать в [0, 1), получено {overlap}")
        zoom = int(region.zoom if zoom is None else zoom)
        step = max(1, int(round(cell_size_px * (1.0 - overlap))))
        half = cell_size_px / 2.0

        centers_x = np.arange(half, region.width - half + 1e-6, step)
        centers_y = np.arange(half, region.height - half + 1e-6, step)
        for py in centers_y:
            for px in centers_x:
                lon, lat = region.pixel_to_lonlat(float(px), float(py))
                cell_gray, cell_georef = basemap(float(lon), float(lat), zoom, cell_size_px, cell_size_px)
                if cell_gray.ndim == 3:
                    cell_gray = cv2.cvtColor(cell_gray, cv2.COLOR_BGR2GRAY)
                for rot in rotations_deg:
                    self.add(
                        cell_gray,
                        Cell(
                            center_lon=cell_georef.center_lon,
                            center_lat=cell_georef.center_lat,
                            zoom=zoom,
                            size_px=cell_size_px,
                            rotation_deg=float(rot),
                        ),
                    )
        return self

    # --- запрос -------------------------------------------------------------

    def _stacked(self) -> np.ndarray:
        if self._matrix is None:
            self._matrix = (
                np.vstack(self._vectors)
                if self._vectors
                else np.empty((0, self.encoder.dim), np.float32)
            )
        return self._matrix

    def query(
        self,
        frame_gray: np.ndarray,
        *,
        k: int = 5,
        prerotate_deg: float = 0.0,
        suppress_radius_m: float | None = None,
    ) -> RetrievalResult:
        """Найти top-K клеток, ближайших к кадру по косинусу эмбеддинга.

        Сигнал уникальности считается с **пространственным подавлением**: зазор
        близости между top-1 и лучшим конкурентом, отстоящим от top-1 дальше
        ``suppress_radius_m``. Так соседние перекрывающиеся клетки того же места
        не занижают уникальность — её роняет только далёкий двойник.

        Args:
            frame_gray: кадр (grayscale).
            k: сколько кандидатов вернуть (несколько — чтобы поймать неоднозначность).
            prerotate_deg: предповорот кадра. При доверии yaw сюда идёт ``−yaw``.
            suppress_radius_m: радиус подавления соседей; по умолчанию — footprint
                клетки top-1 (одна клетка вокруг = «то же место»).
        """
        if len(self) == 0:
            return RetrievalResult(cells=[], similarities=np.empty((0,), np.float32))
        vec = self.encoder.encode(_rotate(frame_gray, prerotate_deg))
        sims = self._stacked() @ vec

        order = np.argsort(sims)[::-1]
        best = self._cells[order[0]]
        if suppress_radius_m is None:
            suppress_radius_m = best.size_px * ground_mpp(best.center_lat, best.zoom)
        # Уникальность: top-1 минус лучший далёкий конкурент (пространственный NMS).
        uniqueness = 1.0
        for i in order[1:]:
            cell = self._cells[i]
            if haversine_m(best.center_lat, best.center_lon, cell.center_lat, cell.center_lon) > suppress_radius_m:
                uniqueness = float(sims[order[0]] - sims[i])
                break

        top = order[: min(k, sims.size)]
        return RetrievalResult(
            cells=[self._cells[i] for i in top], similarities=sims[top], uniqueness=uniqueness
        )


def should_localize(result: RetrievalResult, *, min_uniqueness: float) -> bool:
    """Стоит ли вообще пытаться локализоваться, или отказать по самоподобию.

    Флаг отказа Этажа 1 ([RETRIEVAL.md]): если лучший кандидат не выделяется на
    фоне далёкого двойника (``uniqueness < min_uniqueness``), сцена самоподобна,
    и матчинг с большой вероятностью даст уверенно-неверную точку — честнее
    отказать заранее. Порог берётся из :func:`calibrate_uniqueness_threshold`.
    """
    return bool(result.cells) and result.uniqueness >= min_uniqueness


def calibrate_uniqueness_threshold(
    uniqueness: Sequence[float], resolvable: Sequence[bool]
) -> float:
    """Подобрать порог уникальности, разделяющий разрешимые и неразрешимые места.

    Максимизирует индекс Юдена ``TPR − FPR`` по кандидатным порогам — классика
    для выбора точки среза, не завязанная на баланс классов. ``resolvable`` —
    признак фактической разрешимости (успех локализации / богатая vs однородная
    местность) на калибровочном наборе.

    Returns:
        Порог: решения с ``uniqueness ≥ порог`` считаются разрешимыми.
    """
    u = np.asarray(uniqueness, dtype=float)
    y = np.asarray(resolvable, dtype=bool)
    if u.size == 0 or y.all() or (~y).all():
        return 0.0  # нечего разделять — не отсекаем
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    best_threshold, best_j = 0.0, -np.inf
    for threshold in np.unique(u):
        predicted = u >= threshold
        tpr = int((predicted & y).sum()) / n_pos
        fpr = int((predicted & ~y).sum()) / n_neg
        if tpr - fpr > best_j:
            best_j, best_threshold = tpr - fpr, float(threshold)
    return best_threshold


def recall_at_k(
    index: TerrainIndex,
    queries: Sequence[tuple[np.ndarray, float, float, float]],
    *,
    k: int,
    radius_m: float,
) -> float:
    """Recall@K: доля запросов, где верная клетка попала в top-K.

    Клетка считается верной, если её центр отстоит от истинного центра кадра не
    дальше ``radius_m`` (т.е. клетка реально показывает место кадра).

    Args:
        queries: список ``(frame_gray, true_lat, true_lon, prerotate_deg)``.
        k: размер top-K.
        radius_m: порог совпадения центра клетки с истиной, метры.
    """
    if not queries:
        return 0.0
    hits = 0
    for frame_gray, true_lat, true_lon, prerotate in queries:
        result = index.query(frame_gray, k=k, prerotate_deg=prerotate)
        for cell in result.cells:
            if haversine_m(true_lat, true_lon, cell.center_lat, cell.center_lon) <= radius_m:
                hits += 1
                break
    return hits / len(queries)
