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
    "DinoV2Encoder",
    "MegaLocEncoder",
    "PCAReducer",
    "AnnBackend",
    "NumpyAnn",
    "FaissAnn",
    "Cell",
    "RetrievalResult",
    "TerrainIndex",
    "recall_at_k",
    "should_localize",
    "calibrate_uniqueness_threshold",
]


@runtime_checkable
class Encoder(Protocol):
    """Сменное ядро retrieval-этажа: grayscale-изображение → нормированный вектор.

    Метод :meth:`encode_batch` **необязателен**: он есть у тяжёлых ядер, где пачка
    заметно дешевле поштучной обработки, и его наличие проверяется через
    ``hasattr``. Обвязка обязана работать и без него — тогда клетки кодируются по
    одной. Это оптимизация, а не расширение контракта.
    """

    @property
    def dim(self) -> int:
        """Размерность эмбеддинга."""
        ...

    def encode(self, gray: np.ndarray) -> np.ndarray:
        """Закодировать изображение в L2-нормированный вектор ``(dim,)`` float32."""
        ...


def _l2_rows(matrix: np.ndarray) -> np.ndarray:
    """Построчная L2-нормировка; нулевые строки остаются нулевыми."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0.0)


def _gray_batch_to_tensor(grays, image_size: int, torch, device):
    """Пачка серых кадров → тензор ``(N, 3, S, S)`` с ImageNet-нормализацией.

    Половина стоимости кодирования приходилась не на GPU, а на подготовку входа
    (см. ``docs/OPTIMIZATION_PLAN.md``): каждый кадр разворачивался в float32 RGB
    ещё на CPU и в таком виде уезжал на устройство. Здесь на устройство едет
    **uint8 в один канал** — в 12 раз меньше данных, — а разворот в три канала
    делается broadcast-ом уже на GPU и без копии.
    """
    stacked = np.stack([
        cv2.resize(g, (image_size, image_size), interpolation=cv2.INTER_AREA) for g in grays
    ])
    tensor = torch.from_numpy(stacked).to(device)
    tensor = tensor.to(torch.float32).div_(255.0).unsqueeze(1).expand(-1, 3, -1, -1)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (tensor - mean) / std


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


#: Размерность CLS-токена по варианту DINOv2.
_DINOV2_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


class DinoV2Encoder:
    """Сильный self-supervised энкодер (DINOv2) за тем же интерфейсом :class:`Encoder`.

    Это «боевое ядро» retrieval-этажа: глобальный CLS-токен DINOv2 устойчив к
    appearance gap куда лучше стенд-энкодера, и подключается **одной заменой
    энкодера** — вся обвязка (индекс, ANN, уникальность, интеграция) не меняется.

    torch грузится **лениво** (только при первом кодировании), а веса тянутся
    через ``torch.hub`` один раз. Без torch конструктор и :attr:`dim` работают
    (размерность известна по имени модели), а :meth:`encode` даёт понятную
    ошибку — как сетевые тесты, тяжёлая зависимость не требуется для импорта
    пакета. Валидацию Recall@K на DINOv2 запускать при установленном torch.
    """

    def __init__(
        self, model_name: str = "dinov2_vits14", *, image_size: int = 224, device: str | None = None
    ) -> None:
        if model_name not in _DINOV2_DIMS:
            raise ValueError(f"неизвестная модель {model_name!r}, доступны {sorted(_DINOV2_DIMS)}")
        if image_size % 14 != 0:
            raise ValueError(f"image_size должен быть кратен патчу 14, получено {image_size}")
        self.model_name = model_name
        self.image_size = image_size
        self._device = device
        self._model = None
        self._torch = None

    @property
    def dim(self) -> int:
        return _DINOV2_DIMS[self.model_name]

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - зависит от окружения
            raise RuntimeError(
                "DinoV2Encoder требует torch: установите `pip install torch torchvision` "
                "(и веса DINOv2 подтянутся через torch.hub при первом запуске)"
            ) from exc
        self._torch = torch
        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = torch.hub.load("facebookresearch/dinov2", self.model_name)
        self._model = self._model.to(self._device).eval()

    def encode(self, gray: np.ndarray) -> np.ndarray:
        return self.encode_batch([gray])[0]

    def encode_batch(self, grays) -> np.ndarray:
        """Закодировать пачку кадров: ``(N, dim)`` L2-нормированных строк."""
        grays = list(grays)
        for gray in grays:
            if gray.ndim != 2:
                raise ValueError(f"энкодер ждёт grayscale, получено {gray.ndim}D {gray.shape}")
        if not grays:
            return np.empty((0, self.dim), np.float32)
        self._ensure_model()
        torch = self._torch
        tensor = _gray_batch_to_tensor(grays, self.image_size, torch, self._device)
        with torch.no_grad():
            feats = self._model(tensor).cpu().numpy().astype(np.float32)
        return _l2_rows(feats.reshape(len(grays), -1))


class MegaLocEncoder:
    """VPR-энкодер MegaLoc (DINOv2-B + SALAD-агрегация) за интерфейсом :class:`Encoder`.

    Обученная агрегация мест поверх сильного бэкбона — против сырого CLS-токена
    DINOv2, который на кросс-дате (кадр↔спутник) топит верную клетку в хвост
    ранжирования (замерено: DRZ_19206 — ранг 506/2116). MegaLoc обучен на пяти
    разнородных датасетах и в литературе — топ zero-shot на надир↔ортоспутнике,
    поэтому должен поднять верную клетку в top-K, не меняя ничего вокруг (тот же
    протокол ``encode(gray)→вектор``, та же обвязка индекса/ANN/уникальности).

    Вход, как у ViT-VPR: серый кадр реплицируется в 3 канала, ресайз до кратного
    патчу 14 (322×322 по умолчанию), ImageNet-нормализация; выход — L2-нормированный
    дескриптор размерности 8448. torch и веса грузятся **лениво** через
    ``torch.hub`` (репозиторий ``gmberton/MegaLoc``); без torch конструктор и
    :attr:`dim` работают, а :meth:`encode` даёт понятную ошибку — как DINOv2,
    тяжёлая зависимость не нужна для импорта пакета.

    Замечание для ±5 км с ротационной аугментацией: dim 8448 требует PCA→~1024
    перед FAISS-индексом (иначе память индекса неподъёмна). Для A/B на компактном
    регионе (тысячи клеток) numpy-kNN по полному вектору годится напрямую.
    """

    _DIM = 8448  #: финальная размерность дескриптора MegaLoc (64×128 + 256).

    def __init__(
        self,
        repo: str = "gmberton/MegaLoc",
        entry: str = "get_trained_model",
        *,
        image_size: int = 322,
        device: str | None = None,
    ) -> None:
        if image_size % 14 != 0:
            raise ValueError(f"image_size должен быть кратен патчу 14, получено {image_size}")
        self.repo = repo
        self.entry = entry
        self.image_size = image_size
        self._device = device
        self._model = None
        self._torch = None

    @property
    def dim(self) -> int:
        return self._DIM

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - зависит от окружения
            raise RuntimeError(
                "MegaLocEncoder требует torch: установите `pip install torch torchvision` "
                "(веса MegaLoc подтянутся через torch.hub при первом запуске)"
            ) from exc
        self._torch = torch
        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = torch.hub.load(self.repo, self.entry, trust_repo=True)
        self._model = self._model.to(self._device).eval()

    def encode(self, gray: np.ndarray) -> np.ndarray:
        return self.encode_batch([gray])[0]

    def encode_batch(self, grays) -> np.ndarray:
        """Закодировать пачку кадров: ``(N, dim)`` L2-нормированных строк."""
        grays = list(grays)
        for gray in grays:
            if gray.ndim != 2:
                raise ValueError(f"энкодер ждёт grayscale, получено {gray.ndim}D {gray.shape}")
        if not grays:
            return np.empty((0, self.dim), np.float32)
        self._ensure_model()
        torch = self._torch
        tensor = _gray_batch_to_tensor(grays, self.image_size, torch, self._device)
        with torch.no_grad():
            feats = self._model(tensor).cpu().numpy().astype(np.float32)
        return _l2_rows(feats.reshape(len(grays), -1))


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


def _randomized_svd(
    xc: np.ndarray, k: int, *, seed: int = 0, n_oversamples: int = 10, n_iter: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Рандомизированный усечённый SVD центрированной матрицы (алгоритм Halko).

    Полный SVD матрицы (N × 8448) неподъёмен; здесь нужны только top-k правых
    сингулярных векторов. Случайная проекция + степенные итерации + малый точный
    SVD дают их за O(N·D·k), детерминированно (фиксированный seed). Возвращает
    ``(components (k, D), singular_values (k,))``.
    """
    n, d = xc.shape
    k = min(k, n, d)
    rng = np.random.default_rng(seed)
    sketch = min(k + n_oversamples, min(n, d))
    omega = rng.standard_normal((d, sketch)).astype(np.float32)
    y = xc @ omega  # (N, sketch)
    for _ in range(n_iter):  # степенные итерации — прижимают спектр к главному
        y = xc @ (xc.T @ y)
    q, _ = np.linalg.qr(y)  # ортонормированный базис диапазона (N, sketch)
    b = q.T @ xc  # (sketch, D) — проекция в малое подпространство
    _ub, sv, vt = np.linalg.svd(b, full_matrices=False)
    return vt[:k].astype(np.float32), sv[:k].astype(np.float32)


class PCAReducer:
    """PCA-редукция дескриптора Этажа 1 (опц. whitening) — сжатие без потери сути.

    Зачем ([JOURNAL.md], веха PCA+FAISS): dim 8448 × десятки тысяч клеток ×
    аугментация не влезает в память, и каждое сравнение дорого. PCA проецирует в
    ~1024-мерное подпространство максимальной дисперсии (память ÷8, сравнение ×8
    дешевле). Побочно **plain PCA поднимает ранг верной клетки** (отбрасывает
    шумовой хвост компонент): на реальном MegaLoc (DRZ_19206, 2 км) ранг 49 → 15.

    **Whitening по умолчанию ВЫКЛЮЧЕН** — вопреки книжной эвристике VPR, на нашем
    дескрипторе он *вредит*: малые сингулярные значения хвоста оценены плохо, и
    деление на них раздувает шум (на том же MegaLoc ранг 49 → 334). Оставлен опцией
    на случай другого дескриптора, где уравнивание анизотропных осей окупится.
    Выход L2-нормируется, чтобы скалярное произведение оставалось косинусом (FAISS IP/HNSW).

    Обучается на векторах индекса (``fit``), применяется и к клеткам, и к запросу
    (тот же редуктор — иначе координаты несопоставимы, как и один энкодер).
    """

    def __init__(self, n_components: int = 1024, *, whiten: bool = False, seed: int = 0) -> None:
        if n_components < 1:
            raise ValueError(f"n_components должно быть >= 1, получено {n_components}")
        self.n_components = int(n_components)
        self.whiten = bool(whiten)
        self.seed = int(seed)
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None  # (k, D)
        self.scale_: np.ndarray | None = None  # (k,)

    @property
    def dim(self) -> int:
        return 0 if self.components_ is None else int(self.components_.shape[0])

    @property
    def input_dim(self) -> int | None:
        return None if self.components_ is None else int(self.components_.shape[1])

    def fit(self, x: np.ndarray) -> PCAReducer:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"fit ждёт матрицу (N, D), получено {x.ndim}D {x.shape}")
        n = x.shape[0]
        self.mean_ = x.mean(axis=0)
        xc = x - self.mean_
        components, sv = _randomized_svd(xc, self.n_components, seed=self.seed)
        self.components_ = components
        if self.whiten:
            # координата i делится на std вдоль i-й компоненты (std = σ_i/√(N−1))
            self.scale_ = (np.sqrt(max(n - 1, 1)) / (sv + 1e-8)).astype(np.float32)
        else:
            self.scale_ = np.ones(components.shape[0], dtype=np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise RuntimeError("PCAReducer не обучен — вызовите fit")
        single = x.ndim == 1
        xm = np.atleast_2d(np.asarray(x, dtype=np.float32)) - self.mean_
        proj = (xm @ self.components_.T) * self.scale_  # (n, k), whitened
        norms = np.linalg.norm(proj, axis=1, keepdims=True)
        proj = np.divide(proj, norms, out=np.zeros_like(proj), where=norms > 0.0)
        return proj[0] if single else proj

    def state(self) -> dict[str, np.ndarray]:
        """Массивы для персистентности (совместно с индексом)."""
        return {"pca_mean": self.mean_, "pca_components": self.components_, "pca_scale": self.scale_}

    @classmethod
    def from_state(cls, mean: np.ndarray, components: np.ndarray, scale: np.ndarray) -> PCAReducer:
        red = cls(int(components.shape[0]))
        red.mean_ = mean.astype(np.float32)
        red.components_ = components.astype(np.float32)
        red.scale_ = scale.astype(np.float32)
        return red


@runtime_checkable
class AnnBackend(Protocol):
    """Сменный движок поиска ближайших соседей за неизменным ``TerrainIndex.query``.

    Инвариант ([RETRIEVAL.md]): «ANN сейчас — точный numpy-kNN… для больших
    территорий его заменит FAISS/HNSW; интерфейс query при этом не изменится».
    """

    uniqueness_pool: int  # сколько кандидатов тянуть для сигнала уникальности

    def build(self, matrix: np.ndarray) -> None:
        """Проиндексировать матрицу (N, dim) нормированных векторов."""
        ...

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """top-k: ``(индексы, близости)`` по убыванию близости."""
        ...


class NumpyAnn:
    """Точный kNN по косинусу (numpy) — поведение по умолчанию, как было.

    ``uniqueness_pool`` фактически вся выборка → сигнал уникальности считается по
    полному ранжированию (точно, как в исходной реализации).
    """

    uniqueness_pool = 1_000_000_000

    def __init__(self) -> None:
        self._matrix: np.ndarray | None = None

    def build(self, matrix: np.ndarray) -> None:
        self._matrix = matrix

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        sims = self._matrix @ query
        k = min(k, sims.shape[0])
        # argpartition даёт top-k за O(N), затем сортируем только их
        part = np.argpartition(sims, -k)[-k:]
        order = part[np.argsort(sims[part])[::-1]]
        return order, sims[order]


class FaissAnn:
    """ANN на FAISS (HNSW по умолчанию) — сублинейный поиск для больших территорий.

    HNSW-граф даёт O(log N) на запрос независимо от размера индекса и делает
    ротационную аугментацию/рост территории бесплатными по поиску (в отличие от
    линейного скана numpy). Метрика — inner product по нормированным векторам =
    косинус. Тяжёлая зависимость грузится **лениво**; без ``faiss`` конструктор
    работает, а ``build`` даёт понятную ошибку.

    ``uniqueness_pool`` ограничен (не тянуть весь индекс на каждый запрос):
    далёкий двойник самоподобия высокоближен по определению → попадает в top-пул.
    """

    def __init__(
        self,
        *,
        kind: str = "hnsw",
        hnsw_m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
        uniqueness_pool: int = 128,
    ) -> None:
        if kind not in ("hnsw", "flat"):
            raise ValueError(f"kind ∈ {{'hnsw','flat'}}, получено {kind!r}")
        self.kind = kind
        self.hnsw_m = hnsw_m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.uniqueness_pool = int(uniqueness_pool)
        self._index = None
        self._faiss = None

    def _ensure_faiss(self):
        if self._faiss is None:
            try:
                import faiss
            except ImportError as exc:  # pragma: no cover - зависит от окружения
                raise RuntimeError(
                    "FaissAnn требует faiss: установите `pip install faiss-cpu` "
                    "(или faiss-gpu). Без него используйте NumpyAnn (по умолчанию)."
                ) from exc
            self._faiss = faiss
        return self._faiss

    def build(self, matrix: np.ndarray) -> None:
        faiss = self._ensure_faiss()
        mat = np.ascontiguousarray(matrix, dtype=np.float32)
        d = int(mat.shape[1])
        if self.kind == "flat":
            index = faiss.IndexFlatIP(d)  # точный IP — эталон/малые индексы
        else:
            index = faiss.IndexHNSWFlat(d, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = self.ef_construction
            index.hnsw.efSearch = self.ef_search
        index.add(mat)
        self._index = index

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._index is None:
            raise RuntimeError("FaissAnn не построен — вызовите build")
        q = np.ascontiguousarray(np.atleast_2d(query), dtype=np.float32)
        k = min(k, int(self._index.ntotal))
        sims, idx = self._index.search(q, k)
        idx, sims = idx[0], sims[0]
        keep = idx >= 0  # FAISS помечает недостающих −1
        return idx[keep], sims[keep]


class TerrainIndex:
    """Многомасштабный индекс клеток подложки + ANN-запрос кадром.

    Строится офлайн из источника подложки (:class:`BasemapSource`), в рантайме
    отвечает на запрос кадром top-K клетками. По умолчанию — сырые нормированные
    векторы + точный numpy-kNN (как было). Для больших территорий:
    :meth:`compress` сжимает дескриптор PCA→~1024 (память/скорость), :meth:`use_faiss`
    переключает поиск на FAISS/HNSW (сублинейно) — оба **не меняют** интерфейс
    :meth:`query`. Порядок для ±5 км: ``build(...).compress().use_faiss()``.
    """

    def __init__(
        self, encoder: Encoder, *, reducer: PCAReducer | None = None, ann: AnnBackend | None = None
    ) -> None:
        self.encoder = encoder
        self._vectors: list[np.ndarray] = []
        self._cells: list[Cell] = []
        self._matrix: np.ndarray | None = None  # (N, dim) сырых векторов, лениво
        self._reducer = reducer
        self._reduced: np.ndarray | None = None  # (N, k) после PCA, лениво
        self._ann_backend = ann
        self._ann: AnnBackend | None = None  # построенный движок, лениво

    def __len__(self) -> int:
        return len(self._cells)

    # --- построение ---------------------------------------------------------

    def add(self, cell_gray: np.ndarray, cell: Cell) -> None:
        """Добавить одну клетку: закодировать (с её ротацией) и запомнить с метаданными."""
        self._vectors.append(self.encoder.encode(_rotate(cell_gray, cell.rotation_deg)))
        self._cells.append(cell)
        self._matrix = None
        self._reduced = None
        self._ann = None

    def build(
        self,
        basemap: BasemapSource,
        region: Georef,
        *,
        cell_size_px: int,
        overlap: float = 0.5,
        rotations_deg: Sequence[float] = (0.0,),
        zoom: int | None = None,
        batch_size: int = 8,
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
            batch_size: сколько клеток кодировать за раз, если энкодер умеет
                :meth:`Encoder.encode_batch`. Поштучный вызов оставляет GPU
                простаивать; замер на MegaLoc — 10.0 мс/вектор при батче 1 против
                5.1 мс при батче 8 (``docs/OPTIMIZATION_PLAN.md``). Больше 8 брать
                смысла мало: батч 32 даёт лишь +15% скорости ценой 2.4× памяти.
        """
        if not 0.0 <= overlap < 1.0:
            raise ValueError(f"overlap должен лежать в [0, 1), получено {overlap}")
        if batch_size < 1:
            raise ValueError(f"batch_size должен быть >= 1, получено {batch_size}")
        zoom = int(region.zoom if zoom is None else zoom)
        step = max(1, int(round(cell_size_px * (1.0 - overlap))))
        half = cell_size_px / 2.0

        # Батч копится только если ядро его поддерживает: наличие encode_batch —
        # необязательная часть протокола (см. :class:`Encoder`), и стенд-энкодер
        # без него обязан работать по-прежнему.
        batched = batch_size > 1 and hasattr(self.encoder, "encode_batch")
        pending_images: list[np.ndarray] = []
        pending_cells: list[Cell] = []

        def flush() -> None:
            if not pending_images:
                return
            vectors = self.encoder.encode_batch(pending_images)
            self._vectors.extend(np.asarray(v, dtype=np.float32) for v in vectors)
            self._cells.extend(pending_cells)
            self._matrix = None
            self._reduced = None
            self._ann = None
            pending_images.clear()
            pending_cells.clear()

        centers_x = np.arange(half, region.width - half + 1e-6, step)
        centers_y = np.arange(half, region.height - half + 1e-6, step)
        for py in centers_y:
            for px in centers_x:
                lon, lat = region.pixel_to_lonlat(float(px), float(py))
                cell_gray, cell_georef = basemap(float(lon), float(lat), zoom, cell_size_px, cell_size_px)
                if cell_gray.ndim == 3:
                    cell_gray = cv2.cvtColor(cell_gray, cv2.COLOR_BGR2GRAY)
                for rot in rotations_deg:
                    cell = Cell(
                        center_lon=cell_georef.center_lon,
                        center_lat=cell_georef.center_lat,
                        zoom=zoom,
                        size_px=cell_size_px,
                        rotation_deg=float(rot),
                    )
                    if batched:
                        pending_images.append(_rotate(cell_gray, cell.rotation_deg))
                        pending_cells.append(cell)
                        if len(pending_images) >= batch_size:
                            flush()
                    else:
                        self.add(cell_gray, cell)
        flush()
        return self

    # --- сжатие и движок поиска (масштаб на большие территории) --------------

    def compress(self, n_components: int = 1024, *, whiten: bool = False, seed: int = 0) -> TerrainIndex:
        """Сжать дескриптор клеток PCA→``n_components`` (whitening по умолч. выкл). После build.

        Обучает :class:`PCAReducer` на сырых векторах индекса и переводит поиск на
        сжатое подпространство — память ÷(dim/k), сравнение дешевле, ранг верной
        клетки обычно чуть выше (шумовой хвост отброшен). Запрос будет проходить тот
        же редуктор. Не меняет интерфейс :meth:`query`. Проекция всех клеток считается
        здесь (один раз), а не в первом запросе.
        """
        reducer = PCAReducer(n_components, whiten=whiten, seed=seed).fit(self._stacked())
        self._reducer = reducer
        self._reduced = reducer.transform(self._stacked())  # материализуем сразу
        self._ann = None
        return self

    def use_faiss(self, **kwargs) -> TerrainIndex:
        """Переключить поиск на FAISS/HNSW (сублинейно). После build/compress.

        Аргументы — в :class:`FaissAnn` (``kind``, ``hnsw_m``, ``ef_search``…).
        Без пакета ``faiss`` первый запрос даст понятную ошибку.
        """
        self._ann_backend = FaissAnn(**kwargs)
        self._ann = None
        return self

    # --- персистентность ----------------------------------------------------

    def save(self, path) -> None:
        """Сохранить индекс (поисковые векторы + метаданные, + PCA если сжат) в ``.npz``.

        Энкодер не сохраняется — это код; при загрузке его передают заново (тот
        же, что при построении, иначе векторы несопоставимы). Если индекс сжат PCA,
        сохраняются **сжатые** векторы и состояние редуктора; движок поиска (FAISS)
        не сохраняется — строится заново лениво при первом запросе.
        """
        arrays = dict(
            vectors=self._search_matrix(),
            center_lon=np.array([c.center_lon for c in self._cells], dtype=np.float64),
            center_lat=np.array([c.center_lat for c in self._cells], dtype=np.float64),
            zoom=np.array([c.zoom for c in self._cells], dtype=np.int32),
            size_px=np.array([c.size_px for c in self._cells], dtype=np.int32),
            rotation_deg=np.array([c.rotation_deg for c in self._cells], dtype=np.float64),
        )
        if self._reducer is not None:
            arrays.update(self._reducer.state())
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path, encoder: Encoder) -> TerrainIndex:
        """Загрузить индекс из ``.npz`` и привязать энкодер (тот же, что при сборке).

        Распознаёт сжатые индексы (наличие ключей ``pca_*``): восстанавливает
        редуктор и сжатую матрицу; совместимость энкодера проверяется по входной
        размерности PCA. Несжатые — как раньше, по размерности векторов.
        """
        data = np.load(path)
        vectors = data["vectors"].astype(np.float32)
        index = cls(encoder)
        if "pca_components" in data:
            if data["pca_components"].shape[1] != encoder.dim:
                raise ValueError(
                    f"вход PCA {data['pca_components'].shape[1]} не совпадает с энкодером "
                    f"({encoder.dim}) — вероятно, другой энкодер"
                )
            index._reducer = PCAReducer.from_state(
                data["pca_mean"], data["pca_components"], data["pca_scale"]
            )
            index._reduced = vectors  # уже сжатые
        else:
            if vectors.shape[1] != encoder.dim:
                raise ValueError(
                    f"размерность индекса {vectors.shape[1]} не совпадает с "
                    f"энкодером ({encoder.dim}) — вероятно, другой энкодер"
                )
            index._matrix = vectors
            index._vectors = list(vectors)
        index._cells = [
            Cell(float(lon), float(lat), int(z), int(s), float(r))
            for lon, lat, z, s, r in zip(
                data["center_lon"], data["center_lat"], data["zoom"],
                data["size_px"], data["rotation_deg"],
            )
        ]
        return index

    # --- запрос -------------------------------------------------------------

    def _stacked(self) -> np.ndarray:
        if self._matrix is None:
            self._matrix = (
                np.vstack(self._vectors)
                if self._vectors
                else np.empty((0, self.encoder.dim), np.float32)
            )
        return self._matrix

    def _search_matrix(self) -> np.ndarray:
        """Матрица, по которой идёт поиск: сжатая (если PCA) или сырая."""
        if self._reducer is None:
            return self._stacked()
        if self._reduced is None:
            self._reduced = self._reducer.transform(self._stacked())
        return self._reduced

    def _ensure_ann(self) -> AnnBackend:
        if self._ann is None:
            backend = self._ann_backend if self._ann_backend is not None else NumpyAnn()
            backend.build(self._search_matrix())
            self._ann = backend
        return self._ann

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
        if self._reducer is not None:
            vec = self._reducer.transform(vec)
        ann = self._ensure_ann()
        # Пул под сигнал уникальности: точный движок (numpy) отдаёт полное ранжирование,
        # приближённый (FAISS) — ограниченный top-пул (далёкий двойник высокоближен →
        # попадает в него). ``idx``/``sims`` выровнены и убывают по близости.
        pool = min(len(self), max(k, ann.uniqueness_pool))
        idx, sims = ann.search(vec, pool)

        best = self._cells[int(idx[0])]
        if suppress_radius_m is None:
            suppress_radius_m = best.size_px * ground_mpp(best.center_lat, best.zoom)
        # Уникальность: top-1 минус лучший далёкий конкурент (пространственный NMS).
        uniqueness = 1.0
        for pos in range(1, idx.size):
            cell = self._cells[int(idx[pos])]
            if haversine_m(best.center_lat, best.center_lon, cell.center_lat, cell.center_lon) > suppress_radius_m:
                uniqueness = float(sims[0] - sims[pos])
                break

        top = min(k, idx.size)
        return RetrievalResult(
            cells=[self._cells[int(i)] for i in idx[:top]],
            similarities=sims[:top],
            uniqueness=uniqueness,
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
