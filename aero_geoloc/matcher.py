"""★ СМЕННОЕ ЯДРО МАТЧИНГА ★ — интерфейс :class:`Matcher` и его реализации.

Центральное решение архитектуры (``docs/ARCHITECTURE.md``): матчер спрятан за
единым интерфейсом, и **всё остальное от его внутренностей не зависит**.
Смена матчера — смена одной строки конфигурации, стенд прогоняет любую
реализацию через один и тот же протокол.

Фаза 1 даёт классику (SIFT/AKAZE) — этого достаточно на синтетике, где
appearance gap отсутствует. Обученные матчеры (LightGlue, LoFTR, RoMa)
подключаются в фазе 4 за этим же интерфейсом.

Матчер ничего не знает про географию, приоры и качество: на входе две
grayscale-картинки, на выходе кандидатные соответствия с мусором. Чистит
мусор :mod:`aero_geoloc.pose`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

__all__ = ["Correspondences", "Matcher", "SIFTMatcher", "AKAZEMatcher", "create_matcher"]


@dataclass(frozen=True)
class Correspondences:
    """Кандидатные соответствия «кадр ↔ подложка».

    Attributes:
        pts_q: ``(N, 2)`` float32 — точки в кадре (query), координаты пикселей.
        pts_r: ``(N, 2)`` float32 — соответствующие точки в подложке (reference).
        conf: ``(N,)`` float32 — уверенность на пару в ``[0, 1]``, больше =
            надёжнее. Используется для взвешивания в pose/quality.
    """

    pts_q: np.ndarray
    pts_r: np.ndarray
    conf: np.ndarray

    def __post_init__(self) -> None:
        if self.pts_q.shape != self.pts_r.shape:
            raise ValueError(
                f"pts_q и pts_r должны быть одной формы, получено "
                f"{self.pts_q.shape} и {self.pts_r.shape}"
            )
        if self.pts_q.ndim != 2 or self.pts_q.shape[1] != 2:
            raise ValueError(f"pts_q должен быть (N, 2), получено {self.pts_q.shape}")
        if self.conf.shape != (self.pts_q.shape[0],):
            raise ValueError(
                f"conf должен быть (N,), получено {self.conf.shape} при N={len(self.pts_q)}"
            )

    def __len__(self) -> int:
        return int(self.pts_q.shape[0])

    @classmethod
    def empty(cls) -> Correspondences:
        """Пустой набор — штатный результат, когда матчить нечего."""
        return cls(
            pts_q=np.empty((0, 2), dtype=np.float32),
            pts_r=np.empty((0, 2), dtype=np.float32),
            conf=np.empty((0,), dtype=np.float32),
        )

    def take(self, mask: np.ndarray) -> Correspondences:
        """Подмножество соответствий по булевой маске или массиву индексов."""
        return Correspondences(pts_q=self.pts_q[mask], pts_r=self.pts_r[mask], conf=self.conf[mask])


@runtime_checkable
class Matcher(Protocol):
    """Протокол сменного ядра. Единственный метод, никакого состояния наружу."""

    def match(self, query_gray: np.ndarray, ref_gray: np.ndarray) -> Correspondences:
        """Найти соответствия между кадром и окном подложки (обе — grayscale uint8)."""
        ...


def _check_gray(img: np.ndarray, name: str) -> np.ndarray:
    """Привести вход к 2D uint8, ругаясь на явно неправильные данные."""
    if img is None or img.size == 0:
        raise ValueError(f"{name}: пустое изображение")
    if img.ndim == 3:
        raise ValueError(f"{name}: ожидается grayscale, получено {img.ndim}D {img.shape}")
    if img.dtype != np.uint8:
        raise ValueError(f"{name}: ожидается uint8, получено {img.dtype}")
    return img


class _DescriptorMatcher:
    """Общая часть классических детекторов: детект → дескрипторы → тест Лоу.

    Тест отношения Лоу (Lowe ratio) отбрасывает неоднозначные соответствия:
    если ближайший дескриптор не сильно ближе второго, точка неуникальна.
    Отсюда же берётся уверенность: ``conf = 1 − d1/d2``.
    """

    def __init__(self, detector, norm_type: int, ratio: float, max_matches: int | None) -> None:
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"ratio должен лежать в (0, 1], получено {ratio}")
        self._detector = detector
        self._matcher = cv2.BFMatcher(norm_type, crossCheck=False)
        self.ratio = ratio
        self.max_matches = max_matches

    def match(self, query_gray: np.ndarray, ref_gray: np.ndarray) -> Correspondences:
        _check_gray(query_gray, "query_gray")
        _check_gray(ref_gray, "ref_gray")

        kp_q, des_q = self._detector.detectAndCompute(query_gray, None)
        kp_r, des_r = self._detector.detectAndCompute(ref_gray, None)
        # Для теста Лоу нужны минимум два кандидата на стороне подложки.
        if des_q is None or des_r is None or len(kp_q) < 2 or len(kp_r) < 2:
            return Correspondences.empty()

        pts_q, pts_r, conf = [], [], []
        for pair in self._matcher.knnMatch(des_q, des_r, k=2):
            if len(pair) < 2:
                continue
            best, second = pair
            if second.distance <= 0.0 or best.distance >= self.ratio * second.distance:
                continue
            pts_q.append(kp_q[best.queryIdx].pt)
            pts_r.append(kp_r[best.trainIdx].pt)
            conf.append(1.0 - best.distance / second.distance)

        if not pts_q:
            return Correspondences.empty()

        corr = Correspondences(
            pts_q=np.asarray(pts_q, dtype=np.float32),
            pts_r=np.asarray(pts_r, dtype=np.float32),
            conf=np.asarray(conf, dtype=np.float32),
        )
        if self.max_matches is not None and len(corr) > self.max_matches:
            # Оставляем самые уникальные — RANSAC дешевле на меньшем наборе.
            keep = np.argsort(corr.conf)[::-1][: self.max_matches]
            corr = corr.take(keep)
        return corr


class SIFTMatcher(_DescriptorMatcher):
    """Классика фазы 1: SIFT + тест отношения Лоу.

    Инвариантен к масштабу и повороту, что закрывает неизвестные yaw и ошибку
    высоты без предповорота кадра. На синтетике (кроп из той же подложки) даёт
    практически идеальную привязку; на реальном appearance gap разваливается —
    это и есть повод для фазы 4.

    Args:
        n_features: ограничение числа точек (0 = без ограничения).
        ratio: порог теста Лоу. 0.75 — умеренно строгий; ниже = чище, но меньше.
        contrast_threshold: порог отсева слабоконтрастных точек SIFT.
        max_matches: верхняя граница числа возвращаемых пар.
    """

    def __init__(
        self,
        *,
        n_features: int = 0,
        ratio: float = 0.75,
        contrast_threshold: float = 0.04,
        edge_threshold: float = 10.0,
        max_matches: int | None = 5000,
    ) -> None:
        detector = cv2.SIFT_create(
            nfeatures=n_features,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
        )
        super().__init__(detector, cv2.NORM_L2, ratio, max_matches)


def _akaze_create(threshold: float):
    """AKAZE живёт в разных местах в OpenCV 4 и 5 — находим, где есть."""
    factory = getattr(cv2, "AKAZE_create", None)
    if factory is None:  # OpenCV 5: переехал в contrib-модуль xfeatures2d
        factory = getattr(getattr(cv2, "xfeatures2d", None), "AKAZE_create", None)
    if factory is None:
        raise RuntimeError(
            "AKAZE недоступен в этой сборке OpenCV; поставьте opencv-contrib-python"
        )
    return factory(threshold=threshold)


class AKAZEMatcher(_DescriptorMatcher):
    """Альтернативная классика: AKAZE с бинарными дескрипторами (Hamming).

    Заметно быстрее SIFT и часто устойчивее на «размытых» текстурах, но даёт
    меньше точек. Существует в фазе 1 прежде всего как доказательство того, что
    интерфейс :class:`Matcher` действительно сменный: стенд гоняет обе
    реализации одним и тем же протоколом.

    Args:
        threshold: порог детектора. Дефолт понижен на порядок относительно
            штатного для OpenCV (0.001): тот подобран под контрастные бытовые
            фото, а надирная съёмка и спутниковая подложка локально куда более
            вялые по контрасту, и на них детектор при 0.001 голодает —
            на сцене стенда это разница между 17 и 350 соответствиями.
            На контрастной местности порог имеет смысл поднять обратно.
        ratio: порог теста Лоу. Для бинарных дескрипторов берут мягче, чем для SIFT.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.0001,
        ratio: float = 0.8,
        max_matches: int | None = 5000,
    ) -> None:
        super().__init__(_akaze_create(threshold), cv2.NORM_HAMMING, ratio, max_matches)


_REGISTRY = {"sift": SIFTMatcher, "akaze": AKAZEMatcher}


def create_matcher(name: str = "sift", **kwargs) -> Matcher:
    """Собрать матчер по имени — та самая «одна строка конфигурации».

    Args:
        name: ``"sift"`` или ``"akaze"``. Реализации фазы 4 регистрируются здесь же.
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"неизвестный матчер {name!r}, доступны: {sorted(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)
