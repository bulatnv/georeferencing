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

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import re

import cv2
import numpy as np

from .weights import apply_state_dict, checkpoint_path, read_state_dict

__all__ = [
    "Correspondences",
    "Matcher",
    "SIFTMatcher",
    "ResizedMatcher",
    "AKAZEMatcher",
    "LightGlueMatcher",
    "LoFTRMatcher",
    "RoMaMatcher",
    "RoMaV2Matcher",
    "create_matcher",
    "lightglue_state_dict",
]


@dataclass(frozen=True)
class Correspondences:
    """Кандидатные соответствия «кадр ↔ подложка».

    Attributes:
        pts_q: ``(N, 2)`` float32 — точки в кадре (query), координаты пикселей.
        pts_r: ``(N, 2)`` float32 — соответствующие точки в подложке (reference).
        conf: ``(N,)`` float32 — уверенность на пару в ``[0, 1]``, больше =
            надёжнее. Используется для взвешивания в pose/quality.
        evidence: необязательные свидетельства уровня **всей пары картинок**, а не
            отдельных точек. Нужны потому, что не всё, что знает матчер, ложится
            на точки: плотное ядро оценивает уверенность по ВСЕМУ полю, и доля
            уверенной площади — свойство пары, а не какой-то из выборок. Словарь
            намеренно свободный: связка качества берёт из него то, что для
            текущего ядра откалибровано, а отсутствие ключа — штатный случай
            (у разреженных ядер такого поля просто нет).
    """

    pts_q: np.ndarray
    pts_r: np.ndarray
    conf: np.ndarray
    evidence: dict = field(default_factory=dict)

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
        """Подмножество соответствий по булевой маске или массиву индексов.

        Свидетельства уровня пары переносятся как есть: они не про точки, и
        фильтрация точек их не меняет.
        """
        return Correspondences(pts_q=self.pts_q[mask], pts_r=self.pts_r[mask],
                               conf=self.conf[mask], evidence=dict(self.evidence))


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
            f"AKAZE недоступен в этой сборке OpenCV {cv2.__version__}. Самая частая "
            "причина — в окружении стоят СРАЗУ обе сборки, opencv-python и "
            "opencv-contrib-python: они распаковываются в один каталог `cv2` и "
            "затирают друг друга, после чего contrib-модули пропадают. Проверьте "
            "`pip list | grep opencv`; если сборок две, оставьте одну:\n"
            "    pip uninstall -y opencv-python opencv-contrib-python\n"
            "    pip install opencv-contrib-python\n"
            "Ловушка срабатывает при установке пакетов, зависящих от opencv-python "
            "(например, lightglue), — см. шапку requirements-real.txt."
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


class _LearnedMatcher:
    """Общая часть обучаемых матчеров фазы 4: ленивая загрузка тяжёлого ядра.

    Как :class:`~aero_geoloc.retrieval.DinoV2Encoder`, torch грузится **лениво**
    (только при первом ``match``), веса — один раз. Без зависимостей конструктор
    работает (пакет импортируется без torch), а ``match`` даёт понятную ошибку.
    Это и есть смысл сменного ядра: обвязка (pose, quality, стенд, retrieval) не
    зависит от того, классический матчер внутри или обученный.
    """

    _requires = "torch"

    def __init__(self, *, device: str | None = None) -> None:
        self._device = device
        self._torch = None
        self._loaded = False

    def _import(self):  # pragma: no cover - зависит от окружения
        raise NotImplementedError

    def _missing(self, exc: ImportError) -> RuntimeError:
        """Единая понятная ошибка на любую недостающую часть тяжёлого ядра."""
        missing = f" (не найден {exc.name!r})" if getattr(exc, "name", None) else ""
        return RuntimeError(
            f"{type(self).__name__} требует {self._requires}{missing}: поставьте набор "
            "`pip install -r requirements-real.txt` (веса подтянутся при первом запуске)"
        )

    def _ensure(self) -> None:
        if self._loaded:
            return
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - зависит от окружения
            raise self._missing(exc) from exc
        self._torch = torch
        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self._import()
        except ImportError as exc:
            # torch есть, а ядра матчера нет — самый частый случай на полпути к
            # боевому окружению. Без этой ветки наружу летел голый
            # ModuleNotFoundError мимо контракта «понятная ошибка, как у сети».
            raise self._missing(exc) from exc
        self._loaded = True

    def _tensor(self, gray: np.ndarray):
        return self._torch.from_numpy(gray.astype(np.float32) / 255.0)[None, None].to(self._device)


def lightglue_state_dict(state: dict) -> dict:
    """Веса GIM/MINIMA -> именование ``lightglue`` из cvg.

    Разница чисто в раскладке слоёв: у GIM (и наследующего ему MINIMA) само- и
    кросс-внимание лежат отдельными списками ``self_attn.{i}`` и
    ``cross_attn.{i}``, а у cvg они собраны в один блок
    ``transformers.{i}.{self|cross}_attn``. Формы совпадают полностью — это одна
    архитектура, записанная двумя способами, а не две разные сети.
    """
    out = {}
    for key, value in state.items():
        match = re.match(r"^(self_attn|cross_attn)\.(\d+)\.(.*)$", key)
        if match:
            kind, index, tail = match.groups()
            out[f"transformers.{index}.{kind}.{tail}"] = value
        else:
            out[key] = value
    return out


class LightGlueMatcher(_LearnedMatcher):
    """SuperPoint + LightGlue (``lightglue``) за интерфейсом :class:`Matcher` — фаза 4.

    Быстрый и точный на умеренном appearance gap. Смена классики на него —
    ``create_matcher("lightglue")``, и всё выше матчера не меняется.

    Args:
        max_keypoints: верхняя граница числа точек SuperPoint.
        min_score: порог уверенности LightGlue для пары.
        checkpoint: имя внешних весов из :data:`aero_geoloc.weights.CHECKPOINTS`
            (``gim_lightglue`` / ``minima_lightglue``) вместо штатных. Смена
            обучения — это ровно один аргумент; архитектура та же.
    """

    _requires = "torch и lightglue"

    def __init__(self, *, max_keypoints: int = 2048, min_score: float = 0.0,
                 checkpoint: str | None = None, device: str | None = None) -> None:
        super().__init__(device=device)
        self.max_keypoints = max_keypoints
        self.min_score = min_score
        self.checkpoint = checkpoint
        self.loaded_tensors: dict[str, int] = {}
        self._extractor = None
        self._matcher = None

    def _import(self) -> None:  # pragma: no cover - требует torch+lightglue
        from lightglue import LightGlue, SuperPoint

        extractor = SuperPoint(max_num_keypoints=self.max_keypoints).eval()
        matcher = LightGlue(features="superpoint").eval()
        if self.checkpoint:
            state = read_state_dict(checkpoint_path(self.checkpoint))
            # У GIM веса лежат под model./superpoint., у MINIMA — плоско: обе
            # раскладки разбираются одинаково, чтобы не заводить ветку на автора.
            body = {k[len("model."):]: v for k, v in state.items()
                    if k.startswith("model.")} or state
            self.loaded_tensors["lightglue"] = apply_state_dict(
                matcher, lightglue_state_dict(body), label=self.checkpoint)
            head = {k[len("superpoint."):]: v for k, v in state.items()
                    if k.startswith("superpoint.")}
            # Свой SuperPoint есть не у всех: MINIMA дообучает только матчер.
            # Собирать ядро из половин двух разных обучений нельзя, поэтому
            # детектор берётся из чекпоинта, только если он там действительно есть.
            if head:
                self.loaded_tensors["superpoint"] = apply_state_dict(
                    extractor, head, label=f"{self.checkpoint}:superpoint")
        self._extractor = extractor.to(self._device)
        self._matcher = matcher.to(self._device)

    def match(self, query_gray: np.ndarray, ref_gray: np.ndarray) -> Correspondences:
        _check_gray(query_gray, "query_gray")
        _check_gray(ref_gray, "ref_gray")
        self._ensure()
        from lightglue.utils import rbd  # pragma: no cover

        with self._torch.no_grad():  # pragma: no cover - требует torch
            f0 = self._extractor.extract(self._tensor(query_gray))
            f1 = self._extractor.extract(self._tensor(ref_gray))
            out = self._matcher({"image0": f0, "image1": f1})
        f0, f1, out = (rbd(x) for x in (f0, f1, out))
        matches = out["matches"]
        if matches.shape[0] == 0:
            return Correspondences.empty()
        pts_q = f0["keypoints"][matches[:, 0]].cpu().numpy().astype(np.float32)
        pts_r = f1["keypoints"][matches[:, 1]].cpu().numpy().astype(np.float32)
        scores = out.get("scores")
        conf = (
            scores.detach().cpu().numpy().astype(np.float32)
            if scores is not None
            else np.ones(len(pts_q), np.float32)
        )
        corr = Correspondences(pts_q, pts_r, conf)
        keep = conf >= self.min_score
        return corr.take(keep) if not keep.all() else corr


class LoFTRMatcher(_LearnedMatcher):
    """Detector-free матчер LoFTR (``kornia``) за интерфейсом :class:`Matcher` — фаза 4.

    Плотный матчинг без детектора — сильнее на слабо-текстурных сценах, где
    классике не хватает точек. Args: ``pretrained`` — ``"outdoor"``/``"indoor"``,
    ``min_conf`` — порог уверенности пары, ``checkpoint`` — внешние веса
    (``minima_loftr``) вместо штатных.
    """

    _requires = "torch и kornia"

    def __init__(self, *, pretrained: str = "outdoor", min_conf: float = 0.5,
                 checkpoint: str | None = None, device: str | None = None) -> None:
        super().__init__(device=device)
        self.pretrained = pretrained
        self.min_conf = min_conf
        self.checkpoint = checkpoint
        self.loaded_tensors: dict[str, int] = {}
        self._model = None

    def _import(self) -> None:  # pragma: no cover - требует torch+kornia
        import kornia

        model = kornia.feature.LoFTR(pretrained=None if self.checkpoint else self.pretrained)
        if self.checkpoint:
            state = read_state_dict(checkpoint_path(self.checkpoint))
            body = {k[len("matcher."):]: v for k, v in state.items()
                    if k.startswith("matcher.")} or state
            self.loaded_tensors["loftr"] = apply_state_dict(model, body, label=self.checkpoint)
        self._model = model.eval().to(self._device)

    def match(self, query_gray: np.ndarray, ref_gray: np.ndarray) -> Correspondences:
        _check_gray(query_gray, "query_gray")
        _check_gray(ref_gray, "ref_gray")
        self._ensure()
        with self._torch.no_grad():  # pragma: no cover - требует torch
            out = self._model({"image0": self._tensor(query_gray), "image1": self._tensor(ref_gray)})
        pts_q = out["keypoints0"].cpu().numpy().astype(np.float32)
        pts_r = out["keypoints1"].cpu().numpy().astype(np.float32)
        conf = out["confidence"].cpu().numpy().astype(np.float32)
        if len(pts_q) == 0:
            return Correspondences.empty()
        corr = Correspondences(pts_q, pts_r, conf)
        keep = conf >= self.min_conf
        return corr.take(keep) if not keep.all() else corr



class ResizedMatcher:
    """Матчер на РАБОЧЕМ разрешении: обе картинки уменьшаются ОДНИМ коэффициентом.

    Зачем. Плотные матчеры (LoFTR, а дальше RoMa/MatchAnything/MINIMA —
    ``docs/ROADMAP.md``, фазы 2–3) работают на своей рабочей сетке и полное
    разрешение либо не тянут, либо считают его впустую. Обёртка приводит вход к
    рабочему размеру и **возвращает соответствия в исходных пикселях**, поэтому
    всё выше матчера (pose, quality, георефа) ничего не замечает — это тот же
    инвариант сменного ядра.

    Главное здесь — **общий коэффициент**. Ловушка сработала дважды на одном
    LoFTR и оба раза дала неверный вывод (``docs/JOURNAL.md``):

    1. подача полного разрешения — «0 инлайеров», хотя матчер просто не тот вход
       получил;
    2. приведение КАЖДОЙ картинки к 640 по отдельности. Кадр и окно подложки
       покрывают разную площадь земли, поэтому раздельная нормировка вносит
       расхождение масштабов (у нас 1.5×) — 11 инлайеров вместо 103.

    Поэтому коэффициент считается по наибольшей стороне ОБЕИХ картинок сразу:
    их взаимный масштаб сохраняется в точности.

    ``pad_to`` добивает размеры до кратности (LoFTR внутри делит на 8). Дополнение
    идёт справа и снизу нулями, поэтому координаты точек не сдвигаются.
    """

    def __init__(self, inner: Matcher, *, max_side: int = 640, pad_to: int = 8) -> None:
        self.inner = inner
        self.max_side = int(max_side)
        self.pad_to = int(pad_to)

    def _prepare(self, gray: np.ndarray, scale: float) -> np.ndarray:
        h, w = gray.shape[:2]
        if scale < 1.0:
            gray = cv2.resize(gray, (max(1, round(w * scale)), max(1, round(h * scale))),
                              interpolation=cv2.INTER_AREA)
        if self.pad_to > 1:
            h, w = gray.shape[:2]
            ph, pw = (-h) % self.pad_to, (-w) % self.pad_to
            if ph or pw:
                gray = cv2.copyMakeBorder(gray, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
        return gray

    def match(self, query_gray: np.ndarray, ref_gray: np.ndarray) -> Correspondences:
        _check_gray(query_gray, "query_gray")
        _check_gray(ref_gray, "ref_gray")
        longest = max(query_gray.shape[0], query_gray.shape[1],
                      ref_gray.shape[0], ref_gray.shape[1])
        scale = min(1.0, self.max_side / float(longest)) if self.max_side > 0 else 1.0
        corr = self.inner.match(self._prepare(query_gray, scale),
                                self._prepare(ref_gray, scale))
        if scale >= 1.0 or len(corr) == 0:
            return corr
        return Correspondences(pts_q=(corr.pts_q / scale).astype(np.float32),
                               pts_r=(corr.pts_r / scale).astype(np.float32),
                               conf=corr.conf, evidence=dict(corr.evidence))



class RoMaMatcher(_LearnedMatcher):
    """Плотный матчер RoMa (``romatch``) за интерфейсом :class:`Matcher` — фаза 3.

    Отличие от всех предыдущих ядер принципиальное: **нет детектора**. RoMa
    предсказывает плотное поле соответствий (warp) и попиксельную уверенность
    (certainty), а разреженные пары получаются сэмплированием по этой
    уверенности. Именно поэтому он и интересен на смене сезона: у разреженных
    ядер рассыпается **повторяемость ключевых точек**, а здесь повторять нечего.

    Побочный выход — та самая certainty, которую обзор направления B предлагал
    как готовый сигнал качества (``docs/RESEARCH_B_VERIFICATION.md``): она
    возвращается в ``Correspondences.conf`` и доходит до ``quality.assess``.

    **Про масштаб.** RoMa приводит каждую картинку к своему квадрату (560 на
    грубом уровне, 864 на уточнении) — то есть ресайзит их РАЗДЕЛЬНО и с разным
    соотношением сторон. На первый взгляд это ровно та ловушка, что дважды
    испортила замеры LoFTR, но здесь она не срабатывает: координаты возвращаются
    через ``to_pixel_coordinates`` в ИСХОДНЫХ пикселях каждой картинки, и наружу
    выходит правильная геометрия. Оборачивать этот матчер в
    :class:`ResizedMatcher` не нужно и вредно — он ужимает вход сам.

    Args:
        checkpoint: имя весов из :data:`aero_geoloc.weights.CHECKPOINTS`
            (по умолчанию ``minima_roma``). ``None`` — штатные веса RoMa, которые
            ``romatch`` скачает сам.
        max_samples: сколько пар сэмплировать из плотного поля.
        min_conf: порог уверенности пары.
        coarse_res, upsample_res: рабочие разрешения RoMa. Меньше — быстрее и
            грубее; трогать осознанно, они же задают потолок точности.
        cover_thresh: уровень, выше которого пиксель считается уверенным при
            подсчёте ``certainty_cover``.
    """

    _requires = "torch и romatch"

    def __init__(self, *, checkpoint: str | None = "minima_roma", max_samples: int = 2048,
                 min_conf: float = 0.5, coarse_res: int = 560, upsample_res: int = 864,
                 cover_thresh: float = 0.5, device: str | None = None) -> None:
        super().__init__(device=device)
        self.checkpoint = checkpoint
        self.max_samples = max_samples
        self.min_conf = min_conf
        self.coarse_res = coarse_res
        self.upsample_res = upsample_res
        self.cover_thresh = cover_thresh
        self.loaded_tensors: dict[str, int] = {}
        self._model = None

    def _import(self) -> None:  # pragma: no cover - требует torch+romatch
        from romatch import roma_outdoor

        weights = None
        if self.checkpoint:
            weights = read_state_dict(checkpoint_path(self.checkpoint))
        model = roma_outdoor(
            device=self._device, weights=weights, coarse_res=self.coarse_res,
            upsample_res=self.upsample_res,
        )
        if weights is not None:
            # roma_outdoor грузит веса сам, но «загрузилось» проверяем мы: молчаливый
            # откат на случайную инициализацию дал бы правдоподобный, но бессмысленный
            # результат — та же ловушка, что и у подмен весов LightGlue.
            self.loaded_tensors["roma"] = apply_state_dict(model, weights, label=self.checkpoint)
        self._model = model

    def match(self, query_gray: np.ndarray, ref_gray: np.ndarray) -> Correspondences:
        _check_gray(query_gray, "query_gray")
        _check_gray(ref_gray, "ref_gray")
        self._ensure()
        from PIL import Image  # pragma: no cover

        with self._torch.inference_mode():  # pragma: no cover - требует torch
            im_q = Image.fromarray(cv2.cvtColor(query_gray, cv2.COLOR_GRAY2RGB))
            im_r = Image.fromarray(cv2.cvtColor(ref_gray, cv2.COLOR_GRAY2RGB))
            warp, certainty = self._model.match(im_q, im_r, device=self._device)
            # Сводка плотного поля снимается ДО sample: у RoMa режим сэмплирования
            # содержит "threshold", который обрезает всё выше порога в единицу, и
            # ``conf`` возвращённых пар выходит константой 1.0 — измерено, на верных
            # и на заведомо чужих парах одинаково. Информация живёт в поле целиком.
            cert = certainty.detach().float()
            evidence = {
                "certainty_mean": float(cert.mean().item()),
                "certainty_cover": float((cert > self.cover_thresh).float().mean().item()),
            }
            matches, conf = self._model.sample(warp, certainty, num=self.max_samples)
            if matches.shape[0] == 0:
                return Correspondences.empty()
            hq, wq = query_gray.shape[:2]
            hr, wr = ref_gray.shape[:2]
            pts_q, pts_r = self._model.to_pixel_coordinates(matches, hq, wq, hr, wr)
        corr = Correspondences(
            pts_q.detach().cpu().numpy().astype(np.float32),
            pts_r.detach().cpu().numpy().astype(np.float32),
            conf.detach().cpu().numpy().astype(np.float32),
            evidence=evidence,
        )
        keep = corr.conf >= self.min_conf
        return corr.take(keep) if not keep.all() else corr



class RoMaV2Matcher(_LearnedMatcher):
    """RoMa v2 (``romav2``) на замороженном DINOv3 — фаза 3 из ``docs/ROADMAP.md``.

    Отличие от :class:`RoMaMatcher` (v1) не только в качестве. У v1 уверенность
    возвращённых пар оказалась **константой 1.0** — режим сэмплирования обрезает
    всё выше порога, — и сигнал приходилось доставать из плотного поля отдельно.
    У v2 порог по умолчанию не задан, и ``overlaps`` выходят живыми (медиана
    0.97 при минимуме 0.20 на контрольной паре). Вдобавок v2 отдаёт **матрицу
    точности 2×2 на каждую пару** — ту самую ковариацию ошибки, ради которой
    направление и выбиралось.

    **Про веса и лицензии.** ``romav2`` тянет один чекпоинт (~1.02 ГБ) из релиза
    MIT-лицензированного репозитория, и замороженный **DINOv3 лежит внутри
    него** — поэтому доступ к gated-репозиторию Meta на HuggingFace не нужен.
    Юридически это не отменяет лицензию Meta на сами веса бэкбона: код RoMa v2
    под MIT, а веса DINOv3 распространяются под собственной лицензией Meta.
    Для проприетарного использования это проверять отдельно.

    Args:
        max_samples: сколько пар сэмплировать из плотного поля.
        min_conf: порог ``overlap`` для пары.
        compile: ``torch.compile`` модели. По умолчанию выключено: на Windows
            компиляция либо падает, либо стоит минуты при первом вызове, а выигрыш
            для наших размеров не измерен.
    """

    _requires = "torch и romav2"

    def __init__(self, *, max_samples: int = 2048, min_conf: float = 0.5,
                 compile: bool = False, device: str | None = None) -> None:
        super().__init__(device=device)
        self.max_samples = max_samples
        self.min_conf = min_conf
        self.compile = compile
        self._model = None

    def _import(self) -> None:  # pragma: no cover - требует torch+romav2
        from romav2 import RoMaV2

        self._model = RoMaV2(RoMaV2.Cfg(compile=self.compile)).to(self._device).eval()

    def match(self, query_gray: np.ndarray, ref_gray: np.ndarray) -> Correspondences:
        _check_gray(query_gray, "query_gray")
        _check_gray(ref_gray, "ref_gray")
        self._ensure()
        from PIL import Image  # pragma: no cover

        with self._torch.inference_mode():  # pragma: no cover - требует torch
            im_q = Image.fromarray(cv2.cvtColor(query_gray, cv2.COLOR_GRAY2RGB))
            im_r = Image.fromarray(cv2.cvtColor(ref_gray, cv2.COLOR_GRAY2RGB))
            preds = self._model.match(im_q, im_r)
            matches, overlaps, precision_ab, _ = self._model.sample(preds, self.max_samples)
            if matches.shape[0] == 0:
                return Correspondences.empty()
            hq, wq = query_gray.shape[:2]
            hr, wr = ref_gray.shape[:2]
            pts_q, pts_r = self._model.to_pixel_coordinates(matches, hq, wq, hr, wr)
            # След матрицы точности — насколько туго локализована пара. Больше =
            # увереннее. Это pair-level свидетельство: наружу идёт сводка, потому
            # что связке качества нужно одно число, а не 2048 матриц.
            trace = precision_ab.diagonal(dim1=-2, dim2=-1).sum(-1)
            evidence = {
                "overlap_mean": float(overlaps.mean().item()),
                "precision_median": float(trace.median().item()),
            }
        corr = Correspondences(
            pts_q.detach().cpu().numpy().astype(np.float32),
            pts_r.detach().cpu().numpy().astype(np.float32),
            overlaps.detach().cpu().numpy().astype(np.float32).reshape(-1),
            evidence=evidence,
        )
        keep = corr.conf >= self.min_conf
        return corr.take(keep) if not keep.all() else corr


#: Ядра по имени. Подмены весов — те же классы с другим чекпоинтом: архитектура
#: не меняется, меняется обучение (``docs/ROADMAP.md``, фаза 2).
_REGISTRY = {
    "sift": SIFTMatcher,
    "akaze": AKAZEMatcher,
    "lightglue": LightGlueMatcher,
    "loftr": LoFTRMatcher,
    "gim_lightglue": lambda **kw: LightGlueMatcher(checkpoint="gim_lightglue", **kw),
    "minima_lightglue": lambda **kw: LightGlueMatcher(checkpoint="minima_lightglue", **kw),
    "minima_loftr": lambda **kw: LoFTRMatcher(checkpoint="minima_loftr", **kw),
    "minima_roma": lambda **kw: RoMaMatcher(checkpoint="minima_roma", **kw),
    "romav2": RoMaV2Matcher,
    "roma": lambda **kw: RoMaMatcher(checkpoint=None, **kw),
}


def create_matcher(name: str = "sift", **kwargs) -> Matcher:
    """Собрать матчер по имени — та самая «одна строка конфигурации».

    Args:
        name: ``"sift"``/``"akaze"`` (классика, фаза 1) либо ``"lightglue"``/
            ``"loftr"`` (обучаемые, фаза 4; требуют torch и весов). Смена ядра —
            это ровно смена ``name``, всё остальное не меняется.
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"неизвестный матчер {name!r}, доступны: {sorted(_REGISTRY)}")
    max_side = int(kwargs.pop("max_side", 0))
    matcher = _REGISTRY[key](**kwargs)
    return ResizedMatcher(matcher, max_side=max_side) if max_side > 0 else matcher
