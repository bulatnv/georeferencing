"""Оценка качества: ковариация центра, эллипс ошибки, фьюжн доверия, статус.

Стадия 7 из ``docs/PIPELINE.md`` и инвариант 4 из ``docs/ARCHITECTURE.md``:
результат — **распределение, а не точка**. Модуль превращает геометрию решения и
набор сигналов в ковариацию центра (2×2, метры), эллипс ошибки и статус
``LOCALIZED / LOW_CONFIDENCE / NOT_LOCALIZED``.

Главный **калиброванный** выход — эллипс. Ковариация центра выводится строго:
это распространение ошибки метода наименьших квадратов подобия на центральную
точку кадра. Под гауссовым шумом локализации ключевых точек покрытие эллипса
совпадает с номиналом (проверяется Монте-Карло в ``tests/test_quality.py``:
доля истинных центров внутри 1σ-эллипса ≈ 39% для 2D, ≈ 86% для 2σ). Это и есть
критерий приёмки шага (``docs/TESTING.md``, «проверка калибровки»).

Скалярная ``confidence`` — монотонный фьюжн сигналов; её reliability-диаграмма
(P(верно | confidence)) калибруется уже на лестнице возмущений (следующий шаг
фазы 2), где есть и провалы. Поэтому статус здесь опирается на **калиброванную
неопределённость** (размер эллипса), а не на сырой скаляр — инвариант 6 из
``CLAUDE.md``: доверие не подделываем до проверенного покрытия эллипса.

Что ковариация ловит, а что нет
-------------------------------
Формула распространяет **случайную** ошибку локализации точек — она усредняется
как ``σ/√N``. Систематический сдвиг, общий для всех точек (субпиксельная
предвзятость детектора, интерполяция), так НЕ усредняется, и ковариация его не
видит. На чистой синтетике L0 именно он и доминирует (ошибка ~0.5 px —
устойчивый сдвиг, а не разброс), поэтому end-to-end покрытие на L0 вырождено и
**не** является целью калибровки: осмысленная сквозная проверка калибровки идёт
на лестнице возмущений (шаг 5), где ошибки крупнее и случайны. Абсолютную
погрешность геопривязки подложки (единицы метров, не зависит от ``N``)
добавляют изотропным полом ``systematic_floor_m`` — на синтетике он 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .matcher import Correspondences
from .pose import PoseEstimate, SimilarityTransform
from .types import Status

__all__ = [
    "MIN_NCC",
    "MIN_DINO",
    "MIN_INLIERS_HARD",
    "MIN_INLIERS_DENSE",
    "PHOTOMETRIC_THRESHOLDS",
    "center_covariance",
    "error_ellipse",
    "align_reference",
    "aligned_ncc",
    "aligned_structural",
    "QualityAssessment",
    "assess",
]

#: Калиброванные пороги связки качества — оба условия обязательны (см. :func:`assess`).
#: Значения получены на реальных дрон↔Esri матчах (``docs/JOURNAL.md``, веха
#: калибровки дискриминатора): связка «инлайеры ≥ 8 И NCC ≥ 0.12» даёт ~96%
#: точности. Вынесены в константы, потому что оркестрация (:mod:`aero_geoloc.localize`)
#: прокидывает порог NCC через свои уровни — с литералом в каждом дефолт
#: разъезжался бы молча.
MIN_NCC = 0.12
MIN_INLIERS_HARD = 8

#: Порог инлайеров для **плотных** ядер (RoMa и подобные), где счётчик значит
#: другое.
#:
#: Разреженный SuperPoint+LightGlue отдаёт от полусотни до пары тысяч
#: соответствий, и восемь инлайеров — это «нашлось хоть что-то». Плотное ядро
#: сэмплирует фиксированные 2048 пар из сплошного поля и на ЛЮБОЙ паре картинок
#: даёт сотни «инлайеров»: они укладываются в подобие просто потому, что поле
#: гладкое. Порог 8 там не значит ничего.
#:
#: Измерено на MINIMA-RoMa (15 верных поз, 28 ложных, ``scripts/e2_geometry.py``):
#: у верных инлайеров ≥ 234, у ложных ≤ 178, разделение идеальное (AUC 1.000 в
#: обоих режимах). Полный прогон дал верную позу с 195 инлайерами и две ложных с
#: 57 и 136 — отсюда порог посередине.
#:
#: **Запас тонкий: 178 против 195.** Число держится на одном наборе и двух ложных
#: срабатываниях; при пополнении данных пересчитать, а не унаследовать.
MIN_INLIERS_DENSE = 185

#: Порог для меры согласия «косинус плотных патч-токенов DINOv2» (E3, ROADMAP
#: фаза 1). Взят не из литературы, а из замера **на найденных позах**, а не на
#: оракульном выравнивании: порог применяется именно к тем числам, что считает
#: пайплайн (36 верных поз и 9 ложных, ``scripts/e2_geometry.py``).
#:
#: Почему вообще меняем меру: у NCC на ВЕРНЫХ парах разброс от −0.03 до 0.66,
#: то есть разброс между кадрами больше сезонного, и единого порога у неё нет.
#: Из-за этого калиброванный порог 0.12 режет верную локализацию ``00049``
#: (её NCC −0.018 при верном месте).
#:
#: Почему именно 0.35. Слабейшая верная поза — тот самый ``00049`` с 0.377;
#: сильнейшая ложная — 0.454, но у неё 6 инлайеров, и её отсекает второе условие
#: связки. Из ложных поз с ``инлайеры ≥ 8`` выше порога остаётся одна, ровно та
#: же, что проходила и по NCC, — то есть замена ничего не ослабляет и
#: возвращает ``00049``. **Запас тонкий (0.027)** и держится на одном кадре:
#: при пополнении набора порог пересчитать.
MIN_DINO = 0.35

#: Порог по имени меры — чтобы оркестрация не таскала литералы по уровням.
PHOTOMETRIC_THRESHOLDS = {"ncc": MIN_NCC, "dino": MIN_DINO}


def center_covariance(
    pts_q: np.ndarray,
    pts_r: np.ndarray,
    transform: SimilarityTransform,
    center_q: tuple[float, float],
    *,
    mpp: float = 1.0,
) -> np.ndarray | None:
    """Ковариация 2×2 положения центра кадра из невязок подобия.

    Распространение ошибки МНК: подобие ``кадр → подложка`` (4 параметра
    ``a, b, tx, ty``) оценено по инлайерам; дисперсия невязок даёт ковариацию
    параметров ``σ²·(AᵀA)⁻¹``, а якобиан отображения центра проецирует её на две
    координаты центра. Итог — в единицах ``(пиксель·mpp)²``: при ``mpp`` в метрах
    ковариация выходит в м².

    Дизайн-матрица ``A`` (2N×4) собрана из строк ``[qx, −qy, 1, 0]`` (для ``r_x``)
    и ``[qy, qx, 0, 1]`` (для ``r_y``) — это линеаризация модели подобия.

    Args:
        pts_q, pts_r: инлайерные соответствия (кадр и подложка), ``(N, 2)``.
        transform: оценённое подобие (по нему считаются невязки).
        center_q: центр кадра в его пикселях.
        mpp: разрешение подложки [м/пиксель] для перевода px² → м².

    Returns:
        Ковариация ``2×2`` либо ``None``, если данных мало (``N < 3``) или
        конфигурация точек вырождена (``AᵀA`` необратима).
    """
    pts_q = np.asarray(pts_q, dtype=float)
    pts_r = np.asarray(pts_r, dtype=float)
    n = len(pts_q)
    dof = 2 * n - 4  # число степеней свободы: 2N уравнений − 4 параметра
    if dof < 1:
        return None

    residuals = (pts_r - transform.apply(pts_q)).ravel()
    sigma2 = float(residuals @ residuals) / dof

    design = np.zeros((2 * n, 4))
    design[0::2] = np.column_stack([pts_q[:, 0], -pts_q[:, 1], np.ones(n), np.zeros(n)])
    design[1::2] = np.column_stack([pts_q[:, 1], pts_q[:, 0], np.zeros(n), np.ones(n)])
    try:
        params_cov = np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError:
        return None

    cx, cy = center_q
    jac_center = np.array([[cx, -cy, 1.0, 0.0], [cy, cx, 0.0, 1.0]])
    cov_px = sigma2 * jac_center @ params_cov @ jac_center.T
    return cov_px * (mpp * mpp)


def error_ellipse(cov: np.ndarray) -> tuple[float, float, float]:
    """Эллипс ошибки ``(semi_major, semi_minor, angle_deg)`` из ковариации, 1σ.

    Полуоси — корни собственных значений (СКО вдоль главных осей), угол — наклон
    большой оси к оси X в ``(−90, 90]`` градусов (это направление оси, без знака).
    """
    eigvals, eigvecs = np.linalg.eigh(np.asarray(cov, dtype=float))
    eigvals = np.clip(eigvals, 0.0, None)  # numerically negative → 0
    semi_minor, semi_major = math.sqrt(eigvals[0]), math.sqrt(eigvals[1])
    major_vec = eigvecs[:, 1]
    angle = math.degrees(math.atan2(major_vec[1], major_vec[0]))
    angle = (angle + 90.0) % 180.0 - 90.0
    return (semi_major, semi_minor, angle)


def align_reference(
    query_gray: np.ndarray, ref_gray: np.ndarray, transform: SimilarityTransform
) -> tuple[np.ndarray, np.ndarray]:
    """Подложка, отображённая в систему кадра, и маска валидности.

    Общая часть всех мер согласия: за пределами отображённой зоны данных нет, и
    эти пиксели не должны попадать в статистику — иначе два «пустых поля»
    прекрасно скоррелируют между собой.
    """
    h, w = query_gray.shape[:2]
    warped = cv2.warpAffine(
        ref_gray.astype(np.float32),
        transform.matrix.astype(np.float32),
        (w, h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float("nan"),
    )
    return warped, np.isfinite(warped)


def aligned_structural(
    query_gray: np.ndarray, ref_gray: np.ndarray, transform: SimilarityTransform,
    measure, *, min_valid: int = 256,
) -> float:
    """Мера согласия, которой нужны ДВЕ КАРТИНКИ, а не два вектора пикселей.

    NCC можно посчитать по разрозненным валидным пикселям, а вот всему, что
    смотрит на структуру — градиентам, дескрипторам, свёрточным фичам, — нужна
    связная картинка. Поэтому обе обрезаются по описанному прямоугольнику
    валидной зоны, а редкие дыры внутри неё заполняются средним: заполнение
    структуры не создаёт, а разрывов не оставляет.

    ``measure`` — любая функция ``(a, b) -> float`` из
    :mod:`aero_geoloc.similarity`. При нехватке валидной зоны возвращается
    ``-1.0`` — тот же признак «сравнивать нечего», что и у :func:`aligned_ncc`.
    """
    warped, valid = align_reference(query_gray, ref_gray, transform)
    if int(np.count_nonzero(valid)) < min_valid:
        return -1.0
    rows = np.flatnonzero(valid.any(axis=1))
    cols = np.flatnonzero(valid.any(axis=0))
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    if (y1 - y0) < 16 or (x1 - x0) < 16:
        return -1.0
    patch = warped[y0:y1, x0:x1].copy()
    holes = ~np.isfinite(patch)
    if holes.any():
        patch[holes] = float(np.nanmean(patch))
    return float(measure(query_gray[y0:y1, x0:x1].astype(np.float32), patch))


def aligned_ncc(
    query_gray: np.ndarray, ref_gray: np.ndarray, transform: SimilarityTransform
) -> float:
    """Нормированная кросс-корреляция кадра и подложки, выровненной под него.

    Фотометрический сигнал согласия после оценки/refinement: подложка
    отображается в систему кадра тем же преобразованием, и NCC меряется по зоне,
    куда реально попали пиксели подложки (вне её — не в счёт). Возвращает
    значение в ``[−1, 1]``; ``NaN`` превращается в ``−1`` (нет валидной зоны).
    """
    warped, valid = align_reference(query_gray, ref_gray, transform)
    if int(np.count_nonzero(valid)) < 16:
        return -1.0
    a = query_gray.astype(np.float32)[valid]
    b = warped[valid]
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return -1.0
    return float(a @ b / denom)


@dataclass(frozen=True)
class QualityAssessment:
    """Итог оценки качества одного решения.

    Attributes:
        status: ``LOCALIZED`` или ``LOW_CONFIDENCE`` (``NOT_LOCALIZED`` решается
            выше — гейтами приора и провалом матча/позы).
        confidence: фьюжн сигналов в ``[0, 1]`` (монотонный, не reliability-
            калиброванный до лестницы возмущений).
        covariance_m2: ковариация центра ``2×2``, м².
        error_ellipse_m: ``(semi_major, semi_minor, angle_deg)``, 1σ, метры.
        signals: сырьё для калибровки и разбора (инлайеры, RMSE, NCC, размер эллипса).
    """

    status: Status
    confidence: float
    covariance_m2: np.ndarray
    error_ellipse_m: tuple[float, float, float]
    signals: dict


def _geometric_mean(values: list[float]) -> float:
    """Среднее геометрическое в ``[0, 1]``: любой слабый сигнал тянет итог вниз."""
    clipped = np.clip(np.asarray(values, dtype=float), 1e-6, 1.0)
    return float(np.exp(np.mean(np.log(clipped))))


def assess(
    pose: PoseEstimate,
    corr: Correspondences,
    center_q: tuple[float, float],
    mpp: float,
    *,
    photometric: float | None = None,
    photometric_kind: str = "ncc",
    systematic_floor_m: float = 0.0,
    max_semi_major_m: float = 3.0,
    min_inliers_hard: int = MIN_INLIERS_HARD,
    min_photometric: float | None = None,
    inlier_saturation: int = 30,
) -> QualityAssessment:
    """Свести геометрию решения и сигналы в ковариацию, доверие и статус.

    Статус ``LOCALIZED`` требует **связки** трёх калиброванных условий (все И):
    малый 1σ-эллипс (``≤ max_semi_major_m``), достаточно инлайеров
    (``≥ min_inliers_hard``) И (если дан) фотометрический NCC не ниже порога
    (``≥ min_photometric``). Иначе — ``LOW_CONFIDENCE``. Связка, а не один счётчик: калибровка
    на реальных дрон↔Esri матчах (см. JOURNAL, веха калибровки) показала, что ни
    инлайеры, ни NCC по отдельности не делят верный слабый матч и ложный случайный —
    негативы упираются в 4–9 инлайеров и дают шумный NCC (иногда высокий на
    повторяющемся паттерне), а связка ``инлайеры ≥ 8 И NCC ≥ 0.12`` даёт ~96% точности.
    Репроекционный RMSE калибровка отвергла как неразделяющий (Youden 0) — он
    остаётся в ``signals`` для диагностики, но **не входит в решение**.

    Замечание для поиска по карте: ``LOW_CONFIDENCE`` — это привязка, которой нельзя
    доверять; там, где ложная точка опаснее отказа, принимать только ``LOCALIZED``.

    Args:
        pose: оценка позы с маской инлайеров.
        corr: соответствия (берутся инлайерные).
        center_q: центр кадра в его пикселях.
        mpp: разрешение подложки, м/пиксель.
        photometric: значение меры согласия кадра и выровненной подложки, опц.
            Какая именно это мера — решает оркестрация; связка знает только, что
            больше = лучше, и сравнивает с порогом.
        photometric_kind: имя меры (``ncc`` | ``dino``) — попадает в ``signals``
            и задаёт дефолтный порог. Имя обязательно: число 0.31 значит
            «отлично» для NCC и «на грани» для dino, и без имени такую таблицу
            невозможно прочитать полгода спустя.
        systematic_floor_m: изотропный пол ковариации — абсолютная погрешность
            геопривязки подложки (не зависит от ``N``). Прибавляется как
            ``floor²·I``. На синтетике 0; в бою — паспортная точность подложки.
        max_semi_major_m: порог большой полуоси 1σ-эллипса для ``LOCALIZED``.
        min_inliers_hard: минимум инлайеров в связке (калибр.: 8).
        min_photometric: порог меры в связке. ``None`` — взять калиброванный
            дефолт по ``photometric_kind`` (:data:`PHOTOMETRIC_THRESHOLDS`).
        inlier_saturation: число инлайеров, при котором соответствующий сигнал
            доверия насыщается до 1.
    """
    inlier = pose.inlier_mask
    pts_q, pts_r = corr.pts_q[inlier], corr.pts_r[inlier]
    cov = center_covariance(pts_q, pts_r, pose.transform, center_q, mpp=mpp)
    if cov is None:
        cov = np.full((2, 2), np.inf)
    elif systematic_floor_m > 0.0:
        cov = cov + (systematic_floor_m**2) * np.eye(2)
    ellipse = error_ellipse(cov)
    semi_major_m = ellipse[0]

    n_inliers = pose.n_inliers
    sub_scores = [min(n_inliers / inlier_saturation, 1.0), pose.inlier_ratio]
    if photometric is not None:
        sub_scores.append(max(0.0, photometric))
    confidence = _geometric_mean(sub_scores)

    threshold = (PHOTOMETRIC_THRESHOLDS.get(photometric_kind, MIN_NCC)
                 if min_photometric is None else min_photometric)
    signals = {
        "n_inliers": n_inliers,
        "inlier_ratio": pose.inlier_ratio,
        "reprojection_rmse_px": pose.reprojection_rmse_px,
        "semi_major_m": semi_major_m,
        "semi_minor_m": ellipse[1],
        "photometric": photometric,
        "photometric_kind": photometric_kind,
        "photometric_threshold": threshold,
    }
    # Свидетельства уровня пары от матчера (у плотных ядер — сводка поля
    # уверенности). Кладутся в диагностику всегда, в решение — только то, что
    # откалибровано: инвариант «доверие не подделываем до проверенного покрытия».
    signals.update(corr.evidence)

    # Калиброванная СВЯЗКА (все И): эллипс мал, инлайеров не мало, NCC не провален.
    # RMSE намеренно не участвует — калибровка признала его неразделяющим.
    photometric_ok = photometric is None or photometric >= threshold
    confident = (
        math.isfinite(semi_major_m)
        and semi_major_m <= max_semi_major_m
        and n_inliers >= min_inliers_hard
        and photometric_ok
    )
    status = Status.LOCALIZED if confident else Status.LOW_CONFIDENCE
    return QualityAssessment(
        status=status,
        confidence=confidence,
        covariance_m2=cov,
        error_ellipse_m=ellipse,
        signals=signals,
    )
