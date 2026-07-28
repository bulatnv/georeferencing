"""Оркестрация пайплайна: нормализация → match → pose → считывание координат.

Фаза 1 — минимальный сквозной скелет из ``docs/PLAN.md``: подложка **уже дана**
готовым георефренцированным растром (никакой сети), приор узкий, retrieval и
coarse-to-fine не строятся, refinement и ковариация — тоже нет.

Смысл фазы в том, чтобы доказать сквозную геометрию: если на кропе из той же
подложки центр не восстанавливается с субпиксельной точностью — баг сидит в
геометрии или в pose, и его надо чинить до того, как усложнять ядро матчинга.

В фазе 2 поверх этой функции появится ``localize()``, которая сама тянет
подложку из сети по приору и добавляет coarse-to-fine, refinement и quality.
"""

from __future__ import annotations

import math

import numpy as np

from .basemap import BasemapSource, MissingTileError
from .camera import Camera
from .geo import Georef, ground_mpp, haversine_m, zoom_for_mpp
from .matcher import Correspondences, Matcher, SIFTMatcher
from .pose import PoseEstimate, estimate_similarity, refine_ecc
from .quality import MIN_NCC as quality_min_ncc
from .quality import aligned_ncc, assess
from .retrieval import TerrainIndex, should_localize
from .types import LocalizationResult, Prior, Status

__all__ = ["normalize_gray", "localize_against_reference", "localize"]

#: Во сколько σ приора укладывается допустимое отклонение центра (инвариант 3).
PRIOR_GATE_SIGMA = 3.0

#: Порог NCC в связке качества — переэкспорт калиброванной константы из
#: :mod:`aero_geoloc.quality`, чтобы дефолт оркестрации не мог разойтись с тем,
#: по которому реально решает ``assess``. Обоснование значения — там же.
MIN_NCC = quality_min_ncc


def normalize_gray(image: np.ndarray, *, clahe: bool = False) -> np.ndarray:
    """Привести изображение к grayscale uint8 — стадия 1 из ``docs/PIPELINE.md``.

    Args:
        clahe: выравнивание освещения. На синтетике почти не нужно и по
            умолчанию выключено; для борта (тени, сезон, экспозиция) критично —
            это заготовленный хвост под фазу 4.
    """
    if image is None or image.size == 0:
        raise ValueError("пустое изображение")

    import cv2  # локальный импорт: geo/camera/types остаются свободными от OpenCV

    gray = image
    if gray.ndim == 3:
        channels = gray.shape[2]
        if channels == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        elif channels == 4:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError(f"не умею приводить к grayscale изображение с {channels} каналами")
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if clahe:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return gray


def _match_prerotated(matcher: Matcher, query_gray, ref_gray, prerotate_deg: float):
    """Матч с предповоротом кадра на ``prerotate_deg``, но соответствия — в ИСХОДНЫХ пикселях кадра.

    Обучаемые матчеры (LightGlue/LoFTR) не инвариантны к повороту, а бортовой
    кадр развёрнут на yaw относительно севера. Стратегия «доверяем yaw» из
    ``docs/PIPELINE.md``: повернуть кадр к северу (``prerotate_deg = −yaw``),
    сматчить, а точки со стороны кадра **вернуть в исходные координаты** обратным
    поворотом. Тогда поза считается сразу в системе исходного кадра (поворот ≈
    yaw), и всё выше по конвейеру, включая гейт поворота, работает без изменений.
    Для инвариантного к повороту SIFT ``prerotate_deg = 0`` — обычный матч.
    """
    if prerotate_deg == 0.0:
        return matcher.match(query_gray, ref_gray)
    import cv2

    h, w = query_gray.shape[:2]
    rot = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), prerotate_deg, 1.0)
    rotated = cv2.warpAffine(
        query_gray, rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    corr = matcher.match(rotated, ref_gray)
    inv = cv2.invertAffineTransform(rot)
    pts_q = (corr.pts_q @ inv[:, :2].T + inv[:, 2]).astype(np.float32)
    return Correspondences(pts_q, corr.pts_r, corr.conf)


def _read_result(
    pose: PoseEstimate,
    camera: Camera,
    reference_georef: Georef,
    mpp: float,
) -> tuple[float, float, float, float, list[tuple[float, float]]]:
    """Стадия 6: из преобразования и Georef — центр, курс, высота, отпечаток.

    Все переводы «пиксель ↔ координата» идут через Georef (инвариант 5).
    """
    cx, cy = ((camera.image_width - 1) / 2.0, (camera.image_height - 1) / 2.0)
    center_ref = pose.transform.apply(np.array([cx, cy]))
    center_lon, center_lat = reference_georef.pixel_to_lonlat(
        float(center_ref[0]), float(center_ref[1])
    )

    corners_q = np.array(
        [
            [0.0, 0.0],
            [camera.image_width - 1.0, 0.0],
            [camera.image_width - 1.0, camera.image_height - 1.0],
            [0.0, camera.image_height - 1.0],
        ]
    )
    corners_ref = pose.transform.apply(corners_q)
    lons, lats = reference_georef.pixel_to_lonlat(corners_ref[:, 0], corners_ref[:, 1])
    footprint = [(float(lon), float(lat)) for lon, lat in zip(lons, lats)]

    # Масштаб преобразования → GSD → высота: 1 пиксель кадра = s пикселей
    # подложки = s · mpp метров, а GSD = H / f_px.
    heading_deg = pose.transform.rotation_deg % 360.0
    altitude_est_m = camera.altitude_for_gsd(pose.transform.scale * mpp)

    return float(center_lat), float(center_lon), heading_deg, altitude_est_m, footprint


def localize_against_reference(
    image: np.ndarray,
    camera: Camera,
    prior: Prior,
    reference: np.ndarray,
    reference_georef: Georef,
    *,
    matcher: Matcher | None = None,
    ransac_threshold_px: float = 3.0,
    min_inliers: int = 10,
    trust_yaw: bool = True,
    rotation_tolerance_deg: float = 15.0,
    clahe: bool = False,
    refine: bool = False,
    prerotate_deg: float = 0.0,
    min_ncc: float = MIN_NCC,
) -> LocalizationResult:
    """Локализовать кадр по заранее данному георефренцированному растру подложки.

    Args:
        image: кадр (color или grayscale).
        camera: параметры камеры кадра.
        prior: грубое приближение; работает **как ограничение**, а не подсказка.
        reference: растр подложки (color или grayscale).
        reference_georef: привязка этого растра.
        matcher: сменное ядро; по умолчанию :class:`~aero_geoloc.matcher.SIFTMatcher`.
        trust_yaw: доверять ли ``prior.yaw_deg`` как ограничению на поворот.
            Если ``False``, поворот восстанавливается свободно, а yaw идёт
            только в диагностику (две стратегии из ``docs/PIPELINE.md``).
        rotation_tolerance_deg: допуск на отклонение от yaw при ``trust_yaw``.
        clahe: выравнивание освещения на входе.
        refine: субпиксельный ECC-refinement поверх RANSAC-модели (стадия 5).
            По умолчанию выключен: в фазе 1 его не было, и включение не должно
            менять её поведение молча. Успех/отказ refinement — в диагностике
            под ключом ``refined_ecc``.

    Returns:
        :class:`~aero_geoloc.types.LocalizationResult`. Статус, ``confidence``,
        ковариация центра и эллипс ошибки приходят из :mod:`aero_geoloc.quality`
        (стадия 7): ``LOCALIZED`` при малом эллипсе, иначе ``LOW_CONFIDENCE``;
        ``NOT_LOCALIZED`` — это провал матча/позы или выход за диск приора.
    """
    matcher = matcher if matcher is not None else SIFTMatcher()

    query_gray = normalize_gray(image, clahe=clahe)
    ref_gray = normalize_gray(reference, clahe=clahe)
    if query_gray.shape[:2] != (camera.image_height, camera.image_width):
        raise ValueError(
            f"размер кадра {query_gray.shape[1]}×{query_gray.shape[0]} не совпадает "
            f"с камерой {camera.image_width}×{camera.image_height}"
        )
    if ref_gray.shape[:2] != (reference_georef.height, reference_georef.width):
        raise ValueError(
            f"размер подложки {ref_gray.shape[1]}×{ref_gray.shape[0]} не совпадает "
            f"с Georef {reference_georef.width}×{reference_georef.height}"
        )

    # Стадия 0: что известно до матчинга. Ожидаемый масштаб — во сколько раз
    # пиксель кадра крупнее пикселя подложки; допуск на него даёт приор высоты.
    mpp = reference_georef.mpp
    expected_scale = camera.gsd(prior.altitude_m) / mpp
    scale_lo, scale_hi = prior.scale_bounds
    scale_bounds = (expected_scale * scale_lo, expected_scale * scale_hi)

    base_diagnostics = {
        "expected_scale": expected_scale,
        "scale_bounds": scale_bounds,
        "reference_mpp": mpp,
        "matcher": type(matcher).__name__,
        "confidence_calibrated": False,
    }

    # Стадия 3: матчинг.
    corr = _match_prerotated(matcher, query_gray, ref_gray, prerotate_deg)
    if len(corr) < 3:
        return LocalizationResult.failed(
            "слишком мало соответствий", n_correspondences=len(corr), **base_diagnostics
        )

    # Стадия 4: робастная оценка позы с приорами-ограничениями.
    pose = estimate_similarity(
        corr,
        ransac_threshold_px=ransac_threshold_px,
        min_inliers=min_inliers,
        scale_bounds=scale_bounds,
        expected_rotation_deg=prior.yaw_deg if trust_yaw else None,
        rotation_tolerance_deg=rotation_tolerance_deg,
    )
    if pose is None:
        return LocalizationResult.failed(
            "нет устойчивой модели подобия", n_correspondences=len(corr), **base_diagnostics
        )

    # Стадия 5: субпиксельный refinement поверх робастной модели. Считывание
    # координат ниже пойдёт уже по уточнённой позе, поэтому refinement здесь, до
    # чтения центра и до гейта по приору.
    refined_ecc = False
    if refine:
        refined_transform = refine_ecc(query_gray, ref_gray, pose.transform)
        if refined_transform is not None:
            pose = pose.with_transform(refined_transform, corr)
            refined_ecc = True

    diagnostics = {**base_diagnostics, **pose.diagnostics(), "refined_ecc": refined_ecc}

    # Стадия 6: считывание результата через Georef.
    center_lat, center_lon, heading_deg, altitude_est_m, footprint = _read_result(
        pose, camera, reference_georef, mpp
    )

    # Приор позиции как ограничение: решение вне диска ±3σ отбрасываем целиком.
    prior_offset_m = float(haversine_m(prior.lat, prior.lon, center_lat, center_lon))
    diagnostics["prior_offset_m"] = prior_offset_m
    diagnostics["prior_offset_sigma"] = prior_offset_m / prior.sigma_m
    if prior_offset_m > PRIOR_GATE_SIGMA * prior.sigma_m:
        return LocalizationResult.failed("решение вне диска приора", **diagnostics)

    # Стадия 7: качество — ковариация центра, эллипс, статус LOCALIZED/LOW_CONFIDENCE.
    ncc = aligned_ncc(query_gray, ref_gray, pose.transform)
    quality = assess(pose, corr, camera.principal_point(), mpp, photometric_ncc=ncc, min_ncc=min_ncc)
    diagnostics.update(quality.signals)
    diagnostics["confidence_calibrated"] = True  # эллипс выведен строго (см. quality.py)

    return LocalizationResult(
        status=quality.status,
        center_lat=center_lat,
        center_lon=center_lon,
        heading_deg=heading_deg,
        altitude_est_m=altitude_est_m,
        footprint_lonlat=footprint,
        transform=pose.transform.matrix,
        error_ellipse_m=quality.error_ellipse_m,
        covariance_m2=quality.covariance_m2,
        confidence=quality.confidence,
        diagnostics=diagnostics,
    )


def _even(n: float) -> int:
    """Ближайший сверху чётный размер растра — удобно для ресемпла и центра."""
    return int(math.ceil(n / 2.0) * 2)


def _coarse_center(
    image_gray: np.ndarray,
    camera: Camera,
    prior: Prior,
    coarse_ref_gray: np.ndarray,
    coarse_georef: Georef,
    *,
    matcher: Matcher,
    trust_yaw: bool,
    rotation_tolerance_deg: float,
    ransac_threshold_px: float,
    min_inliers: int,
    prerotate_deg: float = 0.0,
) -> tuple[float, float] | None:
    """Грубый уровень (стадия 2): «ГДЕ примерно» — центр кадра в lon/lat.

    Кадр даунсемплится до разрешения грубой подложки (``mpp ≈ mpp``), поэтому
    остаточный масштаб ≈ 1 (плюс погрешность высоты). Точность здесь не нужна —
    нужен кандидат, который точный уровень уточнит до субпикселя.

    В фазе 2 retrieval ещё нет, и грубый уровень — это один матч по окну вокруг
    приора (top-1). Логика top-K и сигнала уникальности появится с ``retrieval.py``
    (фаза 3); интерфейс оркестрации при этом не изменится.
    """
    gsd = camera.gsd(prior.altitude_m)
    q_scale = gsd / coarse_georef.mpp  # ≤ 1: кадр детальнее грубой подложки
    wq = max(16, int(round(camera.image_width * q_scale)))
    hq = max(16, int(round(camera.image_height * q_scale)))

    import cv2

    query_coarse = cv2.resize(image_gray, (wq, hq), interpolation=cv2.INTER_AREA)

    corr = _match_prerotated(matcher, query_coarse, coarse_ref_gray, prerotate_deg)
    if len(corr) < 3:
        return None
    # Масштаб грубого уровня ≈ 1; допуск на него — тот же приор высоты.
    scale_lo, scale_hi = prior.scale_bounds
    pose = estimate_similarity(
        corr,
        ransac_threshold_px=ransac_threshold_px,
        min_inliers=min_inliers,
        scale_bounds=(scale_lo, scale_hi),
        expected_rotation_deg=prior.yaw_deg if trust_yaw else None,
        rotation_tolerance_deg=rotation_tolerance_deg,
    )
    if pose is None:
        return None

    center_q = np.array([(wq - 1) / 2.0, (hq - 1) / 2.0])
    center_ref = pose.transform.apply(center_q)
    lon, lat = coarse_georef.pixel_to_lonlat(float(center_ref[0]), float(center_ref[1]))
    return float(lon), float(lat)


def _fine_pass(
    image: np.ndarray,
    camera: Camera,
    prior: Prior,
    cand_lon: float,
    cand_lat: float,
    altitude_m: float,
    z_fine: int,
    coarse_mpp: float,
    basemap: BasemapSource,
    *,
    matcher: Matcher,
    refine: bool,
    trust_yaw: bool,
    rotation_tolerance_deg: float,
    ransac_threshold_px: float,
    min_inliers: int,
    clahe: bool,
    prerotate_deg: float = 0.0,
    min_ncc: float,
) -> LocalizationResult:
    """Точный уровень (стадии 3–6): окно нативного разрешения вокруг кандидата.

    Зум ``z_fine`` и высота ``altitude_m`` — то, что цикл переуточнения масштаба
    может менять между итерациями; размер окна зависит от footprint на этой
    высоте. Всё остальное делегируется :func:`localize_against_reference`, которой
    и принадлежит вся геометрия точного уровня.

    В точный уровень отдаётся ИСХОДНЫЙ кадр (не предварительно нормализованный):
    нормализацию — включая CLAHE — делает ``localize_against_reference`` одинаково
    для кадра и подложки. Иначе при ``clahe=True`` кадр выравнивался бы дважды.
    """
    footprint_max = max(camera.footprint_m(altitude_m))
    # Запас вокруг кандидата покрывает неопределённость грубого уровня (единицы
    # его пикселей); нижняя граница — доля footprint, чтобы окно не выродилось.
    fine_margin_m = max(6.0 * coarse_mpp, 0.15 * footprint_max)
    # ...но сверху запас ограничен: у обученного матчера бюджет ключевых точек
    # фиксирован (SuperPoint ~2048), и на окне много больше кадра их плотность
    # падает так, что матч разваливается. Измерено на Саратове (кадр 2783 px,
    # отпечаток 517 м): окно 1.2-1.4× отпечатка → 21-31 инлайер, 1.8× → 5, 2.15× → 5.
    # 0.25·footprint держит окно ≤1.5× кадра. Позиционную неопределённость крупной
    # retrieval-клетки это не теряет: клетки идут с перекрытием 50%, поэтому у
    # истинного центра всегда есть клетка не дальше четверти клетки, и её
    # перебирает мультигипотезный цикл по кандидатам.
    fine_margin_m = min(fine_margin_m, 0.25 * footprint_max)
    fine_size = _even(2.0 * (0.5 * footprint_max + fine_margin_m) / ground_mpp(cand_lat, z_fine))
    fine_ref, fine_georef = basemap(cand_lon, cand_lat, z_fine, fine_size, fine_size)

    prior_fine = Prior(
        lat=cand_lat,
        lon=cand_lon,
        sigma_m=fine_margin_m,
        altitude_m=altitude_m,
        altitude_sigma_m=prior.altitude_sigma_m,
        yaw_deg=prior.yaw_deg,
        pitch_deg=prior.pitch_deg,
        roll_deg=prior.roll_deg,
    )
    return localize_against_reference(
        image,
        camera,
        prior_fine,
        fine_ref,
        fine_georef,
        matcher=matcher,
        ransac_threshold_px=ransac_threshold_px,
        min_inliers=min_inliers,
        trust_yaw=trust_yaw,
        rotation_tolerance_deg=rotation_tolerance_deg,
        clahe=clahe,
        refine=refine,
        prerotate_deg=prerotate_deg,
        min_ncc=min_ncc,
    )


def _fine_with_scale_loop(
    image: np.ndarray,
    camera: Camera,
    prior: Prior,
    cand_lon: float,
    cand_lat: float,
    coarse_mpp: float,
    z_fine: int,
    basemap: BasemapSource,
    *,
    scale_iters: int,
    matcher: Matcher,
    refine: bool,
    trust_yaw: bool,
    rotation_tolerance_deg: float,
    ransac_threshold_px: float,
    min_inliers: int,
    clahe: bool,
    max_zoom: int = 22,
    prerotate_deg: float = 0.0,
    min_ncc: float,
) -> LocalizationResult:
    """Точный уровень вокруг одного кандидата + цикл переуточнения масштаба (стадия 5).

    Восстановленный масштаб даёт оценку высоты; если она указывает на ДРУГОЙ зум
    подложки, растр перетягивается на нём и точный уровень переигрывается уже при
    ``mpp ≈ GSD`` (масштаб ≈ 1). Триггер — смена зума, а не ``|s−1|``: даже при
    верной высоте ``s`` гуляет в пределах кванта зума. Множество посещённых зумов
    гарантирует завершение, а лучший (по инлайерам) проход сохраняется — если
    проход на новом зуме не сойдётся, остаётся предыдущий.
    """

    def fine(altitude_m: float, zoom: int) -> LocalizationResult:
        return _fine_pass(
            image, camera, prior, cand_lon, cand_lat, altitude_m, zoom, coarse_mpp, basemap,
            matcher=matcher, refine=refine, trust_yaw=trust_yaw,
            rotation_tolerance_deg=rotation_tolerance_deg,
            ransac_threshold_px=ransac_threshold_px, min_inliers=min_inliers, clahe=clahe,
            prerotate_deg=prerotate_deg, min_ncc=min_ncc,
        )

    def n_inliers(result: LocalizationResult) -> int:
        return int(result.diagnostics.get("n_inliers", 0))

    best = fine(prior.altitude_m, z_fine)
    z_best, visited, scale_iters_done = z_fine, {z_fine}, 0
    while best.is_localized and scale_iters_done < scale_iters:
        z_next = zoom_for_mpp(camera.gsd(best.altitude_est_m), prior.lat, max_zoom=max_zoom)
        if z_next in visited:
            break
        visited.add(z_next)
        scale_iters_done += 1
        try:
            retried = fine(best.altitude_est_m, z_next)
        except (ValueError, MissingTileError):
            break  # источник не отдаёт этот зум (край данных/детальнее подложки)
        if not (retried.is_localized and n_inliers(retried) > n_inliers(best)):
            break
        best, z_best = retried, z_next

    best.diagnostics["z_fine"] = z_best
    best.diagnostics["scale_iters_done"] = scale_iters_done
    return best


def _retrieval_candidates(
    index: TerrainIndex,
    image_gray: np.ndarray,
    prior: Prior,
    *,
    top_k: int,
    trust_yaw: bool,
    min_uniqueness: float,
    gate_sigma: float,
) -> tuple[list[tuple[float, float, float, float]], dict]:
    """Кандидаты грубого уровня из retrieval-индекса (Этаж 1).

    Возвращает список ``(lon, lat, coarse_mpp, cell_rotation_deg)`` внутри диска
    приора и диагностику. Пустой список — либо низкая уникальность (самоподобие),
    либо все клетки вне диска; причина в диагностике под ключом ``retrieval_reason``.

    **Курс из ротации клетки.** Когда yaw неизвестен (``trust_yaw=False``), индекс
    строят с ротационной аугментацией, и совпавшая клетка несёт угол, под которым
    она совпала: ``rotate(клетка, R) ≈ кадр`` ⟹ курс кадра ≈ ``R``, а выровнять
    кадр к северной подложке можно предповоротом на ``−R``. Угол отдаётся наружу
    (решение о предповороте принимает :func:`localize`): без него аугментация
    чинила бы только Этаж 1, а Этаж 2 всё равно получал бы кадр под произвольным
    углом — и не-инвариантный к повороту матчер (LightGlue/LoFTR) на нём падает.
    """
    prerotate = -prior.yaw_deg if trust_yaw else 0.0
    result = index.query(image_gray, k=top_k, prerotate_deg=prerotate)
    diag = {
        "retrieval_uniqueness": result.uniqueness,
        "retrieval_returned": len(result.cells),
    }
    if not should_localize(result, min_uniqueness=min_uniqueness):
        diag["retrieval_reason"] = "низкая уникальность (самоподобие)"
        return [], diag

    candidates = []
    for cell in result.cells:
        offset_m = float(haversine_m(prior.lat, prior.lon, cell.center_lat, cell.center_lon))
        if offset_m <= gate_sigma * prior.sigma_m:
            # coarse_mpp кандидата задаёт запас точного окна: 6·coarse_mpp = полклетки
            # (истинный центр может лежать где угодно в клетке retrieval).
            cell_footprint_m = cell.size_px * ground_mpp(cell.center_lat, cell.zoom)
            candidates.append(
                (cell.center_lon, cell.center_lat, cell_footprint_m / 12.0, cell.rotation_deg)
            )
    if not candidates:
        diag["retrieval_reason"] = "все клетки вне диска приора"
    return candidates, diag


def localize(
    image: np.ndarray,
    camera: Camera,
    prior: Prior,
    basemap: BasemapSource,
    *,
    matcher: Matcher | None = None,
    refine: bool = True,
    trust_yaw: bool = True,
    rotation_tolerance_deg: float = 15.0,
    ransac_threshold_px: float = 3.0,
    min_inliers: int = 10,
    clahe: bool = False,
    coarse_target_px: int = 2048,
    coarse_min_inliers: int = 6,
    scale_iters: int = 2,
    index: TerrainIndex | None = None,
    retrieval_top_k: int = 5,
    min_uniqueness: float = 0.0,
    max_zoom: int = 22,
    prerotate: bool = False,
    min_ncc: float = MIN_NCC,
    gate_sigma: float = PRIOR_GATE_SIGMA,
) -> LocalizationResult:
    """Полная одиночная локализация: подложка из источника + coarse-to-fine.

    Оркестрация фазы 2 (``docs/PIPELINE.md``): сама тянет подложку по приору
    через сменный :class:`~aero_geoloc.basemap.BasemapSource`, приводит масштабы,
    грубым уровнем находит кандидата, точным — уточняет позу с субпиксельным
    refinement. В отличие от :func:`localize_against_reference`, которой растр
    отдают готовым, здесь широкий приор ±σ и разрешение подложки выбираются сами.

    Args:
        image: кадр (color или grayscale).
        camera: параметры камеры.
        prior: приближение и **ограничение** (позиция, высота, курс).
        basemap: источник окон подложки (сеть — :class:`TileBasemap`,
            стенд — :class:`~aero_geoloc.testbench.SceneBasemap`).
        refine: субпиксельный ECC на точном уровне (по умолчанию включён).
        coarse_target_px: целевой размер грубого растра — от него выбирается зум
            грубого уровня, чтобы весь диск ±σ уместился примерно в этот размер.
        coarse_min_inliers: минимум инлайеров на грубом уровне. Ниже, чем на
            точном: грубый уровень лишь ищет кандидата, а его добросовестность
            перепроверяет точный уровень (``min_inliers``) и финальный гейт по
            исходному приору — ложняк так не проходит, а бедные текстурой места
            не отсекаются раньше времени.
        scale_iters: сколько раз переуточнять масштаб (стадия 5). Восстановленный
            масштаб даёт оценку высоты; если она меняет зум подложки, растр
            перетягивается на нём (``mpp`` снова ≈ ``GSD``) и точный уровень
            переигрывается при масштабе ≈ 1. ``0`` отключает цикл. Обычно хватает
            одной итерации; цикл сходится, как только зум перестаёт меняться.
        index: предпостроенный :class:`~aero_geoloc.retrieval.TerrainIndex`. Если
            задан — грубый уровень идёт **retrieval-запросом** (Этаж 1) вместо
            перебора окна вокруг приора: индекс отдаёт top-K клеток, точный
            уровень пробегает их и берёт лучшую по инлайерам (мультигипотезность
            против самоподобия). Если ``None`` — прежний матч по окну.
        retrieval_top_k: сколько кандидатных клеток брать у индекса.
        min_uniqueness: порог сигнала уникальности retrieval; ниже него —
            честный отказ по самоподобию ещё до матчинга (см.
            :func:`~aero_geoloc.retrieval.calibrate_uniqueness_threshold`).
        min_ncc: порог фотометрического NCC в связке качества (``quality.py``).
            Дефолт :data:`MIN_NCC` (0.12) откалиброван на реальных дрон↔Esri
            матчах и совпадает с дефолтом ``assess``. NCC — **равноправное
            условие связки** ``инлайеры ≥ 8 И NCC ≥ 0.12``, а не необязательная
            добавка: калибровка показала, что по отдельности ни инлайеры, ни NCC
            не делят верный слабый матч и ложный (JOURNAL, веха калибровки).
            Поднимать имеет смысл на same-domain, где NCC верных матчей близок к
            единице; для кросс-домена он низок даже у верных (0.04–0.45), и
            прежний порог 0.3 ложно уводил верные локализации в ``LOW_CONFIDENCE``.
        gate_sigma: во сколько σ приора укладывается допустимое отклонение центра.

    Returns:
        :class:`~aero_geoloc.types.LocalizationResult` со статусом, ковариацией и
        эллипсом из :mod:`aero_geoloc.quality`.
    """
    matcher = matcher if matcher is not None else SIFTMatcher()
    # Предповорот кадра к северу для не-инвариантных к повороту матчеров
    # (LightGlue/LoFTR): при доверии yaw поворачиваем на −yaw (стратегия из
    # PIPELINE.md). Для SIFT не нужен и по умолчанию выключен.
    prerotate_deg = -prior.yaw_deg if (prerotate and trust_yaw) else 0.0
    image_gray = normalize_gray(image, clahe=clahe)
    if image_gray.shape[:2] != (camera.image_height, camera.image_width):
        raise ValueError(
            f"размер кадра {image_gray.shape[1]}×{image_gray.shape[0]} не совпадает "
            f"с камерой {camera.image_width}×{camera.image_height}"
        )

    # Стадия 0/1: приведение масштабов. Зум точного уровня — ближайший к GSD.
    # «Ближайший», а не «детальнее»: у границы зума требование mpp ≤ GSD
    # перепрыгивало бы на целый уровень выше из-за долей процента разницы (а на
    # краю данных подложки такого уровня может и не быть). Какой из двух
    # соседних зумов реально лучше для матчинга, решает цикл ниже по числу
    # инлайеров, а не априорное правило.
    gsd = camera.gsd(prior.altitude_m)
    # Зум клампится к максимуму провайдера: на низкой высоте кадр в разы детальнее
    # подложки, и «идеальный» зум точного уровня может превысить то, что подложка
    # вообще отдаёт (Esri — z19). Тогда кадр даунсемплится до z_fine (ограничение 4
    # из README): совпадут только те частоты, что есть в подложке.
    z_fine = zoom_for_mpp(gsd, prior.lat, max_zoom=max_zoom)
    footprint_max = max(camera.footprint_m(prior.altitude_m))

    # Грубый уровень: зум так, чтобы весь диск ±gate_sigma·σ уместился в ~coarse_target_px.
    search_half_m = 0.5 * footprint_max + gate_sigma * prior.sigma_m
    coarse_mpp_target = 2.0 * search_half_m / coarse_target_px
    z_coarse = min(z_fine, zoom_for_mpp(coarse_mpp_target, prior.lat, mode="coarser", max_zoom=max_zoom))
    coarse_mpp = ground_mpp(prior.lat, z_coarse)
    coarse_size = _even(2.0 * search_half_m / coarse_mpp)

    base_diagnostics = {
        "z_coarse": z_coarse,
        "z_fine": z_fine,
        "coarse_size": coarse_size,
        "gsd": gsd,
        "matcher": type(matcher).__name__,
        "confidence_calibrated": False,
    }

    base_diagnostics["retrieval"] = index is not None

    # Грубый уровень «ГДЕ примерно»: retrieval-запрос к индексу (Этаж 1) либо
    # прежний матч по окну вокруг приора. На выходе — кандидаты (lon, lat, coarse_mpp).
    if index is not None:
        candidates, retrieval_diag = _retrieval_candidates(
            index, image_gray, prior,
            top_k=retrieval_top_k, trust_yaw=trust_yaw,
            min_uniqueness=min_uniqueness, gate_sigma=gate_sigma,
        )
        base_diagnostics.update(retrieval_diag)
        if not candidates:
            return LocalizationResult.failed(
                retrieval_diag.get("retrieval_reason", "retrieval не дал кандидатов"),
                **base_diagnostics,
            )
    else:
        coarse_ref, coarse_georef = basemap(prior.lon, prior.lat, z_coarse, coarse_size, coarse_size)
        coarse_gray = normalize_gray(coarse_ref, clahe=clahe)
        candidate = _coarse_center(
            image_gray, camera, prior, coarse_gray, coarse_georef,
            matcher=matcher, trust_yaw=trust_yaw, rotation_tolerance_deg=rotation_tolerance_deg,
            ransac_threshold_px=ransac_threshold_px, min_inliers=coarse_min_inliers,
            prerotate_deg=prerotate_deg,
        )
        if candidate is None:
            return LocalizationResult.failed("грубый уровень не нашёл кандидата", **base_diagnostics)
        cand_lon, cand_lat = candidate
        coarse_offset_m = float(haversine_m(prior.lat, prior.lon, cand_lat, cand_lon))
        base_diagnostics["coarse_offset_m"] = coarse_offset_m
        if coarse_offset_m > gate_sigma * prior.sigma_m:  # инвариант 3
            return LocalizationResult.failed("кандидат вне диска приора", **base_diagnostics)
        candidates = [(cand_lon, cand_lat, coarse_mpp, 0.0)]

    # Точный уровень + цикл масштаба по каждому кандидату; берём лучший по инлайерам
    # (мультигипотезность ловит неоднозначность на самоподобной местности).
    best: LocalizationResult | None = None
    for cand_lon, cand_lat, cand_coarse_mpp, cell_rotation_deg in candidates:
        # Чем выравнивать кадр под не-инвариантный матчер:
        #   курс известен  → общим −yaw приора (``prerotate_deg``);
        #   курс неизвестен → углом совпавшей клетки индекса (−R), иначе Этаж 2
        #     получит кадр под произвольным поворотом и провалится.
        # Флаг ``prerotate`` остаётся хозяином решения: для SIFT (инвариантен)
        # предповорот не нужен и не делается.
        cand_prerotate = (
            prerotate_deg if trust_yaw else (-cell_rotation_deg if prerotate else 0.0)
        )
        r = _fine_with_scale_loop(
            image, camera, prior, cand_lon, cand_lat, cand_coarse_mpp, z_fine, basemap,
            scale_iters=scale_iters, matcher=matcher, refine=refine, trust_yaw=trust_yaw,
            rotation_tolerance_deg=rotation_tolerance_deg,
            ransac_threshold_px=ransac_threshold_px, min_inliers=min_inliers, clahe=clahe,
            max_zoom=max_zoom, prerotate_deg=cand_prerotate, min_ncc=min_ncc,
        )
        if r.is_localized and (
            best is None
            or int(r.diagnostics.get("n_inliers", 0)) > int(best.diagnostics.get("n_inliers", 0))
        ):
            best = r
    if best is None:
        return LocalizationResult.failed(
            "точный уровень не сошёлся ни на одном кандидате", **base_diagnostics
        )

    # z_fine поставил цикл масштаба; сохраняем его поверх base_diagnostics.
    z_best = best.diagnostics.get("z_fine", z_fine)
    best.diagnostics.update(base_diagnostics)
    best.diagnostics["z_fine"] = z_best
    best.diagnostics["z_fine_initial"] = z_fine

    # Финальный гейт против ИСХОДНОГО приора (инвариант 3).
    final_offset_m = float(haversine_m(prior.lat, prior.lon, best.center_lat, best.center_lon))
    best.diagnostics["prior_offset_m"] = final_offset_m
    best.diagnostics["prior_offset_sigma"] = final_offset_m / prior.sigma_m
    if final_offset_m > gate_sigma * prior.sigma_m:
        return LocalizationResult.failed("решение вне диска приора", **best.diagnostics)
    return best
