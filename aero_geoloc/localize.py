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

from .basemap import BasemapSource
from .camera import Camera
from .geo import Georef, ground_mpp, haversine_m, zoom_for_mpp
from .matcher import Matcher, SIFTMatcher
from .pose import PoseEstimate, estimate_similarity, refine_ecc
from .types import LocalizationResult, Prior, Status

__all__ = ["normalize_gray", "localize_against_reference", "localize"]

#: Во сколько σ приора укладывается допустимое отклонение центра (инвариант 3).
PRIOR_GATE_SIGMA = 3.0


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
        :class:`~aero_geoloc.types.LocalizationResult`. В фазе 1 статус только
        ``LOCALIZED`` или ``NOT_LOCALIZED``: ``LOW_CONFIDENCE`` требует
        откалиброванного порога, а он появляется вместе с ``quality.py`` в
        фазе 2. По той же причине ``confidence`` здесь — сырая доля инлайеров,
        и в диагностике честно помечена как неоткалиброванная.
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
    corr = matcher.match(query_gray, ref_gray)
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

    return LocalizationResult(
        status=Status.LOCALIZED,
        center_lat=center_lat,
        center_lon=center_lon,
        heading_deg=heading_deg,
        altitude_est_m=altitude_est_m,
        footprint_lonlat=footprint,
        transform=pose.transform.matrix,
        confidence=pose.inlier_ratio,
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

    corr = matcher.match(query_coarse, coarse_ref_gray)
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
        gate_sigma: во сколько σ приора укладывается допустимое отклонение центра.

    Returns:
        :class:`~aero_geoloc.types.LocalizationResult`. Как и раньше, статус
        ``LOW_CONFIDENCE`` не выдаётся до ``quality.py`` (следующий шаг фазы 2).
    """
    matcher = matcher if matcher is not None else SIFTMatcher()
    image_gray = normalize_gray(image, clahe=clahe)
    if image_gray.shape[:2] != (camera.image_height, camera.image_width):
        raise ValueError(
            f"размер кадра {image_gray.shape[1]}×{image_gray.shape[0]} не совпадает "
            f"с камерой {camera.image_width}×{camera.image_height}"
        )

    # Стадия 0/1: приведение масштабов. Зум точного уровня — чтобы mpp ≈ GSD.
    gsd = camera.gsd(prior.altitude_m)
    z_fine = zoom_for_mpp(gsd, prior.lat)
    footprint_max = max(camera.footprint_m(prior.altitude_m))

    # Грубый уровень: зум так, чтобы весь диск ±gate_sigma·σ уместился в ~coarse_target_px.
    search_half_m = 0.5 * footprint_max + gate_sigma * prior.sigma_m
    coarse_mpp_target = 2.0 * search_half_m / coarse_target_px
    z_coarse = min(z_fine, zoom_for_mpp(coarse_mpp_target, prior.lat, mode="coarser"))
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

    coarse_ref, coarse_georef = basemap(prior.lon, prior.lat, z_coarse, coarse_size, coarse_size)
    coarse_gray = normalize_gray(coarse_ref, clahe=clahe)
    candidate = _coarse_center(
        image_gray,
        camera,
        prior,
        coarse_gray,
        coarse_georef,
        matcher=matcher,
        trust_yaw=trust_yaw,
        rotation_tolerance_deg=rotation_tolerance_deg,
        ransac_threshold_px=ransac_threshold_px,
        min_inliers=coarse_min_inliers,
    )
    if candidate is None:
        return LocalizationResult.failed("грубый уровень не нашёл кандидата", **base_diagnostics)

    cand_lon, cand_lat = candidate
    coarse_offset_m = float(haversine_m(prior.lat, prior.lon, cand_lat, cand_lon))
    base_diagnostics["coarse_offset_m"] = coarse_offset_m
    # Кандидат обязан лежать в диске приора — иначе это не он (инвариант 3).
    if coarse_offset_m > gate_sigma * prior.sigma_m:
        return LocalizationResult.failed("кандидат вне диска приора", **base_diagnostics)

    # Точный уровень: маленькое окно нативного разрешения вокруг кандидата.
    # Запас покрывает неопределённость грубого уровня (единицы его пикселей).
    fine_margin_m = max(6.0 * coarse_mpp, 0.15 * footprint_max)
    fine_size = _even(2.0 * (0.5 * footprint_max + fine_margin_m) / ground_mpp(cand_lat, z_fine))
    fine_ref, fine_georef = basemap(cand_lon, cand_lat, z_fine, fine_size, fine_size)

    prior_fine = Prior(
        lat=cand_lat,
        lon=cand_lon,
        sigma_m=fine_margin_m,
        altitude_m=prior.altitude_m,
        altitude_sigma_m=prior.altitude_sigma_m,
        yaw_deg=prior.yaw_deg,
        pitch_deg=prior.pitch_deg,
        roll_deg=prior.roll_deg,
    )
    # В точный уровень отдаём ИСХОДНЫЙ кадр (не image_gray): нормализацию —
    # включая CLAHE — сделает localize_against_reference, одинаково для кадра и
    # подложки. Иначе при clahe=True кадр выравнивался бы дважды, а подложка раз.
    result = localize_against_reference(
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
    )
    result.diagnostics.update(base_diagnostics)

    # Финальный гейт против ИСХОДНОГО приора: точный уровень гейтил против
    # кандидата, но решение обязано укладываться и в исходный диск ±σ.
    if result.is_localized:
        final_offset_m = float(
            haversine_m(prior.lat, prior.lon, result.center_lat, result.center_lon)
        )
        result.diagnostics["prior_offset_m"] = final_offset_m
        result.diagnostics["prior_offset_sigma"] = final_offset_m / prior.sigma_m
        if final_offset_m > gate_sigma * prior.sigma_m:
            return LocalizationResult.failed("решение вне диска приора", **result.diagnostics)

    return result
