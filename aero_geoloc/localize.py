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

import numpy as np

from .camera import Camera
from .geo import Georef, haversine_m
from .matcher import Matcher, SIFTMatcher
from .pose import PoseEstimate, estimate_similarity
from .types import LocalizationResult, Prior, Status

__all__ = ["normalize_gray", "localize_against_reference"]

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

    diagnostics = {**base_diagnostics, **pose.diagnostics()}

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
