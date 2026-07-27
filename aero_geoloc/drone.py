"""Загрузка бортового снимка: EXIF/XMP → камера, приор и истинное GPS-положение.

Мост между реальными кадрами с дрона и пайплайном. Из снимка извлекаются:
модель камеры (по 35-мм эквиваленту фокуса), GPS-позиция и высота (ground truth
и одновременно приор), курс/наклон подвеса (DJI XMP ``drone-dji:*``). Это и есть
вход для валидации всего конвейера на настоящем appearance gap (борт ↔ спутник),
которую синтетика показать не может (``docs/PLAN.md``, фаза 4).

Требует Pillow (EXIF) — опциональная зависимость ``real``. Тайлы подложки и
обучаемый матчер (LightGlue/LoFTR) — как обычно, за своими интерфейсами.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .camera import Camera
from .geo import ground_mpp, zoom_for_mpp
from .types import Prior

__all__ = ["DroneShot", "load_drone_shot", "frame_at_mpp", "basemap_zoom_for"]

#: 35-мм кадр: полная ширина 36 мм. f_px = f35 · W / 36, отсюда HFOV = 2·atan(18/f35).
_FILM_WIDTH_MM = 36.0


@dataclass(frozen=True)
class DroneShot:
    """Бортовой снимок с разобранными метаданными.

    Attributes:
        image_bgr: кадр (BGR, полный размер).
        camera: модель камеры (FOV из 35-мм эквивалента фокуса).
        true_lat, true_lon: GPS-позиция камеры — ground truth И центр приора.
        altitude_m: высота над точкой взлёта (``RelativeAltitude``).
        yaw_deg: курс подвеса относительно севера (``GimbalYawDegree``).
        pitch_from_nadir_deg: отклонение оптической оси от надира (0 = строго вниз).
        model: модель камеры из EXIF.
    """

    image_bgr: np.ndarray
    camera: Camera
    true_lat: float
    true_lon: float
    altitude_m: float
    yaw_deg: float
    pitch_from_nadir_deg: float
    model: str

    def prior(self, *, sigma_m: float = 25.0, altitude_sigma_m: float = 20.0) -> Prior:
        """Приор из метаданных: центр = GPS, курс = yaw подвеса.

        GPS точен (единицы метров), поэтому ``sigma_m`` по умолчанию узкий —
        и грубое окно остаётся небольшим, что важно для обучаемых матчеров
        (они деградируют на окне много больше кадра).
        """
        return Prior(
            lat=self.true_lat, lon=self.true_lon, sigma_m=sigma_m,
            altitude_m=self.altitude_m, altitude_sigma_m=altitude_sigma_m,
            yaw_deg=self.yaw_deg, pitch_deg=self.pitch_from_nadir_deg,
        )

    @property
    def is_nadir(self) -> bool:
        """Достаточно ли близок кадр к надиру, чтобы модель подобия была применима."""
        return abs(self.pitch_from_nadir_deg) <= 10.0


def _exif_and_xmp(path: Path):
    from PIL import ExifTags, Image  # локальный импорт: Pillow — опциональная зависимость

    img = Image.open(path)
    exif = img.getexif()
    exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
    gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
    tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
    xmp_match = re.search(rb"<x:xmpmeta.*?</x:xmpmeta>", path.read_bytes(), re.S)
    xmp = xmp_match.group(0).decode("utf-8", "ignore") if xmp_match else ""
    return img, exif_ifd, gps_ifd, tags, xmp


def _dms_to_deg(value, ref: str) -> float:
    deg = float(value[0]) + float(value[1]) / 60.0 + float(value[2]) / 3600.0
    return -deg if ref in ("S", "W") else deg


def _xmp_float(xmp: str, key: str) -> float | None:
    m = re.search(rf'drone-dji:{key}="([^"]+)"', xmp) or re.search(rf"<drone-dji:{key}>([^<]+)<", xmp)
    return float(m.group(1)) if m else None


def load_drone_shot(path: str | Path) -> DroneShot:
    """Прочитать снимок и метаданные в :class:`DroneShot`.

    Курс/высота берутся из XMP DJI (``GimbalYawDegree``/``RelativeAltitude``),
    GPS — из EXIF, FOV — из 35-мм эквивалента фокуса (``FocalLengthIn35mmFilm``).

    Raises:
        ValueError: если в снимке нет нужных полей (не DJI / нет GPS / нет фокуса).
    """
    import cv2

    path = Path(path)
    img, exif_ifd, gps_ifd, tags, xmp = _exif_and_xmp(path)
    W, H = img.size

    f35 = exif_ifd.get(41989)  # FocalLengthIn35mmFilm
    if f35 is None:
        raise ValueError(f"{path.name}: нет FocalLengthIn35mmFilm — не могу восстановить FOV")
    fov_deg = 2.0 * math.degrees(math.atan(0.5 * _FILM_WIDTH_MM / float(f35)))

    if not gps_ifd or 2 not in gps_ifd or 4 not in gps_ifd:
        raise ValueError(f"{path.name}: нет GPS в EXIF")
    lat = _dms_to_deg(gps_ifd[2], gps_ifd.get(1, "N"))
    lon = _dms_to_deg(gps_ifd[4], gps_ifd.get(3, "E"))

    alt = _xmp_float(xmp, "RelativeAltitude")
    yaw = _xmp_float(xmp, "GimbalYawDegree")
    pitch = _xmp_float(xmp, "GimbalPitchDegree")
    if alt is None or yaw is None:
        raise ValueError(f"{path.name}: нет DJI XMP (RelativeAltitude/GimbalYawDegree)")
    # Питч подвеса: −90° = строго вниз (надир). Отклонение от надира = pitch + 90.
    pitch_from_nadir = (pitch + 90.0) if pitch is not None else 0.0

    bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    model = str(tags.get("Model", "")).strip("\x00 ")
    return DroneShot(
        image_bgr=bgr, camera=Camera(W, H, fov_deg=fov_deg),
        true_lat=lat, true_lon=lon, altitude_m=float(alt),
        yaw_deg=float(yaw), pitch_from_nadir_deg=float(pitch_from_nadir), model=model,
    )


def frame_at_mpp(shot: DroneShot, target_mpp: float) -> tuple[np.ndarray, Camera]:
    """Ресемпл кадра до разрешения подложки ``target_mpp`` + согласованная камера.

    Приведение масштаба (стадия 1) для низких высот: кадр в разы детальнее
    подложки, и матчинг на нативном разрешении дорог и неустойчив. Ресемпл до
    ``mpp ≈ GSD_подложки`` делает масштаб ≈ 1. FOV сохраняется, поэтому камера
    просто пересобирается под новый размер.
    """
    import cv2

    scale = shot.camera.gsd(shot.altitude_m) / target_mpp
    new_w = max(16, round(shot.camera.image_width * scale))
    new_h = max(16, round(shot.camera.image_height * scale))
    frame = cv2.resize(shot.image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    camera = Camera(new_w, new_h, fov_deg=shot.camera.fov_deg)
    return frame, camera


def basemap_zoom_for(shot: DroneShot, *, max_zoom: int) -> int:
    """Зум подложки под GSD кадра, клампованный к максимуму провайдера."""
    return zoom_for_mpp(shot.camera.gsd(shot.altitude_m), shot.true_lat, max_zoom=max_zoom)
