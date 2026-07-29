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

__all__ = [
    "DroneShot",
    "load_drone_shot",
    "frame_at_mpp",
    "basemap_zoom_for",
    "lookup_ground_elevation",
]

#: 35-мм кадр: полная ширина 36 мм. f_px = f35 · W / 36, отсюда HFOV = 2·atan(18/f35).
_FILM_WIDTH_MM = 36.0
#: User-Agent для DEM-запросов.
USER_AGENT = "aero-geoloc/0.5 (UAV visual localization)"


@dataclass(frozen=True)
class DroneShot:
    """Бортовой снимок с разобранными метаданными.

    Attributes:
        image_bgr: кадр (BGR, полный размер).
        camera: модель камеры (FOV из 35-мм эквивалента фокуса).
        true_lat, true_lon: GPS-позиция камеры — ground truth И центр приора.
        altitude_m: высота над землёй (AGL). У DJI — ``RelativeAltitude``; у
            survey-камер — ``абс. высота − рельеф`` (см. :func:`load_drone_shot`).
        yaw_deg: курс камеры относительно севера (DJI ``GimbalYawDegree`` либо
            фотограмметрический ``Camera:Yaw``).
        pitch_from_nadir_deg, roll_deg: отклонение оптической оси от надира
            (0 = строго вниз) и крен, градусы.
        model: модель камеры из EXIF.
    """

    image_bgr: np.ndarray
    camera: Camera
    true_lat: float
    true_lon: float
    altitude_m: float
    yaw_deg: float
    pitch_from_nadir_deg: float
    roll_deg: float
    model: str

    def prior(self, *, sigma_m: float = 25.0, altitude_sigma_m: float = 20.0) -> Prior:
        """Приор из метаданных: центр = GPS, курс = yaw камеры.

        GPS точен (единицы метров), поэтому ``sigma_m`` по умолчанию узкий —
        и грубое окно остаётся небольшим, что важно для обучаемых матчеров
        (они деградируют на окне много больше кадра).
        """
        return Prior(
            lat=self.true_lat, lon=self.true_lon, sigma_m=sigma_m,
            altitude_m=self.altitude_m, altitude_sigma_m=altitude_sigma_m,
            yaw_deg=self.yaw_deg, pitch_deg=self.pitch_from_nadir_deg, roll_deg=self.roll_deg,
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


def _xmp_num(xmp: str, name: str) -> float | None:
    """Число из XMP по локальному имени тега в любом пространстве имён."""
    m = re.search(rf'[\w-]+:{name}="([^"]+)"', xmp) or re.search(rf"<[\w-]+:{name}>([^<]+)<", xmp)
    return float(m.group(1)) if m else None


def _orientation(xmp: str) -> tuple[float, float, float, float | None]:
    """Курс, наклон от надира, крен и относительная высота из XMP.

    Поддержаны два формата: **DJI** (``drone-dji:Gimbal*``/``RelativeAltitude``,
    где питч подвеса −90° = надир) и **фотограмметрический** (``Camera:Yaw/Pitch/
    Roll`` у senseFly/S.O.D.A., где Pitch уже отсчитан от надира, а высоты в XMP
    нет). Возвращает ``(yaw, pitch_from_nadir, roll, rel_alt|None)``.
    """
    if _xmp_num(xmp, "GimbalYawDegree") is not None:  # DJI
        gp = _xmp_num(xmp, "GimbalPitchDegree")
        return (
            _xmp_num(xmp, "GimbalYawDegree"),
            (gp + 90.0) if gp is not None else 0.0,
            _xmp_num(xmp, "GimbalRollDegree") or 0.0,
            _xmp_num(xmp, "RelativeAltitude"),
        )
    yaw = _xmp_num(xmp, "Yaw")  # Camera:Yaw (survey)
    if yaw is None:
        raise ValueError("нет курса: ни DJI GimbalYawDegree, ни Camera:Yaw")
    return (yaw, _xmp_num(xmp, "Pitch") or 0.0, _xmp_num(xmp, "Roll") or 0.0, None)


def lookup_ground_elevation(lat: float, lon: float, *, timeout: float = 20.0) -> float:
    """Высота рельефа в точке [м] через открытый DEM-сервис (open-elevation).

    Нужна для survey-камер, у которых в EXIF только **абсолютная** высота над
    уровнем моря: AGL = абс. высота − рельеф. Требует сеть.
    """
    import json
    import urllib.parse
    import urllib.request

    url = "https://api.open-elevation.com/api/v1/lookup?" + urllib.parse.urlencode(
        {"locations": f"{lat},{lon}"}
    )
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return float(json.loads(resp.read())["results"][0]["elevation"])


def load_drone_shot(
    path: str | Path,
    *,
    ground_elevation_m: float | None = None,
    altitude_override_m: float | None = None,
    use_dem: bool = False,
    magnetic_declination_deg: float = 0.0,
) -> DroneShot:
    """Прочитать снимок и метаданные в :class:`DroneShot` (DJI и survey-камеры).

    GPS и фокус (FOV из ``FocalLengthIn35mmFilm``) — из EXIF; курс/наклон/высота —
    из XMP (DJI ``drone-dji:*`` либо фотограмметрический ``Camera:*``, см.
    :func:`_orientation`).

    Высота над землёй (AGL): у DJI — из ``RelativeAltitude``. У survey-камер её в
    XMP нет, поэтому её берут из ``altitude_override_m``, либо считают как
    ``абс. высота (EXIF) − ground_elevation_m``, либо (``use_dem=True``) рельеф
    запрашивается из DEM автоматически (:func:`lookup_ground_elevation`, сеть).

    Курс приводится к **истинному** северу: ``magnetic_declination_deg``
    прибавляется к ``yaw`` (DJI отдаёт курс относительно **магнитного** севера, а
    ENU-геометрия — относительно истинного; для средних широт РФ склонение
    +10…+15°, и без поправки визуальная одометрия систематически дрейфует).

    Raises:
        ValueError: нет GPS/фокуса/курса, либо для survey не задана высота над
            землёй (ни ``ground_elevation_m``/``altitude_override_m``, ни ``use_dem``).
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

    try:
        yaw, pitch_from_nadir, roll, rel_alt = _orientation(xmp)
    except ValueError as exc:
        raise ValueError(f"{path.name}: {exc}") from exc

    if rel_alt is not None:
        altitude = rel_alt
    elif altitude_override_m is not None:
        altitude = altitude_override_m
    elif ground_elevation_m is not None or use_dem:
        gps_alt = gps_ifd.get(6)
        if gps_alt is None:
            raise ValueError(f"{path.name}: нет абс. высоты в EXIF для расчёта AGL")
        ground = ground_elevation_m if ground_elevation_m is not None else lookup_ground_elevation(lat, lon)
        altitude = float(gps_alt) - ground
    else:
        raise ValueError(
            f"{path.name}: нет высоты над землёй — для survey-камеры передайте "
            "ground_elevation_m/altitude_override_m или use_dem=True"
        )

    bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    model = str(tags.get("Model", "")).strip("\x00 ")
    return DroneShot(
        image_bgr=bgr, camera=Camera(W, H, fov_deg=fov_deg),
        true_lat=lat, true_lon=lon, altitude_m=float(altitude),
        yaw_deg=float(yaw) + magnetic_declination_deg,
        pitch_from_nadir_deg=float(pitch_from_nadir),
        roll_deg=float(roll), model=model,
    )


def frame_at_mpp(shot: DroneShot, target_mpp: float) -> tuple[np.ndarray, Camera]:
    """Ресемпл кадра до разрешения подложки ``target_mpp`` + согласованная камера.

    Приведение масштаба (стадия 1) для низких высот: кадр в разы детальнее
    подложки, и матчинг на нативном разрешении дорог и неустойчив. Ресемпл до
    ``mpp ≈ GSD_подложки`` делает масштаб ≈ 1. FOV сохраняется, поэтому камера
    просто пересобирается под новый размер.
    """
    from .camera import resample_to_mpp

    return resample_to_mpp(shot.image_bgr, shot.camera,
                           shot.camera.gsd(shot.altitude_m), target_mpp)


def basemap_zoom_for(shot: DroneShot, *, max_zoom: int) -> int:
    """Зум подложки под GSD кадра, клампованный к максимуму провайдера."""
    return zoom_for_mpp(shot.camera.gsd(shot.altitude_m), shot.true_lat, max_zoom=max_zoom)
