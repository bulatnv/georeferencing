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
    "ShotMeta",
    "read_metadata",
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


@dataclass(frozen=True)
class ShotMeta:
    """Метаданные снимка **без** пикселей: что удалось прочитать и чего не хватает.

    Зачем отдельно от :class:`DroneShot`. Тот декодирует кадр — 12 мегапикселей на
    снимок, — и падает, если чего-то нет. Для обзора папки нужно ровно обратное:
    дёшево и не падая, потому что «нет GPS» — это факт о данных, который надо
    показать в списке, а не исключение (``scripts/exif_index.py``).

    Attributes:
        problems: чего не хватает для локализации, человеческим языком. Пустой
            список означает, что снимок пригоден.
    """

    path: Path
    width: int
    height: int
    model: str
    fov_deg: float | None = None
    lat: float | None = None
    lon: float | None = None
    altitude_m: float | None = None          # AGL, из XMP RelativeAltitude
    absolute_altitude_m: float | None = None
    yaw_deg: float | None = None
    pitch_from_nadir_deg: float | None = None
    roll_deg: float | None = None
    datetime: str | None = None
    problems: tuple[str, ...] = ()

    @property
    def has_position(self) -> bool:
        return self.lat is not None and self.lon is not None

    @property
    def is_nadir(self) -> bool:
        """Тот же критерий, что у :attr:`DroneShot.is_nadir`; без питча — считаем да."""
        return self.pitch_from_nadir_deg is None or abs(self.pitch_from_nadir_deg) <= 10.0

    def gsd_m(self) -> float | None:
        """Разрешение на земле [м/пиксель] из высоты и FOV — если оба известны.

        Считается ровно так же, как это сделает пайплайн из ``--altitude`` и
        ``--fov`` (:class:`~aero_geoloc.camera.Camera`), поэтому число из обзора
        воспроизводимо командой один в один.
        """
        if self.altitude_m is None or self.fov_deg is None:
            return None
        return Camera(self.width, self.height, fov_deg=self.fov_deg).gsd(self.altitude_m)


def read_metadata(path: str | Path) -> ShotMeta:
    """Метаданные снимка без декодирования пикселей. Не бросает на нехватке данных.

    Всё, чего не хватает для локализации, собирается в :attr:`ShotMeta.problems`.
    Строгую проверку делает :func:`load_drone_shot`, который на этом и построен.
    """
    path = Path(path)
    img, exif_ifd, gps_ifd, tags, xmp = _exif_and_xmp(path)
    W, H = img.size
    problems: list[str] = []

    f35 = exif_ifd.get(41989)  # FocalLengthIn35mmFilm
    fov = (2.0 * math.degrees(math.atan(0.5 * _FILM_WIDTH_MM / float(f35)))) if f35 else None
    if fov is None:
        problems.append("нет FocalLengthIn35mmFilm — FOV не восстановить, задайте --gsd")

    lat = lon = None
    if gps_ifd and 2 in gps_ifd and 4 in gps_ifd:
        lat = _dms_to_deg(gps_ifd[2], gps_ifd.get(1, "N"))
        lon = _dms_to_deg(gps_ifd[4], gps_ifd.get(3, "E"))
    else:
        problems.append("нет GPS в EXIF — истину и приор задавать руками")

    yaw = pitch = roll = rel_alt = None
    try:
        yaw, pitch, roll, rel_alt = _orientation(xmp)
    except ValueError:
        problems.append("нет ориентации в XMP — курс неизвестен, сборка карты ×8 дороже")
    if rel_alt is None:
        problems.append("нет RelativeAltitude — высоту над землёй задавать руками")
    if pitch is not None and abs(pitch) > 10.0:
        problems.append(f"наклон от надира {abs(pitch):.0f}° — вне надирной модели")

    return ShotMeta(
        path=path, width=W, height=H, model=str(tags.get("Model", "")).strip("\x00 "),
        fov_deg=fov, lat=lat, lon=lon,
        altitude_m=None if rel_alt is None else float(rel_alt),
        absolute_altitude_m=None if gps_ifd.get(6) is None else float(gps_ifd[6]),
        yaw_deg=yaw, pitch_from_nadir_deg=pitch, roll_deg=roll,
        datetime=str(tags.get("DateTime", "")).strip() or None,
        problems=tuple(problems),
    )


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
    from PIL import Image

    path = Path(path)
    meta = read_metadata(path)              # разбор метаданных живёт в одном месте

    if meta.fov_deg is None:
        raise ValueError(f"{path.name}: нет FocalLengthIn35mmFilm — не могу восстановить FOV")
    if not meta.has_position:
        raise ValueError(f"{path.name}: нет GPS в EXIF")
    if meta.yaw_deg is None:
        raise ValueError(f"{path.name}: нет курса: ни DJI GimbalYawDegree, ни Camera:Yaw")

    if meta.altitude_m is not None:
        altitude = meta.altitude_m
    elif altitude_override_m is not None:
        altitude = altitude_override_m
    elif ground_elevation_m is not None or use_dem:
        if meta.absolute_altitude_m is None:
            raise ValueError(f"{path.name}: нет абс. высоты в EXIF для расчёта AGL")
        ground = (ground_elevation_m if ground_elevation_m is not None
                  else lookup_ground_elevation(meta.lat, meta.lon))
        altitude = meta.absolute_altitude_m - ground
    else:
        raise ValueError(
            f"{path.name}: нет высоты над землёй — для survey-камеры передайте "
            "ground_elevation_m/altitude_override_m или use_dem=True"
        )

    bgr = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    return DroneShot(
        image_bgr=bgr, camera=Camera(meta.width, meta.height, fov_deg=meta.fov_deg),
        true_lat=meta.lat, true_lon=meta.lon, altitude_m=float(altitude),
        yaw_deg=float(meta.yaw_deg) + magnetic_declination_deg,
        pitch_from_nadir_deg=float(meta.pitch_from_nadir_deg or 0.0),
        roll_deg=float(meta.roll_deg or 0.0), model=meta.model,
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
