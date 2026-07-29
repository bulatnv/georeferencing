"""Вход инструмента: снимок + то, что о нём известно, → камера и приор.

Зачем модуль ([TOOL_PLAN.md](../docs/TOOL_PLAN.md), этап T2). Пайплайну нужны
камера и приор, а у владельца на руках бывает разное: у одного снимка есть EXIF с
GPS и высотой, у другого метаданные срезаны и известен только GSD из паспорта
съёмки, у третьего — высота полёта и параметры объектива. Раньше каждый случай
разбирался в своём скрипте по-своему; здесь он один и с явными правилами.

Приоритет источников — от самого надёжного к самому косвенному:

1. **Явный GSD.** Он напрямую задаёт зум подложки и ожидаемый масштаб, поэтому
   если он есть — берётся он.
2. **Высота + параметры камеры** (FOV либо фокус и размер сенсора). GSD
   вычисляется.
3. **EXIF/XMP снимка**: камера, высота, курс.

**GPS из EXIF в приор молча не подставляется.** Это не перестраховка: если взять
координаты снимка как приор и «найти» его рядом с ними, проверка ничего не
доказывает — система искала там, куда ей же и указали. Инструмент показывает, что
GPS в снимке есть, и требует явного ``from_exif=True``.

Модуль намеренно **не** трогает пиксели: он читает только размер кадра. Полный
кадр читается позже, при приведении масштаба, — снимки бывают по 40 Мп.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from .camera import Camera, resample_to_mpp
from .types import Prior

__all__ = ["LocateRequest", "build_request", "InputError"]

#: Высота, принимаемая, когда известен только GSD. На геометрию она не влияет
#: (отпечаток = GSD × ширина кадра), но входит в приор и в оценку высоты на
#: выходе — поэтому о подстановке сообщается, а не умалчивается.
ASSUMED_ALTITUDE_M = 500.0


class InputError(ValueError):
    """Входных данных не хватает или они противоречат друг другу.

    Отдельный тип, потому что это ошибка **пользователя**, а не сбой: сообщение
    должно называть, чего именно не хватает и чем это дать.
    """


@dataclass(frozen=True)
class LocateRequest:
    """Всё, что нужно пайплайну, плюс происхождение каждого числа.

    Происхождение хранится не для красоты: отчёт обязан отличать «приор задан
    владельцем» от «приор взят из EXIF снимка», иначе успешная локализация
    выглядит доказательством, не будучи им.
    """

    image_path: Path
    camera: Camera
    prior: Prior
    gsd_m: float
    gsd_source: str
    prior_source: str
    trust_yaw: bool
    notes: tuple[str, ...] = field(default_factory=tuple)

    def frame_at_mpp(self, target_mpp: float):
        """Кадр, приведённый к разрешению подложки, и согласованная камера.

        Полный кадр читается здесь, а не в конструкторе: снимки бывают по 40 Мп,
        и держать их в памяти до того, как стало ясно, что вход вообще корректен,
        незачем.
        """
        image = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise InputError(f"{self.image_path}: не читается как изображение")
        return resample_to_mpp(image, self.camera, self.gsd_m, target_mpp)

    def basemap_zoom(self, *, max_zoom: int) -> int:
        """Зум подложки под GSD кадра, клампованный к максимуму провайдера."""
        from .geo import zoom_for_mpp

        return zoom_for_mpp(self.gsd_m, self.prior.lat, max_zoom=max_zoom)

    @property
    def footprint_m(self) -> tuple[float, float]:
        """Сколько метров земли покрывает кадр по сторонам."""
        return self.camera.footprint_m(self.prior.altitude_m)

    def describe(self) -> str:
        """Строка для владельца — то, что он должен проверить глазами ДО запуска.

        Отпечаток в метрах здесь главное. Неверный GSD — самая частая ошибка
        ввода, и она даёт не отказ, а уверенно-неверный масштаб. Увидев «кадр
        покрывает 230 м», владелец сразу заметит, если снимал с другой высоты.
        """
        w, h = self.footprint_m
        yaw = f"{self.prior.yaw_deg:.0f}°" if self.trust_yaw else "неизвестен"
        return (
            f"кадр {self.camera.image_width}×{self.camera.image_height}, "
            f"GSD {self.gsd_m:.4f} м ({self.gsd_source}) ⇒ покрывает {w:.0f}×{h:.0f} м; "
            f"приор {self.prior.lat:.5f}/{self.prior.lon:.5f} ±{self.prior.sigma_m:.0f} м "
            f"({self.prior_source}), курс {yaw}"
        )


def _image_size(path: Path) -> tuple[int, int]:
    """Размер кадра без чтения всех пикселей."""
    reduced = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_8)
    if reduced is None:
        raise InputError(f"{path}: не читается как изображение")
    return int(reduced.shape[1] * 8), int(reduced.shape[0] * 8)


def _focal_px(width: int, *, fov_deg: float | None,
              focal_mm: float | None, sensor_width_mm: float | None) -> float | None:
    """Фокус в пикселях из того, чем задана камера."""
    if fov_deg is not None:
        return (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    if focal_mm is not None and sensor_width_mm is not None:
        return width * focal_mm / sensor_width_mm
    return None


def build_request(
    image: str | Path,
    *,
    lat: float | None = None,
    lon: float | None = None,
    sigma_m: float | None = None,
    gsd_m: float | None = None,
    altitude_m: float | None = None,
    fov_deg: float | None = None,
    focal_mm: float | None = None,
    sensor_width_mm: float | None = None,
    yaw_deg: float | None = None,
    from_exif: bool = False,
    altitude_sigma_m: float = 50.0,
) -> LocateRequest:
    """Собрать вход пайплайна из того, что известно о снимке.

    Raises:
        InputError: не хватает данных либо они противоречивы. Сообщение всегда
            называет, чем именно закрыть пробел.
    """
    path = Path(image)
    if not path.exists():
        raise InputError(f"нет файла {path}")
    width, height = _image_size(path)
    notes: list[str] = []

    shot = None
    need_exif = from_exif or (gsd_m is None and altitude_m is None) or yaw_deg is None
    if need_exif:
        try:
            from .drone import load_drone_shot

            shot = load_drone_shot(path)
        except Exception as exc:  # noqa: BLE001 — отсутствие метаданных штатно
            shot = None
            if from_exif:
                raise InputError(
                    f"попросили взять данные из EXIF, но прочитать их не удалось: {exc}. "
                    f"Задайте --lat/--lon и --gsd (или --altitude с параметрами камеры)"
                ) from exc

    # --- GSD и камера --------------------------------------------------------
    if gsd_m is not None:
        if gsd_m <= 0:
            raise InputError("--gsd должен быть > 0")
        used_altitude = altitude_m if altitude_m is not None else ASSUMED_ALTITUDE_M
        if altitude_m is None:
            notes.append(
                f"высота не задана — принята {ASSUMED_ALTITUDE_M:.0f} м. На геометрию это "
                f"не влияет (отпечаток = GSD × ширина кадра), но оценка высоты на выходе "
                f"будет отсчитываться от этого допущения"
            )
        if fov_deg is not None or focal_mm is not None:
            notes.append("заданы и GSD, и параметры камеры — использован GSD как более прямой")
        camera = Camera.from_gsd(width, height, gsd_m=gsd_m, altitude_m=used_altitude)
        gsd_source = "задан явно"
    elif altitude_m is not None and (fov_deg is not None or focal_mm is not None):
        focal = _focal_px(width, fov_deg=fov_deg, focal_mm=focal_mm,
                          sensor_width_mm=sensor_width_mm)
        if focal is None:
            raise InputError(
                "для расчёта GSD из высоты нужен либо --fov, либо --focal-mm вместе "
                "с --sensor-mm"
            )
        used_altitude = altitude_m
        gsd_m = altitude_m / focal
        camera = Camera.from_gsd(width, height, gsd_m=gsd_m, altitude_m=used_altitude)
        gsd_source = "из высоты и параметров камеры"
    elif shot is not None:
        used_altitude = altitude_m if altitude_m is not None else shot.altitude_m
        camera = shot.camera
        gsd_m = camera.gsd(used_altitude)
        gsd_source = "из EXIF/XMP снимка"
        if not shot.is_nadir:
            notes.append(
                f"наклон от надира {abs(shot.pitch_from_nadir_deg):.0f}° — модель плоской "
                f"земли применима плохо, результату верить осторожно"
            )
    else:
        raise InputError(
            "нечем задать масштаб кадра. Дайте одно из:\n"
            "  --gsd 0.065                      (разрешение на земле, м/пиксель)\n"
            "  --altitude 300 --fov 73          (высота съёмки и угол обзора)\n"
            "  --altitude 300 --focal-mm 8.8 --sensor-mm 13.2\n"
            "  --from-exif                      (если в снимке есть метаданные)"
        )

    # --- приор ---------------------------------------------------------------
    if lat is not None and lon is not None:
        prior_lat, prior_lon, prior_source = lat, lon, "задан аргументами"
        if shot is not None and not from_exif:
            notes.append(
                f"в снимке есть GPS {shot.true_lat:.5f}/{shot.true_lon:.5f} — он НЕ "
                f"использован как приор. Это намеренно: искать кадр рядом с его же "
                f"координатами и считать это проверкой нельзя"
            )
    elif from_exif and shot is not None:
        prior_lat, prior_lon = shot.true_lat, shot.true_lon
        prior_source = "GPS из EXIF (НЕ независимая проверка)"
        notes.append(
            "приор взят из GPS самого снимка — успешная локализация здесь ничего не "
            "доказывает, система искала там, куда ей указали"
        )
    else:
        raise InputError(
            "не задан приор: нужны --lat и --lon (либо --from-exif, если в снимке есть GPS)"
        )

    if sigma_m is None:
        raise InputError("не задана погрешность приора: --sigma-km или --sigma-m")
    if sigma_m <= 0:
        raise InputError("погрешность приора должна быть > 0")

    if yaw_deg is None and shot is not None:
        yaw_deg = shot.yaw_deg
    trust_yaw = yaw_deg is not None
    if not trust_yaw:
        notes.append(
            "курс неизвестен — карта района строится с поворотами клеток, это в 8 раз "
            "дороже по времени сборки. Если курс известен, задайте --yaw"
        )

    prior = Prior(
        lat=float(prior_lat), lon=float(prior_lon), sigma_m=float(sigma_m),
        altitude_m=float(used_altitude), altitude_sigma_m=float(altitude_sigma_m),
        yaw_deg=float(yaw_deg) if trust_yaw else 0.0,
        pitch_deg=shot.pitch_from_nadir_deg if shot is not None else 0.0,
        roll_deg=shot.roll_deg if shot is not None else 0.0,
    )
    return LocateRequest(
        image_path=path, camera=camera, prior=prior, gsd_m=float(gsd_m),
        gsd_source=gsd_source, prior_source=prior_source, trust_yaw=trust_yaw,
        notes=tuple(notes),
    )
