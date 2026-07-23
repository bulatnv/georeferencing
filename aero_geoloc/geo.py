"""Геометрия Web Mercator (EPSG:3857) и привязка растра к географическим координатам.

Класс :class:`Georef` — **единственный мост «пиксель ↔ координата»** в системе
(инвариант 5 из ``docs/ARCHITECTURE.md``). Никакой другой модуль не имеет права
пересчитывать координаты самостоятельно.

Соглашения о координатах
------------------------
**Мировые пиксели** пирамиды Web Mercator — непрерывные, начало в углу:
``world = (0, 0)`` — левый верхний угол мира (lon = −180°, lat = +85.0511°),
``world = (256·2^z, 256·2^z)`` — правый нижний. Это стандартная slippy-map
раскладка: номер тайла = ``floor(world_px / 256)``.

**Локальные пиксели растра** — соглашение «центр пикселя», как в OpenCV:
``(0, 0)`` — это ЦЕНТР левого верхнего пикселя, а не его угол. Растр
шириной ``W`` занимает непрерывный диапазон ``[-0.5, W-0.5]``, его центр —
``((W-1)/2, (H-1)/2)``. Выбор продиктован тем, что координаты приходят от
детекторов особых точек и ``cv2.warpAffine``, которые работают именно в этом
соглашении; полупиксельное расхождение здесь стоило бы ~10 см на 600 м.

Ось Y направлена вниз (север — вверх), как в изображении.

Две сферы, и это не баг
-----------------------
:func:`ground_mpp` считает разрешение на сфере Web Mercator
(``R = 6378137``, так проекция определена, и так устроены все справочные
таблицы зумов), а :func:`haversine_m` меряет расстояния на средней сфере
WGS84 (``R = 6371008.8``, конвенция всех реализаций haversine). Отсюда
систематическое расхождение ``R_mean / R_merc ≈ 0.9989`` (0.11%) между
«пикселями × mpp» и «haversine между теми же точками»; на настоящем
эллипсоиде разница ещё до ~0.3% и зависит от широты.

Это модельное свойство, а не ошибка, и оно на порядок ниже собственной
ошибки геопривязки подложки (ограничение 1 в ``docs/README.md``): на
footprint 200 м это ~0.2 м. Расхождение зафиксировано тестом
``test_ground_mpp_and_haversine_differ_only_by_radius_convention`` — если
понадобится суб-метровая абсолютная точность, начинать надо отсюда.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "EARTH_RADIUS_M",
    "EARTH_MEAN_RADIUS_M",
    "TILE_SIZE_PX",
    "EQUATOR_MPP_Z0",
    "MAX_LATITUDE",
    "world_size_px",
    "lonlat_to_world_px",
    "world_px_to_lonlat",
    "ground_mpp",
    "exact_zoom_for_mpp",
    "zoom_for_mpp",
    "haversine_m",
    "Georef",
]

#: Большая полуось WGS84 — радиус сферы, на которой построен Web Mercator.
EARTH_RADIUS_M = 6378137.0
#: Средний радиус Земли (IUGG) — используется для измерения расстояний.
EARTH_MEAN_RADIUS_M = 6371008.8
#: Размер тайла в пикселях (стандарт XYZ).
TILE_SIZE_PX = 256
#: Разрешение подложки на экваторе при zoom = 0: 2πR / 256.
EQUATOR_MPP_Z0 = 2.0 * math.pi * EARTH_RADIUS_M / TILE_SIZE_PX  # 156543.0339...
#: Предел широты Web Mercator (проекция квадратная, полюса недостижимы).
MAX_LATITUDE = 85.05112877980659


def _unwrap(value: np.ndarray, scalar: bool) -> np.ndarray | float:
    """Вернуть python-float, если на вход подавали скаляр, иначе массив."""
    return float(value) if scalar else value


def world_size_px(zoom: float) -> float:
    """Размер всего мира в пикселях на данном зуме: ``256 · 2^zoom``."""
    return TILE_SIZE_PX * 2.0**zoom


def lonlat_to_world_px(lon, lat, zoom: float):
    """(lon, lat) в градусах → непрерывные мировые пиксели Web Mercator.

    Широта клампится к ``±MAX_LATITUDE``: за этим пределом проекция не
    определена. Принимает скаляры или массивы numpy.

    Returns:
        Кортеж ``(x, y)`` того же типа/формы, что и вход.
    """
    scalar = np.isscalar(lon) and np.isscalar(lat)
    lon = np.asarray(lon, dtype=float)
    lat = np.clip(np.asarray(lat, dtype=float), -MAX_LATITUDE, MAX_LATITUDE)

    n = world_size_px(zoom)
    x = (lon + 180.0) / 360.0 * n
    # y = (0.5 − artanh(sin φ) / 2π) · n; форма через log устойчивее tan+sec.
    sin_lat = np.sin(np.radians(lat))
    y = (0.5 - np.log((1.0 + sin_lat) / (1.0 - sin_lat)) / (4.0 * np.pi)) * n
    return _unwrap(x, scalar), _unwrap(y, scalar)


def world_px_to_lonlat(x, y, zoom: float):
    """Непрерывные мировые пиксели Web Mercator → (lon, lat) в градусах.

    Точная обратная функция к :func:`lonlat_to_world_px` (в пределах клампа
    широты). Принимает скаляры или массивы numpy.
    """
    scalar = np.isscalar(x) and np.isscalar(y)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    n = world_size_px(zoom)
    lon = x / n * 360.0 - 180.0
    # sin φ = tanh((0.5 − y/n) · 2π)
    lat = np.degrees(np.arcsin(np.tanh((0.5 - y / n) * 2.0 * np.pi)))
    return _unwrap(lon, scalar), _unwrap(lat, scalar)


def ground_mpp(lat, zoom: float):
    """Истинное разрешение подложки [м/пиксель] на широте ``lat`` и зуме ``zoom``.

    ``mpp(lat, z) = 156543.034 · cos(lat) / 2^z``

    Это метры на пиксель НА ЗЕМЛЕ: Web Mercator растягивает масштаб как
    ``1/cos(lat)``, поэтому на широте 60° пиксель вдвое «мельче», чем на экваторе.
    """
    scalar = np.isscalar(lat)
    lat = np.asarray(lat, dtype=float)
    mpp = EQUATOR_MPP_Z0 * np.cos(np.radians(lat)) / 2.0**zoom
    return _unwrap(mpp, scalar)


def exact_zoom_for_mpp(target_mpp: float, lat: float) -> float:
    """Дробный зум, на котором разрешение подложки в точности равно ``target_mpp``.

    Обратная к :func:`ground_mpp`. Полезна для расчёта коэффициента ресемпла
    кадра при приведении масштабов (стадия 1 пайплайна).
    """
    if target_mpp <= 0.0:
        raise ValueError(f"target_mpp должен быть > 0, получено {target_mpp}")
    return math.log2(EQUATOR_MPP_Z0 * math.cos(math.radians(lat)) / target_mpp)


def zoom_for_mpp(
    target_mpp: float,
    lat: float,
    *,
    mode: str = "nearest",
    min_zoom: int = 0,
    max_zoom: int = 22,
) -> int:
    """Целый зум подложки под желаемое разрешение ``target_mpp`` [м/пиксель].

    Args:
        target_mpp: целевое разрешение, обычно GSD кадра (см. ``Camera.gsd``).
        lat: широта, на которой считается разрешение.
        mode: ``"nearest"`` — ближайший зум; ``"finer"`` — не грубее целевого
            (mpp ≤ target, подложка детальнее кадра); ``"coarser"`` — не
            детальнее целевого.
        min_zoom, max_zoom: границы, в которые клампится результат.

    Returns:
        Целый уровень зума XYZ-пирамиды.
    """
    exact = exact_zoom_for_mpp(target_mpp, lat)
    if mode == "nearest":
        zoom = round(exact)
    elif mode == "finer":  # больше зум → меньше mpp
        zoom = math.ceil(exact)
    elif mode == "coarser":
        zoom = math.floor(exact)
    else:
        raise ValueError(f"неизвестный mode={mode!r}, ожидается nearest/finer/coarser")
    return int(min(max(zoom, min_zoom), max_zoom))


def haversine_m(lat1, lon1, lat2, lon2):
    """Расстояние по большому кругу между точками [м]. Порядок аргументов: lat, lon.

    Сферическая модель со средним радиусом Земли; погрешность относительно
    эллипсоида — до ~0.5%, что заведомо ниже потолка точности подложки.
    Про систематическое расхождение с :func:`ground_mpp` см. docstring модуля.
    Принимает скаляры или массивы numpy (с broadcasting).
    """
    scalar = all(np.isscalar(v) for v in (lat1, lon1, lat2, lon2))
    phi1 = np.radians(np.asarray(lat1, dtype=float))
    phi2 = np.radians(np.asarray(lat2, dtype=float))
    dphi = phi2 - phi1
    dlam = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    dist = 2.0 * EARTH_MEAN_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return _unwrap(dist, scalar)


@dataclass(frozen=True)
class Georef:
    """Привязка растра ``width × height`` к Web Mercator.

    Растр задаётся географическим центром, зумом и размером в пикселях —
    именно в таком виде его отдаёт сшивка XYZ-тайлов (``basemap.py``).

    Локальные координаты — «центр пикселя»: ``(0, 0)`` есть центр левого
    верхнего пикселя (см. модуль-docstring).

    Attributes:
        center_lon, center_lat: центр растра, градусы.
        zoom: уровень Web Mercator. Обычно целый (тайлы), но допускается
            дробный — для растров, полученных ресемплом.
        width, height: размер растра в пикселях.
    """

    center_lon: float
    center_lat: float
    zoom: float
    width: int
    height: int

    # Мировые координаты ЛЕВОГО ВЕРХНЕГО УГЛА растра (не центра пикселя).
    _origin_x: float = field(init=False, repr=False)
    _origin_y: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"размер растра должен быть > 0, получено {self.width}×{self.height}")
        if not -180.0 <= self.center_lon <= 180.0:
            raise ValueError(f"center_lon вне [-180, 180]: {self.center_lon}")
        if abs(self.center_lat) > MAX_LATITUDE:
            raise ValueError(f"center_lat вне предела Web Mercator: {self.center_lat}")

        cx, cy = lonlat_to_world_px(self.center_lon, self.center_lat, self.zoom)
        object.__setattr__(self, "_origin_x", cx - self.width / 2.0)
        object.__setattr__(self, "_origin_y", cy - self.height / 2.0)

    # --- служебные свойства -------------------------------------------------

    @property
    def origin_world_px(self) -> tuple[float, float]:
        """Мировые координаты левого верхнего УГЛА растра."""
        return (self._origin_x, self._origin_y)

    @property
    def center_pixel(self) -> tuple[float, float]:
        """Локальные координаты центра растра: ``((W-1)/2, (H-1)/2)``."""
        return ((self.width - 1) / 2.0, (self.height - 1) / 2.0)

    @property
    def mpp(self) -> float:
        """Разрешение растра [м/пиксель] на широте его центра."""
        return ground_mpp(self.center_lat, self.zoom)

    # --- основной мост пиксель ↔ координата ---------------------------------

    def pixel_to_world_px(self, px, py):
        """Локальные пиксели → непрерывные мировые пиксели Web Mercator."""
        # +0.5: локальные координаты — центры пикселей, мировые — от угла.
        return (
            np.asarray(px, dtype=float) + (self._origin_x + 0.5),
            np.asarray(py, dtype=float) + (self._origin_y + 0.5),
        )

    def world_px_to_pixel(self, wx, wy):
        """Непрерывные мировые пиксели Web Mercator → локальные пиксели."""
        return (
            np.asarray(wx, dtype=float) - (self._origin_x + 0.5),
            np.asarray(wy, dtype=float) - (self._origin_y + 0.5),
        )

    def pixel_to_lonlat(self, px, py):
        """Локальные пиксели растра → (lon, lat) в градусах.

        Принимает скаляры или массивы numpy.
        """
        scalar = np.isscalar(px) and np.isscalar(py)
        wx, wy = self.pixel_to_world_px(px, py)
        lon, lat = world_px_to_lonlat(wx, wy, self.zoom)
        return _unwrap(np.asarray(lon), scalar), _unwrap(np.asarray(lat), scalar)

    def lonlat_to_pixel(self, lon, lat):
        """(lon, lat) в градусах → локальные пиксели растра.

        Результат может выходить за границы растра — это допустимо и
        используется для проверки попадания в кадр.
        """
        scalar = np.isscalar(lon) and np.isscalar(lat)
        wx, wy = lonlat_to_world_px(lon, lat, self.zoom)
        px, py = self.world_px_to_pixel(wx, wy)
        return _unwrap(np.asarray(px), scalar), _unwrap(np.asarray(py), scalar)

    # --- производное --------------------------------------------------------

    def bounds(self) -> tuple[float, float, float, float]:
        """Габариты растра как ``(west, south, east, north)`` в градусах.

        Считается по внешним УГЛАМ растра, а не по центрам крайних пикселей.
        """
        west, north = world_px_to_lonlat(self._origin_x, self._origin_y, self.zoom)
        east, south = world_px_to_lonlat(
            self._origin_x + self.width, self._origin_y + self.height, self.zoom
        )
        return (west, south, east, north)

    def contains_pixel(self, px: float, py: float) -> bool:
        """Лежит ли локальная координата внутри растра."""
        return -0.5 <= px <= self.width - 0.5 and -0.5 <= py <= self.height - 0.5
