"""Загрузка и сшивка XYZ-тайлов подложки → георефренцированный растр.

Единственный модуль системы, которому нужна сеть (стадия «подложка из сети» в
``docs/PIPELINE.md``, фаза 2 из ``docs/PLAN.md``). Отвечает на один вопрос:
«дай растр ``width × height`` вокруг точки ``(lon, lat)`` на зуме ``z`` вместе с
его :class:`~aero_geoloc.geo.Georef`». Что делать с этим растром — матчить,
кропить, приводить масштаб — решают слои выше; basemap про матчинг и приоры
ничего не знает (правило зависимостей из ``docs/ARCHITECTURE.md``: basemap —
фундамент, зависит только от :mod:`aero_geoloc.geo`).

Почему целочисленный кроп, а не ресемпл
---------------------------------------
Мировые пиксели центра ``(lon, lat)`` дробные, поэтому окружить их растром так,
чтобы его пиксели попадали ровно на пиксели тайловой мозаики, в общем случае
нельзя без интерполяции. Мы **не интерполируем подложку**: вырезаем мозаику
целочисленно и строим :class:`Georef` от фактического (целочисленно
выровненного) центра окна. Фактический центр отстоит от запрошенного до
0.5 px по оси — ровно то же соглашение, что уже принято в
:meth:`Georef.crop_around` (решение №4 в ``docs/STATUS.md``). Так «эталонная»
подложка остаётся пиксель-в-пиксель настоящими тайлами, без замыленной
интерполяцией текстуры, а точную привязку всегда даёт ``pixel_to_lonlat``
возвращённого ``Georef`` — единственного моста «пиксель ↔ координата».

Кэш и офлайн
------------
Скачанные тайлы кладутся в дисковый кэш ``root/<provider>/<z>/<x>/<y>.<ext>``.
При ``allow_network=False`` сеть не трогается вообще: отсутствующий тайл — это
ошибка, а не тихий поход в интернет. Поэтому тесты работают на заранее
подготовленном кэше без сети (требование из ``docs/PLAN.md``).

Порядок координат тайла
-----------------------
Esri World Imagery отдаёт тайлы в порядке ``.../tile/{z}/{y}/{x}`` (сначала
строка, потом столбец), в отличие от «осмоподобного» ``{z}/{x}/{y}``. Порядок
инкапсулирован в шаблоне URL провайдера, поэтому наружу не протекает.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import cv2
import numpy as np

from .geo import (
    TILE_SIZE_PX,
    Georef,
    lonlat_to_world_px,
    world_px_to_lonlat,
    world_size_px,
)

__all__ = [
    "TileProvider",
    "ESRI_WORLD_IMAGERY",
    "PROVIDERS",
    "get_provider",
    "TileCache",
    "MissingTileError",
    "load_tile",
    "prefetch_tiles",
    "fetch_basemap",
    "BasemapSource",
    "TileBasemap",
]

#: User-Agent для запросов к тайл-серверам — вежливость и опознаваемость клиента.
USER_AGENT = "aero-geoloc/0.1 (+https://example.invalid; UAV visual localization)"


class MissingTileError(RuntimeError):
    """Тайла нет в кэше, а сеть запрещена (``allow_network=False``)."""


@dataclass(frozen=True)
class TileProvider:
    """Источник XYZ-тайлов.

    Attributes:
        name: короткое имя — им же назван подкаталог кэша.
        url_template: шаблон URL с полями ``{z}``, ``{x}``, ``{y}``. Порядок
            подстановки задаёт сам шаблон (Esri: ``{z}/{y}/{x}``).
        tile_size: размер тайла в пикселях (стандарт — 256).
        max_zoom: максимальный доступный уровень зума.
        tile_ext: расширение файла в кэше (формат, в котором сервер отдаёт тайлы).
        attribution: обязательная атрибуция источника.
    """

    name: str
    url_template: str
    tile_size: int = TILE_SIZE_PX
    max_zoom: int = 22
    tile_ext: str = "jpg"
    attribution: str = ""

    def tile_url(self, z: int, x: int, y: int) -> str:
        """URL тайла ``(z, x, y)`` — столбец ``x``, строка ``y``."""
        return self.url_template.format(z=z, x=x, y=y)


#: Esri World Imagery — спутниковая подложка по умолчанию (стадия 2 пайплайна).
ESRI_WORLD_IMAGERY = TileProvider(
    name="esri_world_imagery",
    url_template=(
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}"
    ),
    max_zoom=19,
    tile_ext="jpg",
    attribution="Esri, Maxar, Earthstar Geographics, and the GIS User Community",
)

#: Реестр провайдеров по имени — точка расширения под другие подложки.
PROVIDERS: dict[str, TileProvider] = {ESRI_WORLD_IMAGERY.name: ESRI_WORLD_IMAGERY}


def get_provider(provider: str | TileProvider) -> TileProvider:
    """Разрешить провайдера по имени либо вернуть переданный объект как есть."""
    if isinstance(provider, TileProvider):
        return provider
    if provider not in PROVIDERS:
        raise ValueError(f"неизвестный провайдер {provider!r}, доступны: {sorted(PROVIDERS)}")
    return PROVIDERS[provider]


# --- дисковый кэш тайлов -----------------------------------------------------


@dataclass(frozen=True)
class TileCache:
    """Дисковый кэш сырых байтов тайлов: ``root/<provider>/<z>/<x>/<y>.<ext>``.

    Хранятся именно байты в родном формате сервера (JPEG), а не декодированный
    массив: так кэш компактен и переносим, а декодирование — забота
    :func:`load_tile`.
    """

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    def path(self, provider: TileProvider, z: int, x: int, y: int) -> Path:
        """Путь к файлу тайла в кэше (без гарантии существования)."""
        return self.root / provider.name / str(z) / str(x) / f"{y}.{provider.tile_ext}"

    def get(self, provider: TileProvider, z: int, x: int, y: int) -> bytes | None:
        """Байты тайла из кэша либо ``None``, если его там нет."""
        p = self.path(provider, z, x, y)
        return p.read_bytes() if p.is_file() else None

    def put(self, provider: TileProvider, z: int, x: int, y: int, data: bytes) -> None:
        """Положить байты тайла в кэш, создав промежуточные каталоги."""
        p = self.path(provider, z, x, y)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


# --- загрузка одного тайла ---------------------------------------------------


#: Сколько тайлов тянуть параллельно. Загрузка — главная стоимость сборки карты
#: нового района (измерено: у одной карты 850 с из 897 ушло в сеть при ~46 с
#: вычислений, см. docs/OPTIMIZATION_PLAN.md). Потолок скромный: провайдер тайлов
#: — общий ресурс, и заваливать его сотней соединений невежливо и чревато баном.
TILE_WORKERS = 8

#: Повторы на тайл при сетевом сбое. Разовые обрывы у провайдера тайлов —
#: обычное дело (ловили ``SSL: handshake timeout``, который ронял весь прогон),
#: и ретрай на уровне тайла дешевле перезапуска сборки карты.
TILE_RETRIES = 2


def _http_get(url: str, *, timeout: float) -> bytes:
    """GET тайла через stdlib ``urllib`` — сетевой путь у тайлов тривиален."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (только http/https)
        return resp.read()


def _http_get_retrying(url: str, *, timeout: float, retries: int = TILE_RETRIES) -> bytes:
    """GET с повторами и нарастающей паузой — против разовых сетевых сбоев."""
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return _http_get(url, timeout=timeout)
        except (urllib.error.URLError, OSError) as exc:
            last = exc
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))
    raise last  # type: ignore[misc]


def prefetch_tiles(
    provider: TileProvider,
    zoom: int,
    tiles: Iterable[tuple[int, int]],
    *,
    cache: TileCache,
    timeout: float = 15.0,
    workers: int = TILE_WORKERS,
) -> int:
    """Скачать в кэш недостающие тайлы **параллельно**. Возвращает число скачанных.

    Зачем: сшивка окна тянет тайлы по одному, и на новом районе сборка карты почти
    целиком состоит из ожидания сети. Здесь недостающие тайлы качаются пачкой в
    пуле потоков (urllib отпускает GIL на время ввода-вывода), после чего сшивка
    находит всё в кэше и только декодирует.

    **Best-effort**: сбой отдельного тайла не поднимается наружу. Тайл, который не
    удалось скачать, останется отсутствующим, и его повторно и уже строго запросит
    :func:`load_tile` в сшивке — там же и родится понятная ошибка. Так поведение
    при недоступной сети остаётся ровно прежним, а префетч влияет только на скорость.
    """
    missing = [(tx, ty) for tx, ty in tiles if cache.get(provider, zoom, tx, ty) is None]
    if not missing:
        return 0

    def fetch(xy: tuple[int, int]) -> bool:
        tx, ty = xy
        try:
            data = _http_get_retrying(provider.tile_url(zoom, tx, ty), timeout=timeout)
        except Exception:  # noqa: BLE001 — см. best-effort в docstring
            return False
        cache.put(provider, zoom, tx, ty, data)
        return True

    if len(missing) == 1:  # ради одного тайла пул не заводим
        return int(fetch(missing[0]))
    with ThreadPoolExecutor(max_workers=min(workers, len(missing))) as pool:
        return sum(pool.map(fetch, missing))


def _decode_tile(data: bytes, provider: TileProvider) -> np.ndarray:
    """Декодировать байты тайла в BGR-массив ``tile_size × tile_size × 3``."""
    arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("не удалось декодировать тайл (повреждённые данные?)")
    ts = provider.tile_size
    if arr.shape[0] != ts or arr.shape[1] != ts:
        raise ValueError(
            f"размер тайла {arr.shape[1]}×{arr.shape[0]} не равен ожидаемому {ts}×{ts}"
        )
    return arr


def load_tile(
    provider: TileProvider,
    z: int,
    x: int,
    y: int,
    *,
    cache: TileCache | None = None,
    allow_network: bool = True,
    timeout: float = 15.0,
) -> np.ndarray:
    """Тайл ``(z, x, y)`` как BGR-массив: сначала кэш, потом (опц.) сеть.

    Args:
        cache: дисковый кэш; если задан — читаем из него и туда же кладём скачанное.
        allow_network: разрешён ли поход в сеть при промахе кэша. ``False`` —
            отсутствующий тайл поднимает :class:`MissingTileError`, а не
            молчаливо идёт в интернет (режим офлайн-тестов).
        timeout: таймаут HTTP-запроса, секунды.
    """
    data = cache.get(provider, z, x, y) if cache is not None else None
    if data is None:
        if not allow_network:
            raise MissingTileError(
                f"тайл {provider.name} z{z}/{x}/{y} отсутствует в кэше, а сеть запрещена"
            )
        try:
            # С повторами: разовый обрыв не должен ронять сборку целой карты.
            data = _http_get_retrying(provider.tile_url(z, x, y), timeout=timeout)
        except (urllib.error.URLError, OSError) as exc:  # сеть недоступна/сервер ответил ошибкой
            raise RuntimeError(
                f"не удалось скачать тайл {provider.name} z{z}/{x}/{y}: {exc}"
            ) from exc
        if cache is not None:
            cache.put(provider, z, x, y, data)
    return _decode_tile(data, provider)


# --- сшивка окна подложки ----------------------------------------------------


def fetch_basemap(
    center_lon: float,
    center_lat: float,
    zoom: int,
    width: int,
    height: int,
    *,
    provider: str | TileProvider = ESRI_WORLD_IMAGERY,
    cache: TileCache | None = None,
    allow_network: bool = True,
    timeout: float = 15.0,
    workers: int = TILE_WORKERS,
) -> tuple[np.ndarray, Georef]:
    """Растр ``width × height`` вокруг ``(center_lon, center_lat)`` на зуме ``zoom``.

    Сшивает покрывающие окно тайлы и вырезает из мозаики **целочисленное** окно
    (без ресемпла подложки, см. модуль-docstring). Возвращаемый ``Georef``
    построен от фактического центра окна, поэтому его пиксели соответствуют
    пикселям растра точно, а не приближённо.

    Args:
        zoom: целый уровень XYZ-пирамиды (обычно из :func:`~aero_geoloc.geo.zoom_for_mpp`).
        width, height: размер растра в пикселях.
        provider: имя провайдера или объект :class:`TileProvider`.
        cache, allow_network, timeout: см. :func:`load_tile`.
        workers: сколько недостающих тайлов качать параллельно (см.
            :func:`prefetch_tiles`). Действует только при заданном ``cache``.

    Returns:
        Кортеж ``(image_bgr, georef)``: BGR-растр ``height × width × 3`` uint8 и
        его привязка.
    """
    provider = get_provider(provider)
    zoom = int(zoom)
    if width <= 0 or height <= 0:
        raise ValueError(f"размер растра должен быть > 0, получено {width}×{height}")
    if not 0 <= zoom <= provider.max_zoom:
        raise ValueError(f"zoom={zoom} вне диапазона провайдера [0, {provider.max_zoom}]")

    ts = provider.tile_size
    world_px = world_size_px(zoom)

    # Центр окна в непрерывных мировых пикселях и целочисленный левый верхний угол.
    cx, cy = lonlat_to_world_px(center_lon, center_lat, zoom)
    x0 = int(round(cx - width / 2.0))
    y0 = int(round(cy - height / 2.0))

    # Окно целиком должно лежать внутри мира подложки: у края пирамиды тайлов
    # для добивки нет, и молча вернуть чёрные поля хуже явной ошибки.
    if x0 < 0 or y0 < 0 or x0 + width > world_px or y0 + height > world_px:
        raise ValueError(
            f"окно {width}×{height} в мировых px ({x0}, {y0}) выходит за границы "
            f"подложки на зуме {zoom} (размер мира {int(world_px)} px)"
        )

    # Диапазон тайлов, покрывающих окно [x0, x0+width) × [y0, y0+height).
    tx_min, tx_max = x0 // ts, (x0 + width - 1) // ts
    ty_min, ty_max = y0 // ts, (y0 + height - 1) // ts

    # Недостающие тайлы окна качаем пачкой ДО сшивки: последовательная загрузка —
    # главная стоимость нового района. Если кэша нет, складывать скачанное некуда,
    # и префетч смысла не имеет — тогда работает прежний путь по одному тайлу.
    if allow_network and cache is not None:
        prefetch_tiles(
            provider,
            zoom,
            ((tx, ty) for ty in range(ty_min, ty_max + 1) for tx in range(tx_min, tx_max + 1)),
            cache=cache,
            timeout=timeout,
            workers=workers,
        )

    # Мозаика из целых тайлов; её левый верхний угол — в мировых px (tx_min·ts, ty_min·ts).
    mosaic = np.empty(((ty_max - ty_min + 1) * ts, (tx_max - tx_min + 1) * ts, 3), dtype=np.uint8)
    for ty in range(ty_min, ty_max + 1):
        for tx in range(tx_min, tx_max + 1):
            tile = load_tile(
                provider, zoom, tx, ty, cache=cache, allow_network=allow_network, timeout=timeout
            )
            oy, ox = (ty - ty_min) * ts, (tx - tx_min) * ts
            mosaic[oy : oy + ts, ox : ox + ts] = tile

    # Целочисленный кроп окна из мозаики.
    cut_x, cut_y = x0 - tx_min * ts, y0 - ty_min * ts
    image = mosaic[cut_y : cut_y + height, cut_x : cut_x + width].copy()

    # Georef от ФАКТИЧЕСКОГО центра окна (целочисленно выровненного).
    actual_lon, actual_lat = world_px_to_lonlat(x0 + width / 2.0, y0 + height / 2.0, zoom)
    georef = Georef(
        center_lon=float(actual_lon),
        center_lat=float(actual_lat),
        zoom=zoom,
        width=width,
        height=height,
    )
    return image, georef


# --- источник подложки для оркестрации ---------------------------------------


class BasemapSource(Protocol):
    """Откуда ``localize`` берёт окно подложки — сменный источник.

    Абстракция ровно та же по духу, что и сменный матчер: оркестрация не должна
    зависеть от того, тянется ли подложка из сети (:class:`TileBasemap`) или
    режется из уже данного растра (стенд). Сигнатура повторяет
    :func:`fetch_basemap`: «дай растр ``width × height`` вокруг ``(lon, lat)`` на
    зуме ``zoom`` вместе с его :class:`~aero_geoloc.geo.Georef`».
    """

    def __call__(
        self, center_lon: float, center_lat: float, zoom: int, width: int, height: int
    ) -> tuple[np.ndarray, Georef]:
        ...


@dataclass(frozen=True)
class TileBasemap:
    """:class:`BasemapSource` поверх XYZ-тайлов через :func:`fetch_basemap`.

    Хранит провайдера, кэш и политику сети, чтобы оркестрация вызывала источник
    единообразно, не зная про тайлы. Промах кэша при ``allow_network=False``
    поднимает :class:`MissingTileError`.
    """

    provider: str | TileProvider = ESRI_WORLD_IMAGERY
    cache: TileCache | None = None
    allow_network: bool = True
    timeout: float = 15.0

    def __call__(
        self, center_lon: float, center_lat: float, zoom: int, width: int, height: int
    ) -> tuple[np.ndarray, Georef]:
        return fetch_basemap(
            center_lon,
            center_lat,
            zoom,
            width,
            height,
            provider=self.provider,
            cache=self.cache,
            allow_network=self.allow_network,
            timeout=self.timeout,
        )
