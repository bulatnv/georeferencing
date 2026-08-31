"""Чтение сторон пары в общей метрической сетке и замер сдвига привязки.

Обе стороны приводятся к **одной north-up сетке в CRS ортофотоплана** (UTM):
у ортоплана это ресемпл своего растра, у подложки — мозаика тайлов
веб-Меркатора, перепроецированная в ту же сетку. Так warp между сторонами
становится метрическим, а не «через две системы координат».

Ловушка, снятая здесь по §2.7 задания: наземный MPP тайла на широте φ равен
``156543.034·cos(φ)/2^z`` — на широте 60° это вдвое мельче номинала. Зум
подбирается по наземному MPP, иначе подложка приходит вдвое грубее ожидания.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.windows import Window

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))              # соседние модули (geom)
sys.path.insert(0, str(_HERE.parents[2]))          # корень проекта: aero_geoloc

from aero_geoloc.basemap import (  # noqa: E402
    ESRI_WORLD_IMAGERY,
    TileCache,
    deepest_imagery_zoom,
    fetch_basemap,
)
from aero_geoloc.geo import ground_mpp  # noqa: E402
from geom import valid_mask  # noqa: E402

NEUTRAL_GRAY = 114  # чем закрашивается невалидное (§5.10 задания)


@dataclass(frozen=True)
class Grid:
    """Сетка в CRS ортоплана: центр, размер в пикселях, метры/пиксель, поворот.

    ``x``/``y`` — координаты центра сетки; пиксель ``(0, 0)`` — центр левого
    верхнего пикселя (конвенция проекта). ``rot_deg`` — курс сетки: азимут,
    в который смотрит её «верх» (0 = север, растёт по часовой, как yaw кадра).
    При ``rot_deg = 0`` сетка north-up.
    """

    x: float
    y: float
    size_px: int
    gsd: float
    rot_deg: float = 0.0

    @property
    def axes(self):
        """Мировые направления осей сетки: (вправо, вверх)."""
        a = math.radians(self.rot_deg)
        ca, sa = math.cos(a), math.sin(a)
        return (ca, -sa), (sa, ca)

    @property
    def origin(self) -> tuple[float, float]:
        """Мировые координаты центра пикселя (0, 0)."""
        half = (self.size_px - 1) / 2.0 * self.gsd
        right, up = self.axes
        return (self.x - half * right[0] + half * up[0],
                self.y - half * right[1] + half * up[1])

    def pixel_centres(self):
        """Мировые координаты центров всех пикселей: (gx, gy), формы (n, n)."""
        n = self.size_px
        half = (n - 1) / 2.0
        j, i = np.meshgrid(np.arange(n, dtype=np.float64), np.arange(n, dtype=np.float64))
        u = (j - half) * self.gsd          # вправо по сетке, метры
        v = (half - i) * self.gsd          # вверх по сетке, метры
        right, up = self.axes
        gx = self.x + u * right[0] + v * up[0]
        gy = self.y + u * right[1] + v * up[1]
        return gx, gy

    def pixel_from_world(self, gx, gy):
        """Мировые координаты → пиксели сетки (обратно к :meth:`pixel_centres`)."""
        dx = np.asarray(gx, dtype=np.float64) - self.x
        dy = np.asarray(gy, dtype=np.float64) - self.y
        right, up = self.axes
        u = dx * right[0] + dy * right[1]
        v = dx * up[0] + dy * up[1]
        half = (self.size_px - 1) / 2.0
        return half + u / self.gsd, half - v / self.gsd

    def corners_world(self):
        """Углы сетки в мировых координатах: (0,0), (n−1,0), (n−1,n−1), (0,n−1)."""
        n = self.size_px
        half = (n - 1) / 2.0 * self.gsd
        right, up = self.axes
        out = []
        for su, sv in ((-1, 1), (1, 1), (1, -1), (-1, -1)):
            out.append((self.x + su * half * right[0] + sv * half * up[0],
                        self.y + su * half * right[1] + sv * half * up[1]))
        return np.array(out)

    def bounds(self):
        """Габаритный axis-aligned прямоугольник (для north-up — точный)."""
        c = self.corners_world()
        return (float(c[:, 0].min()), float(c[:, 1].min()),
                float(c[:, 0].max()), float(c[:, 1].max()))


class OrthoSource:
    """Ортофотоплан: чтение произвольного окна в заданной метрической сетке."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.ds = rasterio.open(self.path)
        self.res_x = abs(self.ds.transform.a)
        self.res_y = abs(self.ds.transform.e)
        self.to_wgs = Transformer.from_crs(self.ds.crs, "EPSG:4326", always_xy=True)
        self.from_wgs = Transformer.from_crs("EPSG:4326", self.ds.crs, always_xy=True)

    def close(self):
        self.ds.close()

    @property
    def bounds(self):
        return self.ds.bounds

    def centre_lonlat(self):
        b = self.ds.bounds
        return self.to_wgs.transform((b.left + b.right) / 2, (b.bottom + b.top) / 2)

    def world_to_pixel(self, gx, gy):
        inv = ~self.ds.transform
        px, py = inv * (np.asarray(gx), np.asarray(gy))
        return np.asarray(px) - 0.5, np.asarray(py) - 0.5   # → пиксель-центр

    def read_grid(self, grid: Grid, *, margin_px: int = 4):
        """Окно ортоплана в сетке ``grid``: (rgb, valid).

        Читается прямоугольник растра, накрывающий сетку (с запасом), в
        разрешении не мельче нужного — дальше билинейный ресемпл в сетку.
        Пустой пересечение с растром даёт полностью невалидный результат.
        """
        gxs, gys = grid.pixel_centres()
        px, py = self.world_to_pixel(gxs, gys)
        x0 = int(np.floor(np.nanmin(px))) - margin_px
        y0 = int(np.floor(np.nanmin(py))) - margin_px
        x1 = int(np.ceil(np.nanmax(px))) + margin_px
        y1 = int(np.ceil(np.nanmax(py))) + margin_px
        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(self.ds.width, x1), min(self.ds.height, y1)
        if x1c <= x0c or y1c <= y0c:
            empty = np.zeros((grid.size_px, grid.size_px, 3), np.uint8)
            return empty, np.zeros((grid.size_px, grid.size_px), bool)

        win = Window(x0c, y0c, x1c - x0c, y1c - y0c)
        # decimated read: не тянем полный масштаб, если сетка грубее растра
        scale = max(1.0, grid.gsd / self.res_x)
        out_w = max(1, int(round(win.width / scale)))
        out_h = max(1, int(round(win.height / scale)))
        arr = self.ds.read(out_shape=(self.ds.count, out_h, out_w), window=win,
                           resampling=Resampling.average if scale > 1.5 else Resampling.bilinear)
        rgb_src = to_rgb(arr)
        sx = out_w / win.width
        sy = out_h / win.height
        map_x = ((px - x0c) * sx).astype(np.float32)
        map_y = ((py - y0c) * sy).astype(np.float32)
        rgb = cv2.remap(rgb_src, map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        inside = ((map_x >= 0) & (map_x <= out_w - 1)
                  & (map_y >= 0) & (map_y <= out_h - 1))
        return rgb, valid_mask(rgb) & inside


def to_rgb(arr: np.ndarray) -> np.ndarray:
    """(каналы, H, W) от rasterio → (H, W, 3) uint8-совместимый массив.

    Число каналов у источника — его частное дело, выше по стеку картинка
    всегда трёхканальная: маска валидности берёт размах по оси каналов, и на
    одноканальном растре она обваливалась несовпадением форм. Панхром
    разворачивается в три одинаковых канала (а не дополняется нулями, иначе
    кадр стал бы синим), у четырёхканальных берутся первые три.
    """
    if arr.shape[0] == 1:
        out = np.repeat(np.transpose(arr[:1], (1, 2, 0)), 3, axis=2)
    else:
        out = np.transpose(arr[:3], (1, 2, 0))
    return np.ascontiguousarray(out)


class OrthoAsBase:
    """Второй ортоплан в роли стороны B — интерфейс тот же, что у подложки.

    Нужен для кросс-датных пар: одна и та же территория, снятая в разные даты
    двумя вылетами. Всё, что выше по стеку (замер привязки, генератор, гейты),
    работает с «подложкой» через `read_grid(grid, zoom, shift_m)` и `min_mpp` —
    поэтому достаточно подменить источник, не трогая логику.

    Отличие от настоящей подложки: зум не при чём, разрешение фиксировано
    разрешением растра, а сдвиг привязки между двумя вылетами обычно меньше,
    чем между вылетом и спутниковой мозаикой, — но не ноль, и мерить его надо
    так же.
    """

    def __init__(self, ortho: "OrthoSource"):
        self.ortho = ortho
        self.path = ortho.path

    @property
    def min_mpp(self) -> float:
        return float(self.ortho.res_x)

    def read_grid(self, grid: Grid, *, zoom: int | None = None,
                  shift_m: tuple[float, float] = (0.0, 0.0)):
        """Окно второго ортоплана в сетке, со сдвигом привязки."""
        shifted = Grid(x=grid.x + shift_m[0], y=grid.y + shift_m[1],
                       size_px=grid.size_px, gsd=grid.gsd, rot_deg=grid.rot_deg)
        rgb, valid = self.ortho.read_grid(shifted)
        return rgb, valid, {"zoom": None, "source": "ortho", "path": str(self.path)}

    def close(self):
        self.ortho.close()


def zoom_for_ground_mpp(lat: float, target_mpp: float, max_zoom: int = 19) -> int:
    """Зум, чей наземный MPP ближе всего к целевому (не грубее вдвое)."""
    best, best_err = max_zoom, float("inf")
    for z in range(1, max_zoom + 1):
        err = abs(np.log(ground_mpp(lat, z) / target_mpp))
        if err < best_err:
            best, best_err = z, err
    return best


class NoImageryError(RuntimeError):
    """У провайдера нет съёмки в этом районе ни на одном пригодном зуме."""


class BasemapSource:
    """Подложка (Esri): кроп, перепроецированный в метрическую сетку ортоплана.

    Перед первым чтением выясняет, **до какого зума в этом районе есть
    реальная съёмка**: `max_zoom` провайдера — предел пирамиды, а не гарантия
    покрытия, и вне городов Esri отдаёт серую заглушку «Map data not yet
    available». На такой заглушке фазовая корреляция даёт случайные пики, и
    сдвиги привязки читались как 14–36 м (замерено на площадках Манитобы,
    где z19 — заглушка, а съёмка есть только до z18). Если съёмки нет и на
    минимальном зуме — :class:`NoImageryError`, а не пары по чистому листу.
    """

    def __init__(self, ortho: OrthoSource, *, cache_dir: str = "tiles",
                 provider=ESRI_WORLD_IMAGERY, min_zoom: int = 14):
        self.ortho = ortho
        self.cache = TileCache(cache_dir)
        self.provider = provider
        self.min_zoom = min_zoom
        self._max_zoom = None          # предел с реальной съёмкой, лениво
        self.probes = []

    @property
    def max_zoom(self) -> int:
        """Самый детальный зум со съёмкой в районе растра (кэшируется)."""
        if self._max_zoom is None:
            lon, lat = self.ortho.centre_lonlat()
            b = self.ortho.bounds
            radius = max(b.right - b.left, b.top - b.bottom) / 2
            zoom, probes = deepest_imagery_zoom(
                lon, lat, radius_m=radius, max_zoom=self.provider.max_zoom,
                min_zoom=self.min_zoom, cache=self.cache)
            self.probes = probes
            if zoom is None:
                raise NoImageryError(
                    f"у {self.provider.name} нет съёмки в районе "
                    f"lat={lat:.4f} lon={lon:.4f} вплоть до zoom {self.min_zoom}")
            self._max_zoom = zoom
        return self._max_zoom

    @property
    def min_mpp(self) -> float:
        """Наземный размер пикселя на самом детальном доступном зуме."""
        lat = self.ortho.centre_lonlat()[1]
        return ground_mpp(lat, self.max_zoom)

    def read_grid(self, grid: Grid, *, zoom: int | None = None,
                  shift_m: tuple[float, float] = (0.0, 0.0)):
        """Кроп подложки в сетке ``grid``: (rgb, valid, info).

        ``shift_m`` — компенсация привязки: точка земли с координатой ``g``
        ортоплана ищется в подложке в точке ``g + shift`` (семантика знака
        закреплена тестом, §4 задания).
        """
        gxs, gys = grid.pixel_centres()
        lon, lat = self.ortho.to_wgs.transform(gxs + shift_m[0], gys + shift_m[1])
        lat0 = float(np.nanmedian(lat))
        lon0 = float(np.nanmedian(lon))
        if zoom is None:
            zoom = zoom_for_ground_mpp(lat0, grid.gsd, self.max_zoom)

        # Окно мозаики — по ГАБАРИТУ сетки, а не по её стороне: повёрнутая
        # сетка занимает в мире прямоугольник до √2 раз шире, и по стороне
        # её углы вылезали за мозаику (в кропе появлялись серые треугольники).
        mpp = ground_mpp(lat0, zoom)
        x0, y0, x1, y1 = grid.bounds()
        span_m = max(x1 - x0, y1 - y0) + 2 * grid.gsd
        side = int(np.ceil(span_m / mpp)) + 16
        side = min(side, 6144)
        img_bgr, gref = fetch_basemap(lon0, lat0, zoom, side, side,
                                      provider=self.provider, cache=self.cache)
        rgb_src = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        px, py = gref.lonlat_to_pixel(lon, lat)
        map_x = np.asarray(px, np.float32)
        map_y = np.asarray(py, np.float32)
        rgb = cv2.remap(rgb_src, map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        inside = ((map_x >= 0) & (map_x <= side - 1)
                  & (map_y >= 0) & (map_y <= side - 1))
        return rgb, valid_mask(rgb) & inside, {"zoom": zoom, "mpp": mpp,
                                               "lat": lat0, "lon": lon0}


# --- замер сдвига привязки -----------------------------------------------------

def gradient_map(rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Градиентная карта для корреляции: Собель, клип по 99-му перцентилю,
    нормировка, зануление невалидного (§4.3 задания). Сравнивать сырые яркости
    нельзя — у сторон разный тон, сезон и обработка."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    mag[~valid] = 0.0
    hi = np.percentile(mag[valid], 99) if valid.any() else 1.0
    if hi <= 0:
        return np.zeros_like(mag)
    return np.clip(mag / hi, 0.0, 1.0) * valid


def _parabolic(prev: float, mid: float, nxt: float) -> float:
    """Смещение вершины параболы по трём отсчётам (−1, 0, +1)."""
    denom = prev - 2.0 * mid + nxt
    if abs(denom) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (prev - nxt) / denom, -0.5, 0.5))


def peak_to_sidelobe(corr: np.ndarray, iy: int, ix: int, exclude_px: int = 3) -> float:
    """Насколько пик возвышается над фоном корреляции, в сигмах фона.

    Абсолютная высота пика плохой гейт: она зависит от контраста фактуры, а
    не от того, нашлось ли соответствие. На смене фактуры (орто зимой против
    летней подложки) корреляция вырождается в шум, но её максимум всё равно
    имеет высоту порядка обычной — и замер молча выдаёт мусор вместо отказа.
    Отношение «пик над фоном» этот случай различает: у настоящего совпадения
    пик стоит одиноко, у шума — теряется среди соседей.
    """
    h, w = corr.shape
    yy = np.minimum(np.abs(np.arange(h) - iy), h - np.abs(np.arange(h) - iy))
    xx = np.minimum(np.abs(np.arange(w) - ix), w - np.abs(np.arange(w) - ix))
    far = (yy[:, None] > exclude_px) | (xx[None, :] > exclude_px)
    bg = corr[far]
    sd = float(bg.std())
    if sd < 1e-12:
        return 0.0
    return float((corr[iy, ix] - bg.mean()) / sd)


def phase_shift(a: np.ndarray, b: np.ndarray, *, max_shift_px: float | None = None,
                with_psr: bool = False):
    """Сдвиг ``b`` относительно ``a`` в пикселях: (dx, dy, peak).

    ``max_shift_px`` — радиус, в котором ищется пик. Это **ограничение, а не
    подсказка** (инвариант проекта): вне радиуса решение не «подтягивается»,
    оно просто не рассматривается — далёкий ложный пик на повторяющейся
    городской фактуре иначе выигрывает у верного.

    Знак: если ``b`` получена из ``a`` сдвигом на (dx, dy) вправо/вниз, метод
    возвращает именно (dx, dy) — закреплено тестом V1.

    Своя реализация, а не ``cv2.phaseCorrelate``: её субпиксельное уточнение
    (взвешенный центроид) на гладких текстурах систематически врёт — замерено,
    сдвиг 0.5 px читался как 1.3. Здесь пик уточняется параболой по трём
    точкам (§4.3 задания), а его высота возвращается как гейт качества
    замера (§7.1: на сплошном лесу корреляции не за что зацепиться).
    """
    if a.shape != b.shape:
        raise ValueError(f"формы не совпадают: {a.shape} vs {b.shape}")
    h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    fa = np.fft.fft2(a.astype(np.float64) * win)
    fb = np.fft.fft2(b.astype(np.float64) * win)
    cross = fb * np.conj(fa)
    mag = np.abs(cross)
    mag[mag < 1e-12] = 1e-12
    corr = np.real(np.fft.ifft2(cross / mag))
    if max_shift_px is not None:
        r = int(min(max(1, round(max_shift_px)), min(h, w) // 2 - 1))
        search = np.full_like(corr, -np.inf)
        search[:r + 1, :r + 1] = corr[:r + 1, :r + 1]          # сдвиги 0…+r
        search[:r + 1, -r:] = corr[:r + 1, -r:]                # отрицательные по x
        search[-r:, :r + 1] = corr[-r:, :r + 1]                # отрицательные по y
        search[-r:, -r:] = corr[-r:, -r:]
        iy, ix = np.unravel_index(int(np.argmax(search)), corr.shape)
    else:
        iy, ix = np.unravel_index(int(np.argmax(corr)), corr.shape)
    peak = float(corr[iy, ix])
    dx = ix + _parabolic(corr[iy, (ix - 1) % w], peak, corr[iy, (ix + 1) % w])
    dy = iy + _parabolic(corr[(iy - 1) % h, ix], peak, corr[(iy + 1) % h, ix])
    if dx > w / 2:
        dx -= w
    if dy > h / 2:
        dy -= h
    if with_psr:
        return float(dx), float(dy), peak, peak_to_sidelobe(corr, iy, ix)
    return float(dx), float(dy), peak
