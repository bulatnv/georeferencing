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

from aero_geoloc.basemap import ESRI_WORLD_IMAGERY, TileCache, fetch_basemap  # noqa: E402
from aero_geoloc.geo import ground_mpp  # noqa: E402
from geom import valid_mask  # noqa: E402

NEUTRAL_GRAY = 114  # чем закрашивается невалидное (§5.10 задания)


@dataclass(frozen=True)
class Grid:
    """North-up сетка в CRS ортоплана: центр, размер в пикселях, метры/пиксель.

    ``x``/``y`` — координаты центра сетки; пиксель ``(0, 0)`` — центр левого
    верхнего пикселя (конвенция проекта).
    """

    x: float
    y: float
    size_px: int
    gsd: float

    @property
    def origin(self) -> tuple[float, float]:
        half = (self.size_px - 1) / 2.0 * self.gsd
        return self.x - half, self.y + half

    def pixel_centres(self):
        ox, oy = self.origin
        j = np.arange(self.size_px, dtype=np.float64)
        gx = ox + j * self.gsd
        gy = oy - j * self.gsd
        return np.meshgrid(gx, gy)

    def bounds(self):
        half = self.size_px / 2.0 * self.gsd
        return (self.x - half, self.y - half, self.x + half, self.y + half)


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
        rgb_src = np.transpose(arr[:3], (1, 2, 0))
        sx = out_w / win.width
        sy = out_h / win.height
        map_x = ((px - x0c) * sx).astype(np.float32)
        map_y = ((py - y0c) * sy).astype(np.float32)
        rgb = cv2.remap(rgb_src, map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        inside = ((map_x >= 0) & (map_x <= out_w - 1)
                  & (map_y >= 0) & (map_y <= out_h - 1))
        return rgb, valid_mask(rgb) & inside


def zoom_for_ground_mpp(lat: float, target_mpp: float, max_zoom: int = 19) -> int:
    """Зум, чей наземный MPP ближе всего к целевому (не грубее вдвое)."""
    best, best_err = max_zoom, float("inf")
    for z in range(1, max_zoom + 1):
        err = abs(np.log(ground_mpp(lat, z) / target_mpp))
        if err < best_err:
            best, best_err = z, err
    return best


class BasemapSource:
    """Подложка (Esri): кроп, перепроецированный в метрическую сетку ортоплана."""

    def __init__(self, ortho: OrthoSource, *, cache_dir: str = "tiles",
                 provider=ESRI_WORLD_IMAGERY):
        self.ortho = ortho
        self.cache = TileCache(cache_dir)
        self.provider = provider

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
            zoom = zoom_for_ground_mpp(lat0, grid.gsd, self.provider.max_zoom)

        # окно мозаики: габарит проекции сетки в пиксели тайлов + запас
        mpp = ground_mpp(lat0, zoom)
        span_m = grid.size_px * grid.gsd
        side = int(np.ceil(span_m / mpp)) + 16
        side = min(side, 4096)
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


def phase_shift(a: np.ndarray, b: np.ndarray, *, max_shift_px: float | None = None):
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
    return float(dx), float(dy), peak
