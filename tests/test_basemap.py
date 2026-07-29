"""Тесты сшивки и георефренцирования подложки — офлайн, без сети.

Идея провенанса: синтетический тайл ``(z, tx, ty)`` заполнен функцией от
**мировых** пикселей, ``f(X, Y)``. Тогда сшитая мозаика воспроизводит ``f`` над
всем покрытым участком мира, а вырезанное окно обязано точно совпасть с
``f(x0+i, y0+j)``. Так один массив-эталон проверяет сразу и стыковку тайлов, и
целочисленное смещение кропа — ошибка в один пиксель по любой оси всплывёт.

Сеть в тестах по умолчанию не трогается; реальный тайл Esri тянется только при
``AERO_GEOLOC_NETWORK_TESTS=1``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from aero_geoloc.basemap import (
    ESRI_WORLD_IMAGERY,
    MissingTileError,
    TileCache,
    TileProvider,
    fetch_basemap,
    get_provider,
    load_tile,
)
from aero_geoloc.geo import haversine_m, lonlat_to_world_px

TS = 256

#: Провайдер для офлайн-тестов: PNG (без потерь) и фиктивный URL, чтобы промах
#: кэша при allow_network=False был явной ошибкой, а не походом в сеть.
FAKE_PROVIDER = TileProvider(
    name="fake_lossless",
    url_template="http://localhost.invalid/{z}/{x}/{y}",
    tile_ext="png",
)


def world_field(x0: int, y0: int, width: int, height: int) -> np.ndarray:
    """BGR-эталон ``height × width``: детерминированная функция мировых пикселей.

    Множители подобраны так, что сдвиг на один пиксель по X или Y заметно меняет
    значение — тест чувствителен к ошибкам выравнивания.
    """
    xs = x0 + np.arange(width, dtype=np.int64)[None, :]
    ys = y0 + np.arange(height, dtype=np.int64)[:, None]
    xs = np.broadcast_to(xs, (height, width))
    ys = np.broadcast_to(ys, (height, width))
    b = ((xs + ys) * 17) % 256
    g = (ys * 131) % 256
    r = (xs * 131) % 256
    return np.stack([b, g, r], axis=-1).astype(np.uint8)


def synth_tile(tx: int, ty: int) -> np.ndarray:
    """Тайл, согласованный с :func:`world_field` (тот же участок мира)."""
    return world_field(tx * TS, ty * TS, TS, TS)


def populate_cache(cache: TileCache, zoom: int, tx_range, ty_range) -> None:
    """Заполнить кэш синтетическими тайлами в заданном диапазоне (PNG, без потерь)."""
    import cv2

    for ty in ty_range:
        for tx in tx_range:
            ok, buf = cv2.imencode(".png", synth_tile(tx, ty))
            assert ok
            cache.put(FAKE_PROVIDER, zoom, tx, ty, buf.tobytes())


# --- провайдер и кэш --------------------------------------------------------


def test_esri_url_order_is_z_y_x():
    # Esri: .../tile/{z}/{y}/{x} — строка перед столбцом, в отличие от OSM.
    url = ESRI_WORLD_IMAGERY.tile_url(18, x=5, y=7)
    assert url.endswith("/18/7/5")


def test_get_provider_by_name_and_object():
    assert get_provider("esri_world_imagery") is ESRI_WORLD_IMAGERY
    assert get_provider(ESRI_WORLD_IMAGERY) is ESRI_WORLD_IMAGERY
    with pytest.raises(ValueError, match="неизвестный провайдер"):
        get_provider("no_such_provider")


def test_tile_cache_roundtrip(tmp_path):
    cache = TileCache(tmp_path)
    assert cache.get(FAKE_PROVIDER, 18, 1, 2) is None
    cache.put(FAKE_PROVIDER, 18, 1, 2, b"payload")
    assert cache.get(FAKE_PROVIDER, 18, 1, 2) == b"payload"
    # Раскладка каталогов: <provider>/<z>/<x>/<y>.<ext>.
    assert cache.path(FAKE_PROVIDER, 18, 1, 2).match("fake_lossless/18/1/2.png")


def test_load_tile_from_cache_offline(tmp_path):
    cache = TileCache(tmp_path)
    populate_cache(cache, 18, [10], [20])
    tile = load_tile(FAKE_PROVIDER, 18, 10, 20, cache=cache, allow_network=False)
    np.testing.assert_array_equal(tile, synth_tile(10, 20))


def test_load_tile_missing_offline_raises(tmp_path):
    cache = TileCache(tmp_path)
    with pytest.raises(MissingTileError):
        load_tile(FAKE_PROVIDER, 18, 0, 0, cache=cache, allow_network=False)


# --- сшивка и георефренцирование --------------------------------------------


def _prepare(center_lon, center_lat, zoom, width, height, tmp_path):
    """Подготовить кэш под окно и вернуть его вместе с эталонной геометрией."""
    cx, cy = lonlat_to_world_px(center_lon, center_lat, zoom)
    x0 = int(round(cx - width / 2.0))
    y0 = int(round(cy - height / 2.0))
    tx_range = range(x0 // TS, (x0 + width - 1) // TS + 1)
    ty_range = range(y0 // TS, (y0 + height - 1) // TS + 1)
    cache = TileCache(tmp_path)
    populate_cache(cache, zoom, tx_range, ty_range)
    return cache, x0, y0


def test_fetch_basemap_stitch_and_crop_are_pixel_exact(tmp_path):
    lon, lat, zoom, w, h = 37.6173, 55.7558, 18, 512, 384
    cache, x0, y0 = _prepare(lon, lat, zoom, w, h, tmp_path)

    image, georef = fetch_basemap(
        lon, lat, zoom, w, h, provider=FAKE_PROVIDER, cache=cache, allow_network=False
    )

    assert image.shape == (h, w, 3)
    # Окно вырезано целочисленно из мозаики → совпадает с эталоном мира пиксель-в-пиксель.
    np.testing.assert_array_equal(image, world_field(x0, y0, w, h))


def test_fetch_basemap_georef_is_integer_aligned(tmp_path):
    lon, lat, zoom, w, h = 37.6173, 55.7558, 18, 512, 512
    cache, x0, y0 = _prepare(lon, lat, zoom, w, h, tmp_path)

    _, georef = fetch_basemap(
        lon, lat, zoom, w, h, provider=FAKE_PROVIDER, cache=cache, allow_network=False
    )

    # Левый верхний угол растра — ровно целочисленный мировой пиксель (x0, y0).
    ox, oy = georef.origin_world_px
    assert ox == float(x0)
    assert oy == float(y0)
    assert georef.width == w and georef.height == h
    assert georef.zoom == zoom


def test_fetch_basemap_center_matches_request_within_half_pixel(tmp_path):
    lon, lat, zoom, w, h = 37.6173, 55.7558, 18, 512, 512
    cache, _, _ = _prepare(lon, lat, zoom, w, h, tmp_path)

    _, georef = fetch_basemap(
        lon, lat, zoom, w, h, provider=FAKE_PROVIDER, cache=cache, allow_network=False
    )

    got_lon, got_lat = georef.pixel_to_lonlat(*georef.center_pixel)
    # Фактический центр отстоит от запрошенного не больше чем на ~0.5 px (целочисленный
    # кроп чётного окна). При mpp ≈ 0.6 м/px это доли метра.
    assert haversine_m(lat, lon, got_lat, got_lon) < georef.mpp


def test_fetch_basemap_out_of_world_raises(tmp_path):
    # На зуме 0 весь мир — 256 px; окно 300 px в него не влезает.
    cache = TileCache(tmp_path)
    with pytest.raises(ValueError, match="выходит за границы"):
        fetch_basemap(0.0, 0.0, 0, 300, 300, provider=FAKE_PROVIDER, cache=cache, allow_network=False)


def test_fetch_basemap_missing_tile_offline_raises(tmp_path):
    # Кэш пуст, сеть запрещена → честная ошибка вместо тихого похода в интернет.
    cache = TileCache(tmp_path)
    with pytest.raises(MissingTileError):
        fetch_basemap(
            37.6173, 55.7558, 18, 256, 256, provider=FAKE_PROVIDER, cache=cache, allow_network=False
        )


def test_fetch_basemap_zoom_out_of_provider_range(tmp_path):
    cache = TileCache(tmp_path)
    with pytest.raises(ValueError, match="вне диапазона провайдера"):
        fetch_basemap(
            37.6173, 55.7558, 25, 256, 256, provider=ESRI_WORLD_IMAGERY, cache=cache
        )


# --- параллельная загрузка тайлов (O1) ---------------------------------------


def test_prefetch_downloads_only_missing_tiles(tmp_path, monkeypatch):
    """Префетч качает лишь недостающее: тайлы из кэша сеть не трогают."""
    import aero_geoloc.basemap as bm

    cache = TileCache(tmp_path)
    populate_cache(cache, 10, range(0, 2), range(0, 1))  # (0,0) и (1,0) уже есть
    asked: list[str] = []

    def fake_get(url, *, timeout, retries=2):
        asked.append(url)
        import cv2
        ok, buf = cv2.imencode(".png", synth_tile(9, 9))
        return buf.tobytes()

    monkeypatch.setattr(bm, "_http_get_retrying", fake_get)
    tiles = [(0, 0), (1, 0), (5, 5), (6, 5)]
    downloaded = bm.prefetch_tiles(FAKE_PROVIDER, 10, tiles, cache=cache)
    assert downloaded == 2  # только (5,5) и (6,5)
    assert len(asked) == 2
    assert cache.get(FAKE_PROVIDER, 10, 5, 5) is not None


def test_prefetch_failure_is_best_effort(tmp_path, monkeypatch):
    """Сбой тайла не поднимается наружу: строгую ошибку даст уже сшивка.

    Так поведение при недоступной сети остаётся прежним, а префетч влияет
    только на скорость.
    """
    import aero_geoloc.basemap as bm

    cache = TileCache(tmp_path)

    def always_fail(url, *, timeout, retries=2):
        raise OSError("сеть недоступна")

    monkeypatch.setattr(bm, "_http_get_retrying", always_fail)
    assert bm.prefetch_tiles(FAKE_PROVIDER, 10, [(1, 1), (2, 2)], cache=cache) == 0
    assert cache.get(FAKE_PROVIDER, 10, 1, 1) is None


def test_fetch_basemap_prefetches_window_tiles(tmp_path, monkeypatch):
    """Сшивка окна сначала тянет недостающие тайлы пачкой, а не по одному.

    Проверяем именно факт префетча: к моменту последовательной сборки мозаики
    все тайлы окна уже в кэше, поэтому одиночная загрузка не вызывается.
    """
    import cv2

    import aero_geoloc.basemap as bm

    cache = TileCache(tmp_path)
    prefetched: list[tuple] = []

    def fake_get(url, *, timeout, retries=2):
        ok, buf = cv2.imencode(".png", synth_tile(0, 0))
        return buf.tobytes()

    real_prefetch = bm.prefetch_tiles

    def spy_prefetch(provider, zoom, tiles, **kw):
        tiles = list(tiles)
        prefetched.append(tuple(tiles))
        return real_prefetch(provider, zoom, tiles, **kw)

    monkeypatch.setattr(bm, "_http_get_retrying", fake_get)
    monkeypatch.setattr(bm, "prefetch_tiles", spy_prefetch)
    bm.fetch_basemap(0.0, 0.0, 10, 400, 400, provider=FAKE_PROVIDER, cache=cache)

    assert prefetched, "префетч не вызывался"
    assert len(prefetched[0]) >= 4  # окно 400x400 накрывает несколько тайлов 256px


def test_fetch_basemap_without_cache_keeps_serial_path(tmp_path, monkeypatch):
    """Без кэша префетчу некуда складывать — работает прежний путь по тайлу."""
    import cv2

    import aero_geoloc.basemap as bm

    called: list[int] = []

    def fake_get(url, *, timeout, retries=2):
        ok, buf = cv2.imencode(".png", synth_tile(0, 0))
        return buf.tobytes()

    monkeypatch.setattr(bm, "_http_get_retrying", fake_get)
    monkeypatch.setattr(bm, "prefetch_tiles",
                        lambda *a, **k: called.append(1) or 0)
    bm.fetch_basemap(0.0, 0.0, 10, 300, 300, provider=FAKE_PROVIDER, cache=None)
    assert not called


def test_http_get_retries_transient_failure(monkeypatch):
    """Разовый сетевой сбой переживается ретраем, а не роняет сборку карты."""
    import aero_geoloc.basemap as bm

    attempts = []

    def flaky(url, *, timeout):
        attempts.append(url)
        if len(attempts) < 3:
            raise OSError("SSL handshake timed out")
        return b"ok"

    monkeypatch.setattr(bm, "_http_get", flaky)
    monkeypatch.setattr(bm.time, "sleep", lambda *_: None)
    assert bm._http_get_retrying("http://x", timeout=1.0, retries=2) == b"ok"
    assert len(attempts) == 3


# --- дымовой тест с реальной сетью (по умолчанию пропускается) ---------------


@pytest.mark.skipif(
    os.environ.get("AERO_GEOLOC_NETWORK_TESTS") != "1",
    reason="сетевой тест выключен; включить: AERO_GEOLOC_NETWORK_TESTS=1",
)
def test_fetch_real_esri_basemap_smoke(tmp_path):
    cache = TileCache(tmp_path)
    image, georef = fetch_basemap(37.6173, 55.7558, 17, 512, 512, cache=cache)
    assert image.shape == (512, 512, 3)
    assert image.std() > 5.0  # реальный снимок, а не однотонная заглушка
    # Повторный вызов обязан обслужиться из кэша без сети.
    image2, _ = fetch_basemap(37.6173, 55.7558, 17, 512, 512, cache=cache, allow_network=False)
    np.testing.assert_array_equal(image, image2)
