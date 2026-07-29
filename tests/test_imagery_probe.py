"""Проверка «есть ли на этом уровне съёмка».

Тест существует из-за конкретного случая: на ``DSC00045`` Esri отдал на z19 HTTP
200 и серую заглушку, пайплайн сопоставил кадр с чистым листом и выдал позу.
Формально всё работало — результат был бессмысленным.

Главное, что проверяется здесь, — **чем нельзя пользоваться**. Порог по
дисперсии в одиночку не разделяет заглушку и настоящую съёмку: измерено по кэшу
проекта, у заглушки std 5.4, а у настоящей съёмки на z18 минимум 3.0 и первый
процентиль 4.8 (вода, тёмный лес). Поэтому вывод делается по набору проб, и
однородная вода в центре района не должна давать ложного «съёмки нет».

Сети здесь нет: кэш готовится руками, ``allow_network=False``.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from aero_geoloc.basemap import (
    ESRI_WORLD_IMAGERY,
    MIN_TILE_STD,
    ImageryProbe,
    TileCache,
    deepest_imagery_zoom,
    probe_imagery,
)
from aero_geoloc.basemap import _probe_tiles
from aero_geoloc.geo import ground_mpp

LON, LAT = 52.7791, 56.7747          # та самая точка из DSC00045
RADIUS_M = 3000.0
PROVIDER = ESRI_WORLD_IMAGERY


def _encode(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return buf.tobytes()


def placeholder() -> np.ndarray:
    """Заглушка Esri: светло-серое поле со слабым диагональным вотермарком.

    Числа подобраны под измеренные: яркость ~205, std ~5 — то есть **внутри**
    хвоста настоящей съёмки. Именно поэтому её ловит байтовое равенство.
    """
    tile = np.full((256, 256, 3), 207, np.uint8)
    for i in range(0, 256, 32):
        cv2.putText(tile, "esri", (i - 8, i + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (196, 196, 196), 1, cv2.LINE_AA)
    return tile


def imagery(seed: int) -> np.ndarray:
    """Настоящая съёмка: богатая текстура, каждый тайл свой."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)


def dark_water(seed: int) -> np.ndarray:
    """Настоящая съёмка сплошной воды: структуры нет, но тайлы **разные**."""
    rng = np.random.default_rng(seed)
    return (np.full((256, 256, 3), 26, np.int16)
            + rng.integers(-2, 3, (256, 256, 3))).clip(0, 255).astype(np.uint8)


def probe_coords(zoom: int) -> list[tuple[int, int]]:
    """Куда лягут пробы. Кладём тайлы ровно туда — район целиком не разложить.

    Кольцо на радиусе 3 км при z19 уходит на 72 тайла от центра, то есть диск —
    это ~21 тысяча тайлов. Разнесённость проб проверяется отдельно, ниже.
    """
    return _probe_tiles(LON, LAT, zoom, RADIUS_M, PROVIDER)


def fill(cache: TileCache, zoom: int, make, *, only=None) -> None:
    """Разложить тайлы по координатам проб (или по подмножеству ``only``)."""
    for x, y in probe_coords(zoom):
        if only is not None and (x, y) not in only:
            continue
        cache.put(PROVIDER, zoom, x, y, _encode(make(x * 7919 + y)))


def probe(cache: TileCache, zoom: int) -> ImageryProbe:
    return probe_imagery(LON, LAT, zoom, radius_m=RADIUS_M, provider=PROVIDER,
                         cache=cache, allow_network=False)


@pytest.fixture
def cache(tmp_path) -> TileCache:
    return TileCache(tmp_path)


# --- две подписи пустоты -----------------------------------------------------

def test_identical_tiles_are_a_placeholder(cache):
    """Байтовое равенство — главный признак: настоящие тайлы так не совпадают."""
    fill(cache, 19, lambda _: placeholder())
    p = probe(cache, 19)
    assert p.identical and not p.has_imagery
    assert "заглушка" in p.describe()


def test_placeholder_is_not_separable_by_variance_alone(cache):
    """Обоснование конструкции: std заглушки лежит в хвосте настоящей съёмки.

    Если этот assert когда-нибудь упадёт, значит заглушка стала контрастной и
    порог по дисперсии её ловит сам — но полагаться на это по-прежнему нельзя.
    """
    gray = cv2.cvtColor(placeholder(), cv2.COLOR_BGR2GRAY)
    assert gray.std() < MIN_TILE_STD
    water = cv2.cvtColor(dark_water(1), cv2.COLOR_BGR2GRAY)
    assert water.std() < MIN_TILE_STD          # настоящая съёмка ниже того же порога


def test_structureless_but_different_tiles_are_also_empty(cache):
    """Чёрное «нет данных» байтам не идентично, но сопоставлять с ним нечего."""
    fill(cache, 19, dark_water)
    p = probe(cache, 19)
    assert not p.identical and p.structured == 0 and not p.has_imagery
    assert "структуры" in p.describe()


# --- и чего детектор делать НЕ должен ----------------------------------------

def test_water_in_the_centre_does_not_condemn_the_region(cache):
    """Кольцо проб ради этого и заведено: район целиком водой не бывает."""
    centre = {probe_coords(19)[0]}
    fill(cache, 19, imagery)                       # весь район — съёмка
    fill(cache, 19, dark_water, only=centre)       # а в центре озеро
    p = probe(cache, 19)
    assert p.has_imagery and p.structured >= 1


def test_probes_are_spread_over_the_region(cache):
    """Пробы обязаны разойтись по району: девять соседних тайлов ничего не значат.

    Кольцо должно стоять примерно на радиусе района, иначе весь смысл теряется —
    вода в центре условно «накроет» и его.
    """
    coords = probe_coords(19)
    assert len(coords) == 9
    cx, cy = coords[0]
    tile_m = ground_mpp(LAT, 19) * PROVIDER.tile_size
    reach = max(max(abs(x - cx), abs(y - cy)) for x, y in coords) * tile_m
    assert reach == pytest.approx(RADIUS_M, rel=0.05)


def test_real_imagery_passes(cache):
    fill(cache, 19, imagery)
    p = probe(cache, 19)
    assert p.has_imagery and p.tiles == 9 and not p.identical


# --- спуск по уровням --------------------------------------------------------

def test_descends_to_the_level_that_has_imagery(cache):
    """Ровно случай DSC00045: z19 — заглушка, z18 — снимок."""
    fill(cache, 19, lambda _: placeholder())
    fill(cache, 18, imagery)
    zoom, probes = deepest_imagery_zoom(LON, LAT, radius_m=RADIUS_M, max_zoom=19,
                                        min_zoom=16, provider=PROVIDER, cache=cache,
                                        allow_network=False)
    assert zoom == 18
    assert [p.zoom for p in probes] == [19, 18]
    assert not probes[0].has_imagery and probes[-1].has_imagery


def test_no_imagery_anywhere_is_reported_as_such(cache):
    """Отказать надо ДО сборки карты, а не после матчинга по пустоте."""
    for z in (19, 18, 17, 16):
        fill(cache, z, lambda _: placeholder())
    zoom, probes = deepest_imagery_zoom(LON, LAT, radius_m=RADIUS_M, max_zoom=19,
                                        min_zoom=16, provider=PROVIDER, cache=cache,
                                        allow_network=False)
    assert zoom is None and len(probes) == 4


def test_deepest_level_wins_when_it_has_imagery(cache):
    fill(cache, 19, imagery)
    fill(cache, 18, imagery)
    zoom, probes = deepest_imagery_zoom(LON, LAT, radius_m=RADIUS_M, max_zoom=19,
                                        min_zoom=16, provider=PROVIDER, cache=cache,
                                        allow_network=False)
    assert zoom == 19 and len(probes) == 1      # ниже спускаться незачем


# --- проверка не подменяет собой диагностику сети ----------------------------

def test_unprobeable_level_does_not_block_the_run(cache):
    """Пустой кэш без сети — это не «съёмки нет», а «проверить нечем».

    Пусть о недоступности скажет обычный путь загрузки: у него сообщение точнее,
    чем «в этом районе нет съёмки».
    """
    p = probe(cache, 19)
    assert p.tiles == 0 and p.has_imagery


def test_probe_rejects_zoom_outside_the_provider(cache):
    with pytest.raises(ValueError, match="вне диапазона"):
        probe(cache, PROVIDER.max_zoom + 1)
