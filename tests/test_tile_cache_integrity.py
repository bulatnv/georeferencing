"""Кэш тайлов: пустышка не должна травить район, а битый тайл — хоронить сборку.

Тесты существуют из-за конкретного падения: сборка карты района под Крефельдом
умерла на ассерте OpenCV ``!buf.empty()`` в декодере тайла, потеряв двенадцать
минут работы. Воспроизвести не удалось — со второй попытки прогон прошёл, — но
разбор вскрыл четыре независимых дефекта, каждый из которых достаточен:

1. ``put`` писал через ``write_bytes``: файл сначала обрезается, и всякий, кто
   прочитает его в этот момент (параллельный поток загрузки, следующий прогон
   после Ctrl+C), получает пустоту;
2. ``get`` возвращал ``b""`` как валидные данные — одна пустышка травила район
   навсегда;
3. декодер отдавал наружу стек из ``imgcodecs`` вместо внятной причины;
4. один битый тайл убивал всю сборку без единой попытки перечитать.

Сети здесь нет. Там, где нужен «сервер», он подменяется.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np
import pytest

from aero_geoloc import basemap as bm
from aero_geoloc.basemap import ESRI_WORLD_IMAGERY, TileCache, load_tile

PROVIDER = ESRI_WORLD_IMAGERY
Z, X, Y = 18, 100, 200


def tile_bytes(seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    ok, buf = cv2.imencode(".jpg", rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


@pytest.fixture
def cache(tmp_path) -> TileCache:
    return TileCache(tmp_path)


# --- пустышка не выдаёт себя за данные ---------------------------------------

def test_empty_file_reads_as_a_miss(cache):
    """Иначе она доходит до декодера и роняет сборку на каждом прогоне."""
    p = cache.path(PROVIDER, Z, X, Y)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    assert cache.get(PROVIDER, Z, X, Y) is None


def test_empty_response_is_not_cached(cache):
    """Сервер, ответивший 200 с пустым телом, не должен оставлять запись."""
    cache.put(PROVIDER, Z, X, Y, b"")
    assert not cache.path(PROVIDER, Z, X, Y).exists()


def test_put_is_atomic_no_partial_file_survives(cache):
    """Запись идёт через временный файл: обрезанного .jpg в кэше не остаётся."""
    data = tile_bytes()
    cache.put(PROVIDER, Z, X, Y, data)
    assert cache.get(PROVIDER, Z, X, Y) == data
    leftovers = list(cache.path(PROVIDER, Z, X, Y).parent.glob("*.part"))
    assert leftovers == []


def test_ready_tile_is_never_overwritten(cache):
    """``(z, x, y)`` — неизменяемое содержимое, перезаписывать его незачем.

    Это не микрооптимизация, а способ убрать целый класс гонок: на Windows
    подмена файла, который кто-то читает, отказывает обеим сторонам (измерено:
    228 из 300 ``os.replace`` и 115 чтений с ``PermissionError``). Не трогая
    готовый файл, мы этого просто не допускаем.
    """
    first = tile_bytes(1)
    cache.put(PROVIDER, Z, X, Y, first)
    cache.put(PROVIDER, Z, X, Y, tile_bytes(2))
    assert cache.get(PROVIDER, Z, X, Y) == first


def test_empty_cached_file_is_replaced_not_kept(cache):
    """Исключение из правила: пустышку из отравленного кэша чинить обязаны."""
    p = cache.path(PROVIDER, Z, X, Y)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    good = tile_bytes(5)
    cache.put(PROVIDER, Z, X, Y, good)
    assert cache.get(PROVIDER, Z, X, Y) == good


def test_racing_writers_never_expose_a_partial_read(cache):
    """Реальный сценарий: два окна подложки делят тайл и качают его разом.

    Читатель обязан увидеть либо промах, либо целые данные — но никогда обрезок.
    Ни один поток при этом не имеет права умереть с исключением.
    """
    data = tile_bytes(7)
    seen: list[bytes | None] = []
    errors: list[BaseException] = []
    start = threading.Barrier(5)
    stop = threading.Event()

    def writer() -> None:
        try:
            start.wait()
            for _ in range(100):
                cache.put(PROVIDER, Z, X, Y, data)
        except BaseException as exc:      # noqa: BLE001 — падение потока и есть дефект
            errors.append(exc)

    def reader() -> None:
        try:
            start.wait()
            while not stop.is_set():
                seen.append(cache.get(PROVIDER, Z, X, Y))
        except BaseException as exc:      # noqa: BLE001
            errors.append(exc)

    writers = [threading.Thread(target=writer) for _ in range(4)]
    readers = [threading.Thread(target=reader)]
    for t in [*writers, *readers]:
        t.start()
    for t in writers:
        t.join()
    stop.set()
    for t in readers:
        t.join()

    assert not errors, f"поток упал: {errors[:1]}"
    assert seen, "читатель не успел ничего прочитать — тест бесполезен"
    assert all(s is None or s == data for s in seen)
    assert list(cache.path(PROVIDER, Z, X, Y).parent.glob("*.part")) == []


# --- внятная ошибка вместо ассерта OpenCV ------------------------------------

def test_empty_buffer_gives_a_readable_error():
    """OpenCV встречает пустой буфер ассертом `!buf.empty()`, а не возвратом None."""
    with pytest.raises(ValueError, match="пустой тайл"):
        bm._decode_tile(b"", PROVIDER)


# --- битый тайл не хоронит сборку --------------------------------------------

def test_corrupt_cached_tile_is_refetched(cache, monkeypatch):
    """Двенадцать минут сборки не должны отменяться одним испорченным файлом."""
    good = tile_bytes(3)
    p = cache.path(PROVIDER, Z, X, Y)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\xff\xd8not a jpeg at all")      # мусор, но не пустой файл

    calls: list[str] = []

    def fake_get(url, *, timeout, retries=2):
        calls.append(url)
        return good

    monkeypatch.setattr(bm, "_http_get_retrying", fake_get)
    tile = load_tile(PROVIDER, Z, X, Y, cache=cache, allow_network=True)
    assert tile.shape == (256, 256, 3)
    assert len(calls) == 1                            # перекачали ровно один раз
    assert cache.get(PROVIDER, Z, X, Y) == good       # и починили запись в кэше


def test_broken_tile_on_the_server_does_not_loop(cache, monkeypatch):
    """Повтор ровно один: сломанный тайл у провайдера не должен зациклить нас."""
    p = cache.path(PROVIDER, Z, X, Y)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"not a jpeg")

    calls: list[str] = []

    def fake_get(url, *, timeout, retries=2):
        calls.append(url)
        return b"also not a jpeg"

    monkeypatch.setattr(bm, "_http_get_retrying", fake_get)
    with pytest.raises(ValueError):
        load_tile(PROVIDER, Z, X, Y, cache=cache, allow_network=True)
    assert len(calls) == 1


def test_offline_corrupt_tile_still_raises(cache):
    """Без сети чинить нечем — ошибка обязана дойти, а не притвориться успехом."""
    p = cache.path(PROVIDER, Z, X, Y)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"not a jpeg")
    with pytest.raises(ValueError):
        load_tile(PROVIDER, Z, X, Y, cache=cache, allow_network=False)
