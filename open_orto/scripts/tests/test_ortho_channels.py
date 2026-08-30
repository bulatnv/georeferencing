"""V-тест: источник отдаёт три канала независимо от того, сколько их в файле.

В наборе встречаются одноканальные (панхром) и четырёхканальные (RGB плюс
альфа или ближний ИК) ортопланы. Всё выше `OrthoSource` считает картинку
трёхканальной — в частности, маска валидности берёт размах по оси каналов, и
на одноканальном растре она обваливалась с несовпадением форм.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from rasters import Grid, OrthoSource  # noqa: E402


def _write(path: Path, bands: int, size: int = 256) -> Path:
    rng = np.random.default_rng(3)
    data = rng.integers(30, 220, size=(bands, size, size), dtype=np.uint8)
    with rasterio.open(path, "w", driver="GTiff", width=size, height=size,
                       count=bands, dtype="uint8", crs="EPSG:32635",
                       transform=from_origin(500000.0, 4600000.0, 0.5, 0.5)) as ds:
        ds.write(data)
    return path


@pytest.mark.parametrize("bands", [1, 3, 4])
def test_read_grid_always_returns_three_channels(tmp_path, bands):
    src = OrthoSource(_write(tmp_path / f"r{bands}.tif", bands))
    try:
        # сетка внутри растра: 256 px по 0.5 м = 128 м стороны
        grid = Grid(x=500032.0, y=4599968.0, size_px=64, gsd=0.5)
        rgb, valid = src.read_grid(grid)
        assert rgb.shape == (64, 64, 3), f"{bands}-канальный дал {rgb.shape}"
        assert valid.shape == (64, 64)
        assert valid.dtype == bool
    finally:
        src.close()


def test_single_band_is_replicated_not_padded(tmp_path):
    """Панхром разворачивается в три одинаковых канала, а не дополняется нулями.

    Нули в двух каналах сделали бы кадр синим и сломали бы и цветовую
    аугментацию, и любую оценку по яркости.
    """
    src = OrthoSource(_write(tmp_path / "pan.tif", 1))
    try:
        rgb, _ = src.read_grid(Grid(x=500032.0, y=4599968.0, size_px=64, gsd=0.5))
        assert np.array_equal(rgb[..., 0], rgb[..., 1])
        assert np.array_equal(rgb[..., 1], rgb[..., 2])
        assert rgb[..., 0].std() > 5, "картинка не должна быть константной"
    finally:
        src.close()
