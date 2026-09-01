"""Компактное хранение OrthoLoC: точность упаковки против её же эталонов.

Проверяется не «как получилось», а объявленные в `scripts/ortholoc_store.py`
границы: шаг хранения GT даёт ошибку не больше половины шага, оси сетки DOP
восстанавливаются точно, сэмпл после записи и чтения отдаёт те же величины,
что исходный, а профиль `eval` не трогает пиксели вовсе — на них снимаются
числа бенчмарка.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ortholoc_store as store  # noqa: E402


def make_sample(h=24, w=32, hd=64, wd=64, sx=0.2, tilt=0.0):
    """Синтетический сэмпл OrthoLoC с аналитически известной геометрией."""
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    # мировые XY кадра: линейная развёртка плюс небольшой наклон плоскости
    pm = np.empty((h, w, 3), np.float32)
    pm[..., 0] = (xx - w / 2) * sx * 1.5 + tilt * yy
    pm[..., 1] = (yy - h / 2) * sx * 1.5
    pm[..., 2] = 3.0 + 0.01 * xx
    pm[0, 0, :] = np.nan                       # дыра в разметке

    dsm = np.empty((hd, wd, 3), np.float32)
    dsm[..., 0] = ((np.arange(wd) - (wd - 1) / 2) * sx)[None, :]
    dsm[..., 1] = ((np.arange(hd) - (hd - 1) / 2) * -sx)[:, None]
    dsm[..., 2] = 5.0 + 0.02 * np.mgrid[0:hd, 0:wd][1]

    return {
        "sample_id": np.asarray("T00_R0000"),
        "image_query": rng.integers(0, 255, (h, w, 3), dtype=np.uint8),
        "image_dop": rng.integers(0, 255, (hd, wd, 3), dtype=np.uint8),
        "point_map": pm,
        "dsm": dsm,
        "scale": np.asarray([sx, -sx], dtype=np.float16),
        "extrinsics": np.eye(3, 4, dtype=np.float32),
        "intrinsics": np.eye(3, dtype=np.float32),
        "vertices": np.stack([np.zeros(9), np.zeros(9), np.arange(9.0)], 1).astype(np.float32),
        "faces": np.zeros((4, 3), np.int32),
    }


class _Dict(dict):
    """Словарь с интерфейсом ``np.load`` — исходный сэмпл в памяти."""

    @property
    def files(self):
        return list(self.keys())


def test_gt_packing_stays_within_half_step():
    """Ошибка упаковки GT не больше половины объявленного шага."""
    rng = np.random.default_rng(3)
    gt_x = rng.uniform(-500, 1500, (40, 60))
    gt_y = rng.uniform(-500, 1500, (40, 60))
    gt_x[0, 0] = np.nan
    gt_y[0, 0] = np.nan

    packed, mask = store.pack_gt(gt_x, gt_y)
    back_x, back_y = store.unpack_gt(packed, mask)

    ok = np.isfinite(gt_x)
    assert not mask[0, 0]
    assert np.isnan(back_x[0, 0]) and np.isnan(back_y[0, 0])
    err = np.hypot(back_x[ok] - gt_x[ok], back_y[ok] - gt_y[ok])
    # ошибка по каждой оси ≤ шаг/2, значит по модулю ≤ шаг/√2
    assert err.max() <= store.GT_STEP / np.sqrt(2) + 1e-9


def test_gt_range_covers_dop_with_margin():
    """Смещение и шаг выбраны так, что координаты DOP влезают с запасом."""
    lo = -store.GT_BIAS
    hi = 65535 * store.GT_STEP - store.GT_BIAS
    assert lo <= -1024 and hi >= 3000        # сторона DOP 1024 px плюс вылет кадра


def test_pointmap_roundtrip_is_analytic():
    """Мировые XY ↔ пиксели DOP — взаимно обратные преобразования."""
    pm = np.zeros((5, 7, 3), np.float32)
    pm[..., 0] = np.linspace(-10, 10, 7)[None, :]
    pm[..., 1] = np.linspace(-4, 4, 5)[:, None]
    scale = (0.25, -0.25)

    gt_x, gt_y = store.gt_from_pointmap(pm, scale, 64, 64)
    back = store.pointmap_from_gt(gt_x, gt_y, pm[..., 2], scale, 64, 64)

    assert np.allclose(back[..., :2], pm[..., :2], atol=1e-6)


@pytest.mark.parametrize("profile", ["train", "eval"])
def test_written_sample_reads_back_within_declared_accuracy(tmp_path, profile):
    """Запись и чтение сохраняют геометрию в пределах объявленных границ."""
    src = _Dict(make_sample())
    path = tmp_path / "T00_R0000.npz"
    store.write(path, src, profile=profile)
    step = store.PROFILES[profile]["gt_step"]

    with store.open_sample(path) as slim:
        assert isinstance(slim, store.SlimSample)
        pm = slim["point_map"]
        dsm = slim["dsm"]

        ok = np.isfinite(src["point_map"][..., 0])
        # шаг GT в пикселях переводится в метры множителем |scale|
        tol_m = step / 2 * abs(float(src["scale"][0])) + 1e-6
        assert np.abs(pm[..., 0][ok] - src["point_map"][..., 0][ok]).max() <= tol_m
        assert np.abs(pm[..., 1][ok] - src["point_map"][..., 1][ok]).max() <= tol_m
        assert not np.isfinite(pm[0, 0, 0])          # дыра осталась дырой

        # оси сетки DOP восстанавливаются точно, высоты — в пределах float16
        assert np.array_equal(dsm[..., 0], src["dsm"][..., 0])
        assert np.array_equal(dsm[..., 1], src["dsm"][..., 1])
        assert np.abs(dsm[..., 2] - src["dsm"][..., 2]).max() <= 0.01

        assert slim.median_vertex_z == pytest.approx(4.0)
        assert slim["image_query"].shape == src["image_query"].shape


def test_rotated_dsm_grid_is_refused(tmp_path):
    """Повёрнутая сетка DOP не укладывается в два вектора — это отказ, а не тихая порча."""
    src = _Dict(make_sample())
    dsm = src["dsm"].copy()
    dsm[..., 0] += 0.5 * np.mgrid[0:dsm.shape[0], 0:dsm.shape[1]][0]   # X поехал по строкам
    src["dsm"] = dsm

    with pytest.raises(ValueError, match="не осевая"):
        store.build(src)


def test_grid_axes_tolerate_holes_but_not_orphan_heights():
    """Дыры в XY допустимы, пока в них нет высоты: иначе точка ожила бы из ничего."""
    base = make_sample()["dsm"].copy()
    holed = base.copy()
    holed[900 % base.shape[0]:, :, :] = np.nan          # нижние строки вне съёмки
    assert store.grid_axes(holed) is not None

    orphan = base.copy()
    orphan[10:, :, :2] = np.nan                          # XY нет, а высота осталась
    assert store.grid_axes(orphan) is None


def test_eval_profile_keeps_pixels_exact(tmp_path):
    """Профиль eval кодирует изображения без потерь: пиксели не меняются.

    Это не косметика: изображение — вход матчера, и потери в нём сдвигают
    измеряемую величину. Замерено парным сравнением на 40 сэмплах: JPEG q95
    меняет inl1 ванильной RoMa на 0.0103 по медиане, повторный прогон на тех же
    данных — на 0.0057, профиль eval — на 0.0040.
    """
    src = _Dict(make_sample())
    lossless, lossy = tmp_path / "e.npz", tmp_path / "t.npz"
    store.write(lossless, src, profile="eval")
    store.write(lossy, src, profile="train")

    with store.open_sample(lossless) as a, store.open_sample(lossy) as b:
        assert a.profile["codec"] == "webp"
        for key in ("image_query", "image_dop"):
            assert np.array_equal(a[key], src[key]), f"{key} изменился в профиле eval"
        # у профиля train потери допустимы; проверяем лишь, что это те же кадры
        assert b["image_query"].shape == src["image_query"].shape


def test_delta_encoding_survives_holes_and_falls_back(tmp_path):
    """Дельта-кодирование восстанавливает поле точно и честно откатывается."""
    rng = np.random.default_rng(11)
    values = np.cumsum(rng.integers(0, 40, (12, 20)), axis=1).astype(np.uint16)
    mask = np.ones(values.shape, bool)
    mask[3, 5:9] = False                       # дыра внутри строки
    mask[7] = False                            # строка без единого валидного

    enc = store.encode_plane(values, mask, "f")
    assert str(enc["f_mode"]) == "delta"
    back = store.decode_plane(enc, "f")
    assert np.array_equal(back[mask], values[mask])

    # поле со скачком больше int16 хранится напрямую, а не портится молча
    jumpy = np.zeros((4, 6), np.uint16)
    jumpy[:, 3:] = 65535
    enc2 = store.encode_plane(jumpy, np.ones(jumpy.shape, bool), "f")
    assert str(enc2["f_mode"]) == "raw"
    assert np.array_equal(store.decode_plane(enc2, "f"), jumpy)


def test_both_formats_share_the_reading_interface(tmp_path):
    """Исходный и компактный сэмплы читаются одним и тем же кодом."""
    src = _Dict(make_sample())
    raw_path, slim_path = tmp_path / "raw.npz", tmp_path / "slim.npz"
    np.savez_compressed(raw_path, **src)
    store.write(slim_path, src)

    assert store.is_slim(slim_path) and not store.is_slim(raw_path)
    with store.open_sample(raw_path) as raw, store.open_sample(slim_path) as slim:
        for key in ("image_query", "image_dop", "point_map", "dsm", "scale"):
            assert raw[key].shape == slim[key].shape
        rx, ry = raw["gt"]
        sx_, sy_ = slim["gt"]
        ok = np.isfinite(rx)
        half = store.PROFILES["train"]["gt_step"] / 2 + 1e-9
        assert np.abs(rx[ok] - sx_[ok]).max() <= half
        assert np.abs(ry[ok] - sy_[ok]).max() <= half
        assert raw.median_vertex_z == pytest.approx(slim.median_vertex_z)
