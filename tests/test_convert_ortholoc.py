"""Аналитические эталоны конвертера OrthoLoC → канонический формат пары.

Геометрия проверяется против ручных построений, а не «как получилось»:
warp из синтетического point_map обязан вернуть тождество, проекция
синтетического пинхола — замкнуться в исходные пиксели, окно B — дать
предсказуемую ко-видимость.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from convert_ortholoc import (  # noqa: E402
    balance_per_scene,
    choose_window,
    jpeg_bytes,
    load_pair,
    median_gsd,
    project_to_camera,
    pseudo_season,
    vegetation_weight,
    warp_from_pointmap,
    warp_to_window,
    window_covis,
)


def synth_pointmap(h, w, gsd, wb, hb):
    """point_map кадра, снятого точно в надир по центру DOP-сетки:
    пиксель (i, j) кадра смотрит на пиксель (i, j) DOP."""
    jj, ii = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    x = (jj - (wb - 1) / 2.0) * gsd
    y = (ii - (hb - 1) / 2.0) * (-gsd)
    return np.stack([x, y, np.zeros_like(x)], axis=-1)


# --- warp_from_pointmap --------------------------------------------------------

def test_nadir_pointmap_gives_identity_warp():
    pm = synth_pointmap(64, 64, 0.2, 64, 64)
    gx, gy = warp_from_pointmap(pm, (0.2, -0.2), 64, 64)
    jj, ii = np.meshgrid(np.arange(64, dtype=float), np.arange(64, dtype=float))
    assert np.allclose(gx, jj, atol=1e-9)
    assert np.allclose(gy, ii, atol=1e-9)


def test_offcentre_pointmap_shifts_warp():
    """Сдвиг мировых координат на 10 GSD — сдвиг warp ровно на 10 пикселей."""
    pm = synth_pointmap(32, 32, 0.5, 128, 128)
    pm[..., 0] += 10 * 0.5
    gx, _ = warp_from_pointmap(pm, (0.5, -0.5), 128, 128)
    assert np.allclose(gx[0, 0], (0 - 63.5) + 63.5 + 10)


# --- окно B и ко-видимость -----------------------------------------------------

def test_window_covis_counts_inside_only():
    gx, gy = np.meshgrid(np.arange(10, dtype=float), np.arange(10, dtype=float))
    finite = np.isfinite(gx)
    assert window_covis(gx, gy, finite, (0, 0, 10, 10)) == 1.0
    assert window_covis(gx, gy, finite, (0, 0, 5, 10)) == pytest.approx(0.5)
    assert window_covis(gx, gy, finite, (20, 20, 5, 5)) == 0.0


def test_choose_window_is_deterministic_and_above_floor():
    gx, gy = np.meshgrid(np.arange(400, dtype=float) + 300,
                         np.arange(400, dtype=float) + 300)
    win1, c1 = choose_window(gx, gy, 1024, 1024, np.random.default_rng(7))
    win2, c2 = choose_window(gx, gy, 1024, 1024, np.random.default_rng(7))
    assert win1 == win2 and c1 == c2            # детерминизм от rng
    assert c1 >= 0.30                           # не ниже пола спеки


def test_choose_window_full_covis_when_footprint_small():
    """Отпечаток меньше любого окна — covis 1.0, боевая ситуация, не ошибка."""
    gx, gy = np.meshgrid(np.arange(100, dtype=float) + 460,
                         np.arange(100, dtype=float) + 460)
    _, c = choose_window(gx, gy, 1024, 1024, np.random.default_rng(1))
    assert c == 1.0


def test_warp_to_window_masks_outside_and_shifts():
    gx = np.array([[5.0, 100.0], [np.nan, 7.0]])
    gy = np.array([[5.0, 5.0], [1.0, 7.0]])
    warp, mask = warp_to_window(gx, gy, (4, 4, 10, 10))
    assert mask.tolist() == [[1, 0], [0, 1]]
    assert warp[0, 0].tolist() == [1.0, 1.0]     # 5-4
    assert np.isnan(warp[0, 1]).all()            # 100 вне окна
    assert warp.dtype == np.float16


# --- проекция в камеру ---------------------------------------------------------

def test_project_round_trip_synthetic_pinhole():
    """Точки, развёрнутые из пикселей через K и глубину, проецируются назад."""
    K = np.array([[700.0, 0, 320], [0, 700.0, 240], [0, 0, 1]])
    R = np.eye(3)
    t = np.array([0.0, 0.0, 0.0])
    ext = np.hstack([R, t[:, None]])
    px = np.array([10.0, 320.0, 600.0])
    py = np.array([15.0, 240.0, 400.0])
    z = np.array([50.0, 80.0, 120.0])
    pts = np.stack([(px - 320) / 700 * z, (py - 240) / 700 * z, z], axis=-1)
    u, v, zz = project_to_camera(pts, K, ext)
    assert np.allclose(u, px) and np.allclose(v, py) and np.allclose(zz, z)


def test_project_behind_camera_gets_negative_depth():
    K = np.eye(3)
    ext = np.hstack([np.eye(3), np.zeros((3, 1))])
    _, _, z = project_to_camera(np.array([[0.0, 0, -5.0]]), K, ext)
    assert z[0] < 0


# --- dop_scale: восстановление масштаба из dsm ---------------------------------

def test_dop_scale_falls_back_to_dsm_grid():
    """У сцены L06 нет члена scale — масштаб восстанавливается из шага
    мировых координат сетки dsm."""
    from convert_ortholoc import dop_scale
    hb = wb = 16
    jj, ii = np.meshgrid(np.arange(wb, dtype=float), np.arange(hb, dtype=float))
    dsm = np.stack([(jj - (wb - 1) / 2) * 0.25,
                    (ii - (hb - 1) / 2) * -0.25,
                    np.zeros_like(jj)], axis=-1)
    d = {"dsm": dsm}                      # dict без .files — ветка fallback
    s = dop_scale(d)
    assert s[0] == pytest.approx(0.25) and s[1] == pytest.approx(-0.25)


def test_dop_scale_prefers_explicit_member():
    from convert_ortholoc import dop_scale

    class Fake(dict):
        files = ("scale",)

    d = Fake(scale=np.array([0.3, -0.3]))
    assert dop_scale(d)[0] == pytest.approx(0.3)


# --- median_gsd ----------------------------------------------------------------

def test_median_gsd_of_uniform_grid():
    pm = synth_pointmap(32, 32, 0.25, 32, 32)
    assert median_gsd(pm) == pytest.approx(0.25, abs=1e-6)


# --- балансировка по сценам ----------------------------------------------------

def _paths(scene, variant, n):
    tag = "xDOP" if variant == "xDOP" else "R"
    return [Path(f"{scene}_{tag}{i:04d}.npz") for i in range(n)]


def test_balance_caps_big_scene_and_keeps_small():
    files = _paths("L01", "R", 300) + _paths("L02", "R", 40)
    got = balance_per_scene(files, cap=100, split="train")
    scenes = [f.stem.split("_")[0] for f in got]
    assert scenes.count("L01") == 100      # большая сцена обрезана до квоты
    assert scenes.count("L02") == 40       # маленькая взята целиком


def test_balance_is_stratified_by_variant_and_deterministic():
    files = _paths("L01", "R", 240) + _paths("L01", "xDOP", 60)  # 80% / 20%
    got1 = balance_per_scene(files, cap=100, split="train")
    got2 = balance_per_scene(files, cap=100, split="train")
    assert got1 == got2                    # детерминизм
    n_x = sum("_xDOP" in f.stem for f in got1)
    assert n_x == 20                       # доля xDOP сохранена (20 из 100)


# --- псевдосезоны ---------------------------------------------------------------

def _green_field():
    """Синтетика: левая половина — зелёная «растительность», правая — серая дорога."""
    img = np.zeros((32, 64, 3), np.uint8)
    img[:, :32] = (55, 140, 45)
    img[:, 32:] = (128, 128, 128)
    return img


def test_vegetation_weight_separates_green_from_gray():
    w = vegetation_weight(_green_field())
    # серым пикселям сигмоида даёт ~0.12 — лёгкое глобальное участие, это
    # осознанно (резкая маска дала бы матчеру читерский контур)
    assert w[16, 8] > 0.9 and w[16, 56] < 0.15


def test_autumn_shifts_green_hue_but_not_road():
    img = _green_field()
    out = pseudo_season(img, "autumn", np.random.default_rng(0))
    hsv_in = pytest.importorskip("cv2").cvtColor(img, 41)   # RGB2HSV
    import cv2 as _cv2
    hsv_out = _cv2.cvtColor(out, _cv2.COLOR_RGB2HSV)
    assert hsv_out[16, 8, 0] < hsv_in[16, 8, 0] - 15        # зелень ушла к оранжевому
    road_diff = np.abs(out[16, 56].astype(int) - img[16, 56].astype(int)).max()
    assert road_diff <= 12                                  # дорога почти не тронута


def test_winter_desaturates_and_brightens_globally():
    import cv2 as _cv2
    img = _green_field()
    out = pseudo_season(img, "winter", np.random.default_rng(0))
    hsv_in = _cv2.cvtColor(img, _cv2.COLOR_RGB2HSV).astype(int)
    hsv_out = _cv2.cvtColor(out, _cv2.COLOR_RGB2HSV).astype(int)
    assert hsv_out[..., 1].mean() < hsv_in[..., 1].mean() * 0.6   # насыщенность упала
    assert hsv_out[..., 2].mean() > hsv_in[..., 2].mean()          # кадр светлее


def test_pseudo_season_is_deterministic_and_shape_preserving():
    img = _green_field()
    o1 = pseudo_season(img, "spring", np.random.default_rng(5))
    o2 = pseudo_season(img, "spring", np.random.default_rng(5))
    assert np.array_equal(o1, o2)
    assert o1.shape == img.shape and o1.dtype == np.uint8


def test_unknown_season_raises():
    with pytest.raises(ValueError):
        pseudo_season(_green_field(), "summer", np.random.default_rng(0))


# --- запись/чтение пары --------------------------------------------------------

def test_pair_roundtrip(tmp_path):
    # гладкое изображение (градиент) — реалистичный случай для JPEG q95;
    # белый шум был бы худшим случаем и мерил бы не формат, а энтропию
    gy, gx = np.mgrid[0:32, 0:48]
    rgb = np.stack([gx * 5, gy * 7, (gx + gy) * 3], axis=-1).astype(np.uint8)
    warp = np.full((32, 48, 2), 3.5, dtype=np.float16)
    mask = np.ones((32, 48), dtype=np.uint8)
    meta = {"scene": "L01", "covis_frac": 0.5}
    np.savez_compressed(
        tmp_path / "p.npz",
        image_a_jpeg=jpeg_bytes(rgb), image_b_jpeg=jpeg_bytes(rgb),
        warp_ab=warp, mask_ab=mask,
        meta=np.str_(json.dumps(meta)), pinhole=np.str_("null"))
    got = load_pair(tmp_path / "p.npz")
    assert got["image_a"].shape == (32, 48, 3)
    assert got["meta"]["scene"] == "L01"
    assert got["pinhole"] is None
    assert got["warp_ab"].dtype == np.float32
    # JPEG q95 близок к исходнику
    assert float(np.abs(got["image_a"].astype(int) - rgb.astype(int)).mean()) < 12
