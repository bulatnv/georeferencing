"""Гейт привязки (§3 задания): годится ли ортофотоплан для пар с подложкой.

Печатает паспорт растра, берёт несколько кропов в разных частях рабочей
зоны, замеряет сдвиг «орто ↔ подложка» фазовой корреляцией и собирает
панель-шахматку для просмотра глазами. Вердикт даёт человек; скрипт даёт
числа и картинку.

    python open_orto/scripts/gate_check.py --raster open_orto/data/<файл>.tif \\
        --out open_orto/work/gate
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from rasterio.enums import Resampling

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rasters import BasemapSource, Grid, OrthoSource, gradient_map, phase_shift  # noqa: E402

CROP_GSD = 0.15      # м/пиксель для замера: детальнее подложки Esri z19 не берём
CROP_PX = 1024       # ≈154 м земли


def overview_mask(ortho: OrthoSource, width: int = 1200):
    """Обзорная маска валидных данных. По §2.5 задания она завышает покрытие —
    служит только для выбора кандидатов, каждый из которых потом проверяется
    нативным чтением."""
    ds = ortho.ds
    h = max(1, int(width * ds.height / ds.width))
    arr = ds.read(out_shape=(ds.count, h, width), resampling=Resampling.average)
    rgb = np.transpose(arr[:3], (1, 2, 0))
    from geom import valid_mask
    return valid_mask(rgb), rgb


def pick_anchors(mask: np.ndarray, ortho: OrthoSource, n: int, *, erode_px: int = 12):
    """Кандидаты-якоря по обзорной маске: эрозия + равномерная сетка по валидным."""
    m = cv2.erode(mask.astype(np.uint8), np.ones((erode_px, erode_px), np.uint8))
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return []
    b = ortho.bounds
    sx = (b.right - b.left) / mask.shape[1]
    sy = (b.top - b.bottom) / mask.shape[0]
    idx = np.linspace(0, len(xs) - 1, n * 4).astype(int)
    pts = []
    for i in idx:
        gx = b.left + (xs[i] + 0.5) * sx
        gy = b.top - (ys[i] + 0.5) * sy
        if all(math.hypot(gx - p[0], gy - p[1]) > 400 for p in pts):
            pts.append((gx, gy))
        if len(pts) >= n:
            break
    return pts


def measure(ortho: OrthoSource, base: BasemapSource, gx: float, gy: float):
    """Замер в одной точке: (панель, запись с числами) либо (None, запись)."""
    grid = Grid(x=gx, y=gy, size_px=CROP_PX, gsd=CROP_GSD)
    a_rgb, a_val = ortho.read_grid(grid)
    rec = {"x": round(gx, 1), "y": round(gy, 1), "valid_a": round(float(a_val.mean()), 3)}
    if rec["valid_a"] < 0.9:
        rec["status"] = "мало данных ортоплана"
        return None, rec
    b_rgb, b_val, info = base.read_grid(grid)
    rec.update(zoom=info["zoom"], mpp=round(info["mpp"], 3),
               valid_b=round(float(b_val.mean()), 3))
    ga = gradient_map(a_rgb, a_val)
    gb = gradient_map(b_rgb, b_val)
    dx, dy, peak = phase_shift(ga, gb)
    rec.update(dx_px=round(dx, 2), dy_px=round(dy, 2),
               shift_m=round(math.hypot(dx, dy) * CROP_GSD, 2), peak=round(peak, 4))

    # контроль остатка: перечитать подложку со сдвигом и замерить снова
    shift_m = (dx * CROP_GSD, -dy * CROP_GSD)   # +y пикселей = −Y метров
    b2_rgb, b2_val, _ = base.read_grid(grid, zoom=info["zoom"], shift_m=shift_m)
    dx2, dy2, peak2 = phase_shift(ga, gradient_map(b2_rgb, b2_val))
    rec.update(resid_m=round(math.hypot(dx2, dy2) * CROP_GSD, 2), peak2=round(peak2, 4))
    rec["status"] = "ok"

    tile = 96
    yy, xx = np.mgrid[0:CROP_PX, 0:CROP_PX]
    checker = (((yy // tile) + (xx // tile)) % 2).astype(bool)
    mix_raw = a_rgb.copy()
    mix_raw[checker] = b_rgb[checker]
    mix_fix = a_rgb.copy()
    mix_fix[checker] = b2_rgb[checker]
    panel = np.hstack([a_rgb, b_rgb, mix_raw, mix_fix])
    return panel, rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raster", required=True)
    ap.add_argument("--points", type=int, default=4)
    ap.add_argument("--out", default="open_orto/work/gate")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ortho = OrthoSource(args.raster)
    lon, lat = ortho.centre_lonlat()
    b = ortho.bounds
    print(f"растр: {Path(args.raster).name}")
    print(f"  CRS {ortho.ds.crs} | {ortho.ds.width}×{ortho.ds.height} | "
          f"res {ortho.res_x:.4f} м | центр lat={lat:.5f} lon={lon:.5f}")
    print(f"  охват {(b.right-b.left):.0f}×{(b.top-b.bottom):.0f} м")

    mask, _ = overview_mask(ortho)
    print(f"  обзорная доля валидных: {mask.mean():.3f} (завышена, §2.5)")
    anchors = pick_anchors(mask, ortho, args.points)
    print(f"  якорей для замера: {len(anchors)}")

    base = BasemapSource(ortho)
    panels, recs = [], []
    for gx, gy in anchors:
        panel, rec = measure(ortho, base, gx, gy)
        recs.append(rec)
        print("  ", rec)
        if panel is not None:
            panels.append(panel)
    if panels:
        img = np.vstack(panels)
        k = 2000 / img.shape[1]
        img = cv2.resize(img, (2000, int(img.shape[0] * k)))
        dst = out / f"gate_{Path(args.raster).stem}.jpg"
        cv2.imwrite(str(dst), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"панель (орто | подложка | шахматка сырая | шахматка после сдвига): {dst}")
    ok = [r for r in recs if r.get("status") == "ok"]
    if ok:
        sh = np.array([r["shift_m"] for r in ok])
        rs = np.array([r["resid_m"] for r in ok])
        print(f"ИТОГ: замеров {len(ok)}/{len(recs)} | сдвиг медиана {np.median(sh):.2f} м "
              f"(min {sh.min():.2f}, max {sh.max():.2f}) | остаток медиана {np.median(rs):.2f} м")
    ortho.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
