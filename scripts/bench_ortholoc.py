"""Бенчмарк матчеров на OrthoLoC: соответствия против плотного GT.

Каждый npz датасета самодостаточен: кадр UAV, ортофото (DOP) и попиксельная
мировая координата кадра (``point_map``). GT-отображение «пиксель кадра →
пиксель DOP» выводится аналитически (DOP ортографичен), поэтому матчер
меряется прямо по своим соответствиям — EPE против GT и доли инлайеров при
пиксельных порогах. RANSAC-подобия здесь нет сознательно: сцены наклонные
(медиана ~20°) и с рельефом, 4-DoF модель нашего тракта к ним не применима,
а вопрос бенчмарка — «насколько верны сами соответствия».

Пороги ядер — боевые калибровки треков A/F/F2, фиксируются в CSV:
roma/minima_roma — дефолты; romav2 — 0.05/0.05; loftr — 0.2/0.2 @640;
minima_loftr — 0.05/0.05 @640.

    python scripts/bench_ortholoc.py --splits test_inPlace,test_outPlace \\
        --matchers loftr,minima_loftr,roma,minima_roma,romav2 \\
        --out eval_out/ortholoc_bench.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ortholoc_store  # noqa: E402
from aero_geoloc.matcher import create_matcher  # noqa: E402

#: Боевые калибровки ядер (треки A/F/F2). Ключи create_matcher.
CONFIGS = {
    "loftr":        dict(coarse_thr=0.2, min_conf=0.2, max_side=640),
    "minima_loftr": dict(coarse_thr=0.05, min_conf=0.05, max_side=640),
    "roma":         dict(),
    "minima_roma":  dict(),
    "romav2":       dict(),
}

FIELDS = [
    "sample", "split", "scene", "variant", "matcher", "config",
    "tilt_deg", "height_m", "gsd",
    "n_pairs", "n_gt_valid", "frac_gt_invalid",
    "epe_med_px", "epe_p90_px",
    "inl1_frac", "inl3_frac", "inl5_frac", "n_inl5",
    "epe_med_m", "sec",
]


def load_sample(path: Path):
    # оба формата датасета читаются одним интерфейсом: исходный сэмпл и
    # компактный (scripts/ortholoc_store.py) отдают одни и те же величины
    d = ortholoc_store.open_sample(path)
    q = cv2.cvtColor(d["image_query"], cv2.COLOR_RGB2GRAY)
    dop = cv2.cvtColor(d["image_dop"], cv2.COLOR_RGB2GRAY)
    pm = d["point_map"].astype(np.float64)
    sx, sy = np.asarray(d["scale"], dtype=np.float64)
    ext = d["extrinsics"]
    hd, wd = dop.shape[:2]
    # GT-карта: пиксель кадра -> пиксель DOP
    gt_x = pm[..., 0] / sx + (wd - 1) / 2.0
    gt_y = pm[..., 1] / sy + (hd - 1) / 2.0
    tilt = float(np.degrees(np.arccos(np.clip(-ext[2, 2], -1, 1))))
    R, t = ext[:, :3], ext[:, 3]
    # высота съёмки — над медианной вершиной меша сцены; в компактном формате
    # меш не хранится, но его медиана Z сохранена отдельным числом
    height = float((-R.T @ t)[2] - d.median_vertex_z)
    return q, dop, gt_x, gt_y, tilt, height, float(abs(sx))


def evaluate(corr, gt_x, gt_y, gsd):
    """EPE предсказанных пар против GT; пары вне меша/кадра — отдельным счётом."""
    n = len(corr)
    if n == 0:
        return dict(n_pairs=0, n_gt_valid=0, frac_gt_invalid="", epe_med_px="",
                    epe_p90_px="", inl1_frac="", inl3_frac="", inl5_frac="",
                    n_inl5=0, epe_med_m="")
    h, w = gt_x.shape
    xi = np.clip(np.round(corr.pts_q[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(corr.pts_q[:, 1]).astype(int), 0, h - 1)
    gx, gy = gt_x[yi, xi], gt_y[yi, xi]
    valid = np.isfinite(gx) & np.isfinite(gy)
    if not valid.any():
        return dict(n_pairs=n, n_gt_valid=0, frac_gt_invalid=1.0, epe_med_px="",
                    epe_p90_px="", inl1_frac="", inl3_frac="", inl5_frac="",
                    n_inl5=0, epe_med_m="")
    epe = np.hypot(corr.pts_r[valid, 0] - gx[valid], corr.pts_r[valid, 1] - gy[valid])
    return dict(
        n_pairs=n, n_gt_valid=int(valid.sum()),
        frac_gt_invalid=round(float(1 - valid.mean()), 4),
        epe_med_px=round(float(np.median(epe)), 2),
        epe_p90_px=round(float(np.percentile(epe, 90)), 2),
        inl1_frac=round(float((epe < 1).mean()), 4),
        inl3_frac=round(float((epe < 3).mean()), 4),
        inl5_frac=round(float((epe < 5).mean()), 4),
        n_inl5=int((epe < 5).sum()),
        epe_med_m=round(float(np.median(epe)) * gsd, 2),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default="data/OrthoLoC")
    parser.add_argument("--splits", default="test_inPlace,test_outPlace")
    parser.add_argument("--matchers", default="loftr,minima_loftr,roma,minima_roma,romav2")
    parser.add_argument("--limit", type=int, default=0, help="первых N сэмплов на сплит (0 = все)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    matchers = [m.strip() for m in args.matchers.split(",") if m.strip()]
    files = []
    for s in splits:
        got = sorted((root / s).glob("*.npz"))
        if args.limit:
            got = got[: args.limit]
        files += [(s, f) for f in got]
    print(f"сэмплов: {len(files)}, ядра: {matchers}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = out.exists() and out.stat().st_size > 0
    done = set()
    if wrote_header:                      # докачивающий режим: продолжить CSV
        with out.open(encoding="utf-8") as fh:
            done = {(r["sample"], r["matcher"]) for r in csv.DictReader(fh)}
        print(f"в {out} уже {len(done)} строк — продолжаю", flush=True)

    with out.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if not wrote_header:
            writer.writeheader()
        for name in matchers:
            cfg = CONFIGS[name]
            matcher = create_matcher(name, **cfg)
            cfg_str = ";".join(f"{k}={v}" for k, v in cfg.items()) or "default"
            t_arm = time.perf_counter()
            n_done = 0
            for split, f in files:
                if (f.stem, name) in done:
                    continue
                try:
                    q, dop, gt_x, gt_y, tilt, height, gsd = load_sample(f)
                except Exception as exc:  # noqa: BLE001 - битый файл не валит прогон
                    print("SKIP", f.name, exc, flush=True)
                    continue
                started = time.perf_counter()
                corr = matcher.match(q, dop)
                sec = time.perf_counter() - started
                row = dict(
                    sample=f.stem, split=split, scene=f.stem.split("_")[0],
                    variant="xDOP" if "_xDOP" in f.stem else "R",
                    matcher=name, config=cfg_str, tilt_deg=round(tilt, 1),
                    height_m=round(height, 0) if np.isfinite(height) else "",
                    gsd=round(gsd, 3), sec=round(sec, 2),
                    **evaluate(corr, gt_x, gt_y, gsd),
                )
                writer.writerow(row)
                n_done += 1
                if n_done % 100 == 0:
                    fh.flush()
                    print(f"[{name}] {n_done} сэмплов, {time.perf_counter()-t_arm:.0f} с",
                          flush=True)
            fh.flush()
            del matcher
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            print(f"[{name}] готово: {n_done} строк за {time.perf_counter()-t_arm:.0f} с",
                  flush=True)
    print("готово:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
