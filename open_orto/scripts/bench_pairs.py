"""Прогон матчеров по корпусу пар канонического формата.

Меряет соответствия против плотного GT (``warp_ab``): EPE и доли инлайеров.
В отличие от бенчмарка OrthoLoC, здесь у самой разметки есть измеренный шум:
на парах с подложкой остаток ≈ 3 px (привязка + параллакс зданий), на
контрольных ``same_source`` ≈ 0. Поэтому:

- сравнение ядер честно на **любых** парах (шум разметки общий для всех);
- абсолютные числа читать надо с оглядкой: порог 3 px на парах с подложкой
  соизмерим с шумом GT, поэтому в таблицу выведены и 5, и 10 px;
- контрольная ось ``same_source`` даёт **верхнюю границу** возможностей ядра
  при точной разметке.

    python open_orto/scripts/bench_pairs.py --dataset open_orto/dataset \\
        --matchers loftr,minima_loftr,roma,minima_roma,romav2 \\
        --out open_orto/work/bench_pairs.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[2]))

from aero_geoloc.matcher import create_matcher  # noqa: E402

#: Боевые калибровки ядер — те же, что в bench_ortholoc.py (треки A/F/F2).
CONFIGS = {
    "loftr":        dict(coarse_thr=0.2, min_conf=0.2, max_side=640),
    "minima_loftr": dict(coarse_thr=0.05, min_conf=0.05, max_side=640),
    "roma":         dict(),
    "minima_roma":  dict(),
    "romav2":       dict(),
}

FIELDS = [
    "pair", "scene", "pair_kind", "layout", "matcher", "config",
    "height_m", "tilt_deg", "yaw_deg", "delta_yaw_deg", "scale_ratio",
    "b_px", "covis_frac",
    "n_pairs", "n_gt_valid", "epe_med_px", "epe_p90_px",
    "inl1_frac", "inl3_frac", "inl5_frac", "inl10_frac", "n_inl5",
    "epe_med_m", "sec",
]


def load_pair(path: Path):
    d = np.load(path, allow_pickle=False)
    return {
        "image_a": cv2.imdecode(d["image_a_jpeg"], cv2.IMREAD_COLOR),
        "image_b": cv2.imdecode(d["image_b_jpeg"], cv2.IMREAD_COLOR),
        "warp": d["warp_ab"].astype(np.float32),
        "mask": d["mask_ab"].astype(bool),
        "meta": json.loads(str(d["meta"])),
    }


def evaluate(corr, warp, mask, gsd_b):
    """EPE предсказанных пар против GT; пары вне маски считаются отдельно."""
    n = len(corr)
    empty = dict(n_pairs=n, n_gt_valid=0, epe_med_px="", epe_p90_px="",
                 inl1_frac="", inl3_frac="", inl5_frac="", inl10_frac="",
                 n_inl5=0, epe_med_m="")
    if n == 0:
        return empty
    h, w = mask.shape
    xi = np.clip(np.round(corr.pts_q[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(corr.pts_q[:, 1]).astype(int), 0, h - 1)
    ok = mask[yi, xi]
    if not ok.any():
        return dict(empty, n_gt_valid=0)
    gt = warp[yi[ok], xi[ok]]
    good = np.isfinite(gt).all(axis=1)
    if not good.any():
        return dict(empty, n_gt_valid=0)
    epe = np.hypot(corr.pts_r[ok][good, 0] - gt[good, 0],
                   corr.pts_r[ok][good, 1] - gt[good, 1])
    return dict(
        n_pairs=n, n_gt_valid=int(good.sum()),
        epe_med_px=round(float(np.median(epe)), 2),
        epe_p90_px=round(float(np.percentile(epe, 90)), 2),
        inl1_frac=round(float((epe < 1).mean()), 4),
        inl3_frac=round(float((epe < 3).mean()), 4),
        inl5_frac=round(float((epe < 5).mean()), 4),
        inl10_frac=round(float((epe < 10).mean()), 4),
        n_inl5=int((epe < 5).sum()),
        epe_med_m=round(float(np.median(epe)) * gsd_b, 2),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="open_orto/dataset")
    ap.add_argument("--matchers", default="loftr,minima_loftr,roma,minima_roma,romav2")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="open_orto/work/bench_pairs.csv")
    args = ap.parse_args()
    from cpu_affinity import pin_to_performance
    pin_to_performance(verbose=False)

    files = sorted(Path(args.dataset).glob("*.npz"))
    if args.limit:
        files = files[: args.limit]
    matchers = [m.strip() for m in args.matchers.split(",") if m.strip()]
    print(f"пар: {len(files)}, ядра: {matchers}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists() and out.stat().st_size > 0:
        with out.open(encoding="utf-8") as fh:
            done = {(r["pair"], r["matcher"]) for r in csv.DictReader(fh)}
        print(f"уже посчитано строк: {len(done)}", flush=True)

    with out.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if not done:
            writer.writeheader()
        for name in matchers:
            cfg = CONFIGS[name]
            matcher = create_matcher(name, **cfg)
            cfg_str = ";".join(f"{k}={v}" for k, v in cfg.items()) or "default"
            t0, n_done = time.perf_counter(), 0
            for f in files:
                if (f.stem, name) in done:
                    continue
                pair = load_pair(f)
                m = pair["meta"]
                a = cv2.cvtColor(pair["image_a"], cv2.COLOR_BGR2GRAY)
                b = cv2.cvtColor(pair["image_b"], cv2.COLOR_BGR2GRAY)
                started = time.perf_counter()
                corr = matcher.match(a, b)
                sec = time.perf_counter() - started
                row = dict(
                    pair=f.stem, scene=m["scene"], pair_kind=m["pair_kind"],
                    layout=m["pair_layout"], matcher=name, config=cfg_str,
                    height_m=m["height_m"], tilt_deg=m["tilt_deg"],
                    yaw_deg=m["yaw_deg"], delta_yaw_deg=m.get("delta_yaw_deg", ""),
                    scale_ratio=m["scale_ratio"], b_px=m["b_px"],
                    covis_frac=m["covis_frac"], sec=round(sec, 2),
                    **evaluate(corr, pair["warp"], pair["mask"], m["gsd_b"]))
                writer.writerow(row)
                n_done += 1
                if n_done % 40 == 0:
                    fh.flush()
                    print(f"  [{name}] {n_done}/{len(files)}, "
                          f"{time.perf_counter()-t0:.0f} с", flush=True)
            fh.flush()
            del matcher
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            print(f"[{name}] готово: {n_done} строк за {time.perf_counter()-t0:.0f} с",
                  flush=True)
    print("готово:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
