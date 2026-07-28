"""Калибровка дискриминатора качества матча на максимуме реальных данных.

Проблема (см. JOURNAL, веха про офлайн-карту): в поиске по карте гейт позиции не
помогает, и отделять верный матч от ложного должно КАЧЕСТВО матча. Но верный матч
на низкой высоте/кросс-сезоне слаб (~5–8 инлайеров) и по счётчику не отличим от
случайного ложного. Нужен многосигнальный порог (NCC, RMSE, эллипс), откалиброванный
на реальных данных.

Метод — прямая генерация размеченных матчей (не полагаясь на успех пайплайна):
для каждого кадра с GPS матчим кадр против референса Esri
  • НА ИСТИНЕ → метка ВЕРНО (даже если матч слаб),
  • СО СДВИГОМ (несколько азимутов) → метка ЛОЖНО.
Гейт приора отключён (огромная σ), метка ставится по фактической ошибке. Так все
кадры с GPS дают и позитивы, и негативы, а сигналы (инлайеры, inlier_ratio, RMSE,
эллипс, NCC) собираются с каждого. Затем — разделяющие пороги по индексу Юдена.

    python scripts/calibrate_quality.py --data for_binding --out calib_quality.csv

Нужны: torch + LightGlue, Pillow, сеть (тайлы Esri; кэш в --cache).
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from aero_geoloc.basemap import ESRI_WORLD_IMAGERY, MissingTileError, TileBasemap, TileCache  # noqa: E402
from aero_geoloc.drone import basemap_zoom_for, frame_at_mpp, load_drone_shot  # noqa: E402
from aero_geoloc.geo import ground_mpp, haversine_m  # noqa: E402
from aero_geoloc.localize import localize_against_reference, normalize_gray  # noqa: E402
from aero_geoloc.matcher import LightGlueMatcher  # noqa: E402
from aero_geoloc.retrieval import MegaLocEncoder, calibrate_uniqueness_threshold  # noqa: E402

from map_benchmark import build_or_load_map, cluster_shots  # noqa: E402  (соседний скрипт)

FIELDS = ["dataset", "file", "altitude_m", "ref_kind", "label", "error_m",
          "n_inliers", "inlier_ratio", "reprojection_rmse_px", "ellipse_major_m", "photometric"]

# Сигналы и направление «лучше у верного»: +1 больше=лучше, −1 меньше=лучше.
SIGNALS = {
    "n_inliers": +1,
    "inlier_ratio": +1,
    "photometric": +1,
    "reprojection_rmse_px": -1,
    "ellipse_major_m": -1,
}


def _match_row(shot, ds_name, file_name, frame, camera, basemap, z, lat, lon, kind, args):
    """Сматчить кадр против референса @(lat,lon); вернуть строку сигналов или None."""
    mpp = ground_mpp(shot.true_lat, z)
    half = max(camera.footprint_m(shot.altitude_m))
    size = int(2 * half / mpp)
    try:
        ref, gref = basemap(lon, lat, z, size, size)
    except (MissingTileError, RuntimeError):
        return None
    # Приор с огромной σ — гейт позиции не срабатывает, сигналы считаются всегда.
    prior = shot.prior(sigma_m=1_000_000.0)
    r = localize_against_reference(
        frame, camera, prior, ref, gref, matcher=LightGlueMatcher(),
        prerotate_deg=-shot.yaw_deg, min_inliers=args.min_inliers, min_photometric=-1.0,
        ransac_threshold_px=6.0,
    )
    d = r.diagnostics
    if not r.is_localized or "n_inliers" not in d:
        return None  # не набралось даже слабой модели — сигналов нет
    err = haversine_m(shot.true_lat, shot.true_lon, r.center_lat, r.center_lon)
    if kind == "truth":
        label = 1 if err < args.true_m else (0 if err > args.false_m else None)
    else:
        label = 0 if err > args.false_m else None  # сдвинутый референс, но матч ушёл к истине? тогда неоднозначно
    if label is None:
        return None
    return {
        "dataset": ds_name, "file": file_name, "altitude_m": round(shot.altitude_m, 1),
        "ref_kind": kind, "label": label, "error_m": round(err, 1),
        "n_inliers": d.get("n_inliers", 0), "inlier_ratio": round(d.get("inlier_ratio", 0.0), 4),
        "reprojection_rmse_px": round(d.get("reprojection_rmse_px", float("nan")), 4),
        "ellipse_major_m": round(r.error_ellipse_m[0], 3),
        "photometric": round(d.get("photometric", float("nan")), 4),
    }


def collect(args) -> list[dict]:
    """Позитивы — референс на истине; ТРУДНЫЕ негативы — похожие клетки из top-K ретривала.

    Случайно сдвинутый референс матчер и так отвергает (лёгкий негатив) — ложные
    срабатывания рождаются от appearance-похожих клеток, которые выдаёт Этаж 1.
    Поэтому негативы берём из top-K индекса (клетки далеко от истины), а не из сдвига.
    """
    mz = ESRI_WORLD_IMAGERY.max_zoom
    basemap = TileBasemap(cache=TileCache(args.cache))
    encoder = MegaLocEncoder()
    rows: list[dict] = []

    for ds in sorted(p for p in Path(args.data).iterdir() if p.is_dir()):
        files = sorted(ds.glob("*.JPG"))
        named = []
        for path in files:
            try:
                shot = load_drone_shot(str(path))
            except Exception:  # noqa: BLE001
                continue
            if shot.is_nadir and shot.true_lat is not None:
                named.append((path.name, shot))
        if not named:
            continue  # набор без GPS/надира (images, quary_images)
        if len(named) > args.max_per_ds:
            step = len(named) // args.max_per_ds
            named = named[::step]

        # Отдельная namespace для сборки карт (совместима с map_benchmark).
        margs = argparse.Namespace(
            data=str(ds), maps_dir=args.maps_dir, map_radius_km=args.map_radius_km,
            index_mpp=0.37, index_cell_m=125.0, overlap=0.5, pca_dim=1024,
            ef_search=128, rebuild=False,
        )
        clusters = cluster_shots(named, max_km=args.cluster_km)
        print(f"[{ds.name}] {len(named)} кадров → {len(clusters)} кластер(ов)", flush=True)
        for i, cluster in enumerate(clusters):
            index, _ = build_or_load_map(cluster, i, margs, basemap, encoder, mz)
            for name, shot in cluster["shots"]:
                z = basemap_zoom_for(shot, max_zoom=mz)
                frame, camera = frame_at_mpp(shot, ground_mpp(shot.true_lat, z))
                # ВЕРНО: референс на истине.
                row = _match_row(shot, ds.name, name, frame, camera, basemap, z,
                                 shot.true_lat, shot.true_lon, "truth", args)
                if row:
                    rows.append(row)
                # ТРУДНЫЕ ЛОЖНЫЕ: похожие клетки из top-K, далёкие от истины.
                rr = index.query(normalize_gray(frame), k=args.top_k, prerotate_deg=-shot.yaw_deg)
                taken = 0
                for cell in rr.cells:
                    if taken >= args.max_neg:
                        break
                    if haversine_m(shot.true_lat, shot.true_lon, cell.center_lat, cell.center_lon) < args.false_m:
                        continue  # клетка у истины — не негатив
                    row = _match_row(shot, ds.name, name, frame, camera, basemap, z,
                                     cell.center_lat, cell.center_lon, "false", args)
                    if row:
                        rows.append(row); taken += 1
                pos = sum(1 for r in rows if r["dataset"] == ds.name and r["label"] == 1)
                neg = sum(1 for r in rows if r["dataset"] == ds.name and r["label"] == 0)
                print(f"  {ds.name}/{name} alt={shot.altitude_m:.0f}: верных={pos} трудн.ложных={neg}", flush=True)
    return rows


def report(rows: list[dict]) -> None:
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    print(f"\n{'='*70}\nКАЛИБРОВКА КАЧЕСТВА: {len(pos)} верных, {len(neg)} ложных матчей\n{'='*70}")
    print("\nпо датасетам (верных / ложных):")
    for ds in sorted({r["dataset"] for r in rows}):
        p = sum(1 for r in pos if r["dataset"] == ds)
        n = sum(1 for r in neg if r["dataset"] == ds)
        print(f"  {ds:14} верных={p:4d}  ложных={n:4d}")
    if not pos or not neg:
        print("\nнет одновременно верных и ложных — калибровка невозможна")
        return

    print("\nразделяющая сила сигналов (порог Юдена; TPR удержано верных, FPR пропущено ложных):")
    scored = []
    for sig, direction in SIGNALS.items():
        vals, labs = [], []
        for r in rows:
            v = r[sig]
            if isinstance(v, float) and math.isnan(v):
                continue
            vals.append(direction * float(v))
            labs.append(r["label"])
        if len(set(labs)) < 2:
            continue
        thr = calibrate_uniqueness_threshold(vals, labs)
        keep = np.array(vals) >= thr
        y = np.array(labs, dtype=bool)
        tpr = int((keep & y).sum()) / max(int(y.sum()), 1)
        fpr = int((keep & ~y).sum()) / max(int((~y).sum()), 1)
        youden = tpr - fpr
        real_thr = direction * thr  # порог в исходных единицах
        cmp = "≥" if direction > 0 else "≤"
        scored.append((youden, sig, cmp, real_thr, tpr, fpr))
    for youden, sig, cmp, thr, tpr, fpr in sorted(scored, reverse=True):
        # медианы по классам для наглядности
        mp = statistics.median([float(r[sig]) for r in pos if not (isinstance(r[sig], float) and math.isnan(r[sig]))])
        mn = statistics.median([float(r[sig]) for r in neg if not (isinstance(r[sig], float) and math.isnan(r[sig]))])
        print(f"  {sig:22} {cmp} {thr:8.3f}  Youden={youden:.2f}  (верно {tpr:.0%} / ложно {fpr:.0%})"
              f"  медианы: верн={mp:.3f} лож={mn:.3f}")

    print(f"\nВЫВОД: сигнал с наибольшим Youden — лучший одиночный дискриминатор; "
          f"комбинировать top-2 как замену грубому min_inliers в quality.assess.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="for_binding")
    parser.add_argument("--cache", default="tiles")
    parser.add_argument("--maps-dir", default="maps", help="офлайн-карты (переиспользуются с map_benchmark)")
    parser.add_argument("--map-radius-km", type=float, default=2.0)
    parser.add_argument("--cluster-km", type=float, default=10.0)
    parser.add_argument("--max-per-ds", type=int, default=60, help="кадров с датасета (прореживание крупных)")
    parser.add_argument("--top-k", type=int, default=25, help="сколько клеток ретривала смотреть для негативов")
    parser.add_argument("--max-neg", type=int, default=5, help="макс. трудных негативов на кадр")
    parser.add_argument("--min-inliers", type=int, default=4, help="низкий порог для сбора даже слабых моделей")
    parser.add_argument("--true-m", type=float, default=50.0, help="ошибка < → ВЕРНО")
    parser.add_argument("--false-m", type=float, default=150.0, help="ошибка > → ЛОЖНО (между — пропуск)")
    parser.add_argument("--out", default="calib_quality.csv")
    args = parser.parse_args()

    print(f"сбор: до {args.max_per_ds} кадров/датасет, негативы из top-{args.top_k} ретривала (до {args.max_neg}/кадр)")
    rows = collect(args)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    report(rows)
    print(f"\nсырьё → {args.out} ({len(rows)} строк)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
