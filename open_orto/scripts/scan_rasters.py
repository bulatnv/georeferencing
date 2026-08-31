"""Скан каталога ортофотопланов: паспорта растров и проверка на дубли.

Имя файла ничего не говорит о содержимом: один и тот же ортоплан приходит под
разными идентификаторами. Поэтому дубли ищутся **по географии** — центр растра
переводится в WGS84 и сравнивается с местами уже обработанных площадок,
которые известны из метаданных собранных пар (`anchor_xy` плюс CRS площадки).

Совпадение по координате — ещё не приговор: соседние вылеты одной территории
законно перекрываются. Поэтому кандидат считается дублем, только если рядом
оказался центр старой площадки **и** совпал размер растра — тогда это тот же
исходник, а не соседний участок.

    python open_orto/scripts/scan_rasters.py --data-dir E:/open_ortophoto_data \\
        --out open_orto/work/rasters_scan_new.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

#: Насколько близко должны сойтись центры, чтобы заподозрить дубль (м).
NEAR_M = 800.0
#: Допуск на совпадение размера растра в пикселях (доля).
SIZE_TOL = 0.02

FIELDS = ["name", "w", "h", "bands", "crs", "res", "span_x", "span_y", "km2", "mb",
          "lon", "lat", "lon_min", "lat_min", "lon_max", "lat_max",
          "дубль", "совпадает_с"]


def passport(path: Path) -> dict | None:
    """Паспорт растра: размеры, CRS, разрешение, площадь и центр в WGS84."""
    try:
        with rasterio.open(path) as ds:
            crs = str(ds.crs) if ds.crs else ""
            res = float(ds.res[0]) if ds.res else 0.0
            metric = bool(ds.crs and ds.crs.is_projected)
            b = ds.bounds
            span_x = (b.right - b.left) if metric else 0.0
            span_y = (b.top - b.bottom) if metric else 0.0
            cx, cy = (b.left + b.right) / 2, (b.bottom + b.top) / 2
            lon = lat = None
            bbox = (None, None, None, None)
            if ds.crs:
                try:
                    # центр и габарит разом: габарит нужен кластеризации —
                    # две крупные площадки с центрами в 3 км могут
                    # перекрываться, и по одному центру этого не увидеть
                    xs, ys = warp_transform(ds.crs, "EPSG:4326",
                                            [cx, b.left, b.right], [cy, b.bottom, b.top])
                    lon, lat = float(xs[0]), float(ys[0])
                    bbox = (min(xs[1:]), min(ys[1:]), max(xs[1:]), max(ys[1:]))
                except Exception:  # noqa: BLE001
                    pass
            return dict(name=path.stem, w=ds.width, h=ds.height, bands=ds.count,
                        crs=crs, res=round(res, 4) if metric else 0.0,
                        span_x=round(span_x), span_y=round(span_y),
                        km2=round(span_x * span_y / 1e6, 3),
                        mb=round(path.stat().st_size / 2**20),
                        lon=round(lon, 6) if lon is not None else "",
                        lat=round(lat, 6) if lat is not None else "",
                        **{k: (round(v, 6) if v is not None else "")
                           for k, v in zip(("lon_min", "lat_min", "lon_max", "lat_max"), bbox)})
    except Exception as exc:  # noqa: BLE001
        print(f"  {path.name}: не открылся — {exc}", flush=True)
        return None


def known_places(datasets, old_scan: Path | None):
    """Где уже собраны пары: [(lon, lat, имя площадки, w, h)] по метаданным.

    Координата площадки берётся из `anchor_xy` любой её пары, а CRS — из
    прежнего скана: сами растры к этому моменту с диска могли исчезнуть, но
    места, которые они покрывали, датасет помнит.
    """
    crs_by_name, size_by_name = {}, {}
    if old_scan and old_scan.exists():
        for r in csv.DictReader(old_scan.open(encoding="utf-8")):
            crs_by_name[r["name"]] = r["crs"]
            size_by_name[r["name"]] = (int(r["w"]), int(r["h"]))

    seen, out = {}, []
    for root in datasets:
        root = Path(root)
        if not root.exists():
            continue
        for f in sorted(root.glob("*.npz")):
            tag = f.name.split("_")[1] if "_" in f.name else ""
            if tag in seen:
                continue
            try:
                meta = json.loads(str(np.load(f, allow_pickle=False)["meta"]))
            except Exception:  # noqa: BLE001
                continue
            scene = meta.get("scene", "")
            if scene in seen:
                continue
            xy = meta.get("anchor_xy")
            crs = crs_by_name.get(scene)
            if not xy or not crs:
                continue
            try:
                xs, ys = warp_transform(crs, "EPSG:4326", [xy[0]], [xy[1]])
            except Exception:  # noqa: BLE001
                continue
            seen[scene] = True
            seen[tag] = True
            out.append((float(xs[0]), float(ys[0]), scene) + size_by_name.get(scene, (0, 0)))
    return out


def haversine_m(lon1, lat1, lon2, lat2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default="E:/open_ortophoto_data")
    ap.add_argument("--out", default="open_orto/work/rasters_scan_new.csv")
    ap.add_argument("--skip-dupcheck", action="store_true",
                    help="только паспорта: сверка с уже обработанными площадками "
                         "читает метаданные тысяч пар и здесь не нужна")
    ap.add_argument("--old-scan", default="open_orto/work/rasters_scan.csv")
    ap.add_argument("--datasets", nargs="*", default=[
        "open_orto/dataset_base", "open_orto/dataset_ss", "open_orto/dataset",
        "open_orto/dataset_base_quarantine"])
    args = ap.parse_args()
    from cpu_affinity import pin_to_performance
    pin_to_performance(verbose=False)

    files = sorted(Path(args.data_dir).glob("*.tif"))
    print(f"растров в каталоге: {len(files)}", flush=True)

    places = []
    if not args.skip_dupcheck:
        print("собираю места уже обработанных площадок...", flush=True)
        places = known_places(args.datasets, Path(args.old_scan))
    print(f"известных площадок с координатой: {len(places)}", flush=True)

    rows, dup = [], 0
    for i, f in enumerate(files, 1):
        p = passport(f)
        if not p:
            continue
        p["дубль"], p["совпадает_с"] = "", ""
        if p["lon"] != "":
            for lon, lat, scene, w, h in places:
                d = haversine_m(p["lon"], p["lat"], lon, lat)
                if d > NEAR_M:
                    continue
                same_size = (w and h
                             and abs(p["w"] - w) <= SIZE_TOL * max(p["w"], w)
                             and abs(p["h"] - h) <= SIZE_TOL * max(p["h"], h))
                p["дубль"] = "да" if same_size else "рядом"
                p["совпадает_с"] = f"{scene} ({d:.0f} м)"
                dup += p["дубль"] == "да"
                break
        rows.append(p)
        if i % 100 == 0:
            print(f"  {i}/{len(files)}, дублей {dup}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    metric = [r for r in rows if float(r["km2"]) > 0]
    near = [r for r in rows if r["дубль"] == "рядом"]
    print(f"\nвсего {len(rows)}: в метрических CRS {len(metric)}, "
          f"дублей {dup}, рядом с обработанными (перекрытие) {len(near)}")
    print(f"скан: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
