"""Географические кластеры площадок и сплиты train / val / heldout.

Делить по площадке недостаточно: соседние вылеты снимают смежную территорию, и
площадка из train может оказаться в трёхстах метрах от площадки в held-out —
модель увидит те же дома по обе стороны, а прибавка окажется утечкой.

Поэтому единица деления — **географический кластер**: связные компоненты графа,
где площадки соединены, если их центры ближе порога **или** пересекаются их
габариты. Второе условие важнее первого: две крупные площадки с центрами в
трёх километрах могут перекрываться краями.

Сплиты набираются не по проценту, а **по различимости**: held-out должен быть
таким, чтобы парное сравнение двух чекпоинтов различало интересную прибавку, а
не тонуло в шуме доли.

    python open_orto/scripts/geo_split.py --seed 20260831
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

from scenes_dedup import haversine_m  # noqa: E402

#: Порог соседства центров. Два километра — расстояние, на котором соседние
#: вылеты ещё снимают общую застройку.
NEAR_KM = 2.0
#: Целевые размеры в боевых парах. 640 даёт порог различимости 0.028 по доле
#: успешных пар при p ≈ 0.39; ниже 400 порог уползает за 0.035, и половина
#: интересных прибавок станет неотличима от шума.
HELDOUT_MIN = 640
VAL_MIN = 400
#: Доля успеха базового ядра — от неё считается порог различимости.
BASE_P = 0.39


def discriminable(n: int, p: float = BASE_P) -> float:
    """Порог различимости доли при парном сравнении на n парах."""
    if n <= 0:
        return float("nan")
    se = math.sqrt(p * (1 - p) / n)
    return 1.96 * se * math.sqrt(2) / 2


def bbox_overlap(a, b) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def build_clusters(places: dict, near_km: float = NEAR_KM):
    """Связные компоненты графа соседства: {площадка: id кластера}."""
    names = sorted(places)
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, a in enumerate(names):
        pa = places[a]
        for b in names[i + 1:]:
            pb = places[b]
            # грубый отсев по широте до дорогой формулы расстояния
            if abs(pa["lat"] - pb["lat"]) > near_km / 111.0 + 0.5:
                continue
            near = haversine_m(pa["lon"], pa["lat"], pb["lon"], pb["lat"]) <= near_km * 1000
            if near or bbox_overlap(pa["bbox"], pb["bbox"]):
                union(a, b)
    comp = defaultdict(list)
    for n in names:
        comp[find(n)].append(n)
    return {n: f"c{i:04d}" for i, (_, members) in
            enumerate(sorted(comp.items(), key=lambda kv: (-len(kv[1]), kv[0])))
            for n in members}, comp


def cluster_profile(rows):
    """Профиль набора пар для стратификации: доли категорий и медианы."""
    bm = [r for r in rows if r["pair_kind"] != "same_source"]
    prof = {}
    total = max(len(bm), 1)
    for k in ("layout", "вердикт", "corpus"):
        for v, n in Counter(r.get(k, "") for r in bm).items():
            prof[f"{k}={v}"] = n / total
    for k in ("height_m", "tilt_deg"):
        vals = [float(r[k]) for r in bm if r.get(k)]
        prof[f"med_{k}"] = float(np.median(vals)) if vals else 0.0
    return prof


def profile_gap(a: dict, b: dict) -> float:
    """Насколько профиль сплита отличается от корпусного (меньше — лучше)."""
    keys = set(a) | set(b)
    gap = 0.0
    for k in keys:
        va, vb = a.get(k, 0.0), b.get(k, 0.0)
        gap += abs(va - vb) / (300.0 if k.startswith("med_") else 1.0)
    return gap


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scan", default="open_orto/work/rasters_scan_all.csv")
    ap.add_argument("--manifest", default="openaerialmap_dataset/manifest.csv")
    ap.add_argument("--near-km", type=float, default=NEAR_KM)
    ap.add_argument("--seed", type=int, default=20260831)
    ap.add_argument("--tries", type=int, default=400,
                    help="сколько раскладок перебрать: берётся та, чей профиль "
                         "ближе к корпусному")
    ap.add_argument("--out-json", default="openaerialmap_dataset/splits.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    from cpu_affinity import pin_to_performance
    pin_to_performance(verbose=False)

    man = list(csv.DictReader(Path(args.manifest).open(encoding="utf-8")))
    used = {r["scene"] for r in man}
    places = {}
    for r in csv.DictReader(Path(args.scan).open(encoding="utf-8")):
        if r["name"] in used and r["lon"] and r["lon_min"]:
            places[r["name"]] = dict(
                lon=float(r["lon"]), lat=float(r["lat"]),
                bbox=(float(r["lon_min"]), float(r["lat_min"]),
                      float(r["lon_max"]), float(r["lat_max"])))
    print(f"площадок: {len(used)}, с геометрией: {len(places)}")

    cl, comp = build_clusters(places, args.near_km)
    sizes = Counter(cl.values())
    print(f"кластеров: {len(sizes)} | крупнейший: {max(sizes.values())} площадок")
    hist = Counter(sizes.values())
    print("  размеры: " + ", ".join(f"{k} площадк(и) — {v} кластеров"
                                    for k, v in sorted(hist.items())))

    # дубли обязаны лежать в одном кластере — это проверка построения
    by_dup = defaultdict(set)
    for r in man:
        if r.get("dup_size") and int(r["dup_size"]) > 1:
            by_dup[r["dup_group"]].add(r["scene"])
    broken = [g for g, sc in by_dup.items() if len({cl.get(s) for s in sc}) > 1]
    print(f"групп дублей: {len(by_dup)}, разорванных кластеризацией: {len(broken)}")

    rows_by_cluster = defaultdict(list)
    for r in man:
        rows_by_cluster[cl.get(r["scene"], "c_none")].append(r)
    combat = {c: sum(1 for r in rs if r["pair_kind"] != "same_source")
              for c, rs in rows_by_cluster.items()}
    corpus_prof = cluster_profile(man)

    # ——— раскладка кластеров по сплитам
    rng = np.random.default_rng(args.seed)
    clusters = sorted(rows_by_cluster)
    best = None
    for _ in range(args.tries):
        order = list(clusters)
        rng.shuffle(order)
        assign, got = {}, {"heldout": 0, "val": 0}
        # крупный кластер, попав в оценочный сплит, раздувает его втрое и
        # забирает у обучения территории: ограничиваем вклад одного кластера
        # третью цели, такие уходят в train
        cap = {"heldout": HELDOUT_MIN * 0.34, "val": VAL_MIN * 0.34}
        for c in order:
            for sp, need in (("heldout", HELDOUT_MIN), ("val", VAL_MIN)):
                if got[sp] < need and combat[c] <= cap[sp]:
                    assign[c] = sp
                    got[sp] += combat[c]
                    break
            else:
                assign[c] = "train"
        if got["heldout"] < HELDOUT_MIN or got["val"] < VAL_MIN:
            continue
        gap = sum(profile_gap(cluster_profile([r for c, rs in rows_by_cluster.items()
                                               if assign[c] == sp for r in rs]),
                              corpus_prof) for sp in ("heldout", "val"))
        if best is None or gap < best[0]:
            best = (gap, dict(assign), dict(got))
    if best is None:
        print("не удалось набрать сплиты нужного размера")
        return 1
    gap, assign, got = best
    print(f"\nвыбрана раскладка: расхождение профиля {gap:.3f} "
          f"(из {args.tries} попыток, сид {args.seed})")

    # ——— итоги
    stats = {}
    for sp in ("train", "val", "heldout"):
        rs = [r for c, rows in rows_by_cluster.items() if assign[c] == sp for r in rows]
        bm = [r for r in rs if r["pair_kind"] != "same_source"]
        h = [float(r["height_m"]) for r in bm]
        stats[sp] = dict(
            clusters=sum(1 for c in clusters if assign[c] == sp),
            scenes=len({r["scene"] for r in rs}), pairs=len(rs), combat=len(bm),
            same_source=len(rs) - len(bm),
            partial=round(100 * np.mean([r["layout"] == "partial" for r in bm])),
            height_med=round(float(np.median(h))) if h else 0,
            discriminable=round(discriminable(len(bm)), 4))
    print(f"\n{'сплит':9} {'кластеров':>10} {'площадок':>9} {'пар':>6} "
          f"{'боевых':>7} {'partial':>8} {'H мед':>6} {'порог':>7}")
    for sp, d in stats.items():
        print(f"{sp:9} {d['clusters']:10} {d['scenes']:9} {d['pairs']:6} "
              f"{d['combat']:7} {d['partial']:7}% {d['height_med']:6} "
              f"{d['discriminable']:7.3f}")

    if args.dry_run:
        return 0

    head = list(man[0].keys())
    for col in ("geo_cluster", "split"):
        if col not in head:
            head.append(col)
    for r in man:
        c = cl.get(r["scene"], "c_none")
        r["geo_cluster"], r["split"] = c, assign[c]
    with Path(args.manifest).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=head)
        w.writeheader()
        w.writerows(man)
    print(f"\nманифест: {len(man)} строк, колонки geo_cluster/split")

    Path(args.out_json).write_text(json.dumps({
        "seed": args.seed, "near_km": args.near_km,
        "clusters": len(sizes), "largest_cluster": max(sizes.values()),
        "dup_groups": len(by_dup), "dup_groups_broken": len(broken),
        "splits": stats,
        "cluster_of_scene": cl,
        "split_of_cluster": assign,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"сплиты: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
