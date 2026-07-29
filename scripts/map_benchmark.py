"""Офлайн-карта местности: строим/сохраняем индексы, ищем каждый снимок, замеряем время.

Двухфазный сценарий «поиск снимка по офлайн-карте»:

  ОФЛАЙН (один раз, кэшируется на диск): регион вокруг кластера снимков →
  нарезка на клетки → MegaLoc-энкодер → PCA→1024 → сохранение в ``.npz``
  (кодировки клеток + состояние PCA). Дорогое кодирование не повторяется.

  ОНЛАЙН (на каждый снимок): загрузка карты из ``.npz`` (без пере-кодирования и
  без докачки тайлов) → FAISS/HNSW-поиск top-K клеток по кадру → LightGlue-поза.
  Приор — центр карты с σ во всю карту, т.е. честный поиск по всей карте, а не
  подсказка. Замеряется среднее время поиска.

Снимки автоматически группируются в кластеры по близости (карта на кластер).

    python scripts/map_benchmark.py --data for_binding/DRZ --map-radius-km 2 --top-k 25
    python scripts/map_benchmark.py --data for_binding/DRZ --rebuild   # пересобрать карты

Нужны: torch + MegaLoc + LightGlue, faiss-cpu, Pillow, сеть (только при первой сборке).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aero_geoloc.basemap import ESRI_WORLD_IMAGERY, TileBasemap, TileCache  # noqa: E402
from aero_geoloc.drone import basemap_zoom_for, frame_at_mpp, load_drone_shot  # noqa: E402
from aero_geoloc.geo import Georef, ground_mpp, haversine_m, zoom_for_mpp  # noqa: E402
from aero_geoloc.localize import localize, normalize_gray  # noqa: E402
from aero_geoloc.matcher import LightGlueMatcher  # noqa: E402
from aero_geoloc.retrieval import MegaLocEncoder, TerrainIndex  # noqa: E402
from aero_geoloc.types import Status  # noqa: E402


def cluster_shots(named_shots, *, max_km: float):
    """Сгруппировать ``(имя, shot)`` в кластеры: снимок входит в кластер, если он
    ближе ``max_km`` к его текущему центроиду; иначе заводится новый кластер."""
    clusters: list[dict] = []
    for name, s in named_shots:
        placed = False
        for c in clusters:
            if haversine_m(c["lat"], c["lon"], s.true_lat, s.true_lon) < max_km * 1000.0:
                c["shots"].append((name, s))
                c["lat"] = sum(x.true_lat for _, x in c["shots"]) / len(c["shots"])
                c["lon"] = sum(x.true_lon for _, x in c["shots"]) / len(c["shots"])
                placed = True
                break
        if not placed:
            clusters.append({"lat": s.true_lat, "lon": s.true_lon, "shots": [(name, s)]})
    return clusters


def build_or_load_map(cluster, i, args, basemap, encoder, mz):
    """Вернуть (index, offline_seconds|None). Строит и сохраняет карту, либо грузит с диска."""
    map_path = Path(args.maps_dir) / f"{Path(args.data).name}_cluster{i}.npz"
    if map_path.exists() and not args.rebuild:
        index = TerrainIndex.load(map_path, encoder)
        index.use_faiss(kind="hnsw", ef_search=args.ef_search)
        print(f"  карта загружена с диска: {map_path.name} ({len(index)} клеток, dim={index._reducer.dim})")
        return index, None

    # Регион вокруг центроида кластера на грубом зуме (клетка ≈ footprint кадра).
    z_index = zoom_for_mpp(args.index_mpp, cluster["lat"], max_zoom=mz)
    mpp_index = ground_mpp(cluster["lat"], z_index)
    cell_px = max(32, round(args.index_cell_m / mpp_index))
    region_px = int(2 * args.map_radius_km * 1000.0 / mpp_index)
    region = Georef(cluster["lon"], cluster["lat"], z_index, region_px, region_px)

    t0 = time.perf_counter()
    index = TerrainIndex(encoder).build(
        basemap, region, cell_size_px=cell_px, overlap=args.overlap, rotations_deg=(0.0,)
    )
    t_encode = time.perf_counter() - t0
    t0 = time.perf_counter()
    index.compress(args.pca_dim, whiten=False)
    t_pca = time.perf_counter() - t0
    Path(args.maps_dir).mkdir(parents=True, exist_ok=True)
    index.save(map_path)  # кодировки клеток + PCA → диск (в т.ч. для FAISS)
    index.use_faiss(kind="hnsw", ef_search=args.ef_search)
    offline_s = t_encode + t_pca
    print(f"  карта построена и сохранена: {map_path.name} ({len(index)} клеток @z{z_index}, "
          f"кодирование {t_encode:.0f}с [{1000*t_encode/max(len(index),1):.0f}мс/кл], PCA {t_pca:.0f}с)")
    return index, offline_s


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="for_binding/DRZ", help="каталог снимков")
    parser.add_argument("--maps-dir", default="maps", help="куда класть офлайн-карты (.npz)")
    parser.add_argument("--map-radius-km", type=float, default=2.0, help="радиус офлайн-карты вокруг кластера")
    parser.add_argument("--cluster-km", type=float, default=10.0, help="порог группировки снимков в кластеры")
    parser.add_argument("--index-mpp", type=float, default=0.37)
    parser.add_argument("--index-cell-m", type=float, default=125.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--pca-dim", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--min-inliers", type=int, default=6)
    parser.add_argument("--ef-search", type=int, default=128)
    parser.add_argument("--rebuild", action="store_true", help="пересобрать карты, даже если .npz есть")
    parser.add_argument("--cache", default="tiles")
    args = parser.parse_args()

    mz = ESRI_WORLD_IMAGERY.max_zoom
    basemap = TileBasemap(cache=TileCache(args.cache))
    encoder = MegaLocEncoder()  # один экземпляр (ленивая загрузка весов один раз)

    files = sorted(Path(args.data).glob("*.JPG"))
    shots = []
    for f in files:
        try:
            s = load_drone_shot(str(f))
        except Exception as e:  # noqa: BLE001
            print(f"{f.name}: ошибка чтения — {e}")
            continue
        if s.is_nadir:
            shots.append((f.name, s))
        else:
            print(f"{f.name}: не надирный — пропущен")
    if not shots:
        print("нет надирных снимков")
        return 1

    clusters = cluster_shots(shots, max_km=args.cluster_km)
    print(f"\n{len(shots)} надирных снимков → {len(clusters)} кластер(ов) "
          f"(карта {2*args.map_radius_km:.0f}×{2*args.map_radius_km:.0f} км на кластер)\n")

    rows = []
    offline_total = 0.0
    for i, cluster in enumerate(clusters):
        print(f"[Кластер {i}] {len(cluster['shots'])} снимков вокруг {cluster['lat']:.4f},{cluster['lon']:.4f}")
        index, offline_s = build_or_load_map(cluster, i, args, basemap, encoder, mz)
        if offline_s:
            offline_total += offline_s

        for name, s in cluster["shots"]:
            z_fine = basemap_zoom_for(s, max_zoom=mz)
            frame, camera = frame_at_mpp(s, ground_mpp(s.true_lat, z_fine))
            # Приор = центр карты, σ во всю карту → поиск по ВСЕЙ карте (не подсказка).
            sigma_m = args.map_radius_km * 1000.0
            prior = s.prior(sigma_m=sigma_m)
            prior = type(prior)(lat=cluster["lat"], lon=cluster["lon"], sigma_m=sigma_m,
                                altitude_m=prior.altitude_m, altitude_sigma_m=prior.altitude_sigma_m,
                                yaw_deg=prior.yaw_deg, pitch_deg=prior.pitch_deg, roll_deg=prior.roll_deg)
            offset_km = haversine_m(s.true_lat, s.true_lon, cluster["lat"], cluster["lon"]) / 1000.0

            t0 = time.perf_counter()
            r = localize(frame, camera, prior, basemap, index=index, matcher=LightGlueMatcher(),
                         prerotate=True, max_zoom=mz, min_photometric=0.12, min_inliers=args.min_inliers,
                         retrieval_top_k=args.top_k, ransac_threshold_px=6.0)
            search_s = time.perf_counter() - t0
            # Поиск по карте: принимаем только LOCALIZED (связка калиброванного качества);
            # LOW_CONFIDENCE = ненадёжная привязка → отказ, ложная точка опаснее.
            if r.status is Status.LOCALIZED:
                err = haversine_m(s.true_lat, s.true_lon, r.center_lat, r.center_lon)
                status = f"{err:5.1f} м"
            else:
                err = None
                status = "ОТКАЗ"
            rows.append((name, offset_km, status, r.status.value, search_s, err))
            print(f"    {name:<16} сдвиг от центра {offset_km:.2f}км → {status:<8} "
                  f"({r.status.value}) за {search_s:.1f}с")

    # --- сводка ---
    ok = [r for r in rows if r[5] is not None]
    times = [r[4] for r in rows]
    print("\n=== СВОДКА ===")
    print(f"снимков: {len(rows)}, локализовано: {len(ok)}/{len(rows)}")
    if ok:
        errs = [r[5] for r in ok]
        print(f"ошибка (лок.): медиана {statistics.median(errs):.1f} м, макс {max(errs):.1f} м")
    if offline_total:
        print(f"офлайн (сборка карт, один раз): {offline_total:.0f} с")
    print(f"ВРЕМЯ ПОИСКА по офлайн-карте: среднее {statistics.mean(times):.1f} с, "
          f"медиана {statistics.median(times):.1f} с (min {min(times):.1f}, max {max(times):.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
