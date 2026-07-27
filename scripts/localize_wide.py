"""Локализация кадра по ГРУБОМУ приору ±2–5 км — исходная задача проекта.

GPS снимка используется только как ground truth и как центр, от которого приор
искусственно **сдвигается** на 2–5 км (имитация: точного положения нет, есть
лишь грубая область). Двухэтажная система: DINOv2-индекс региона (Этаж 1)
схлопывает диск в top-K клеток по внешнему виду, LightGlue (Этаж 2) уточняет
позу. Индекс строится на грубом зуме так, что клетка ≈ footprint кадра, — иначе
регион ±5 км неподъёмен по числу клеток/тайлов.

    python scripts/localize_wide.py --image test_images/00049.JPG --offset-km 2 --bearing 60

Нужны: torch + LightGlue + DINOv2 (torch.hub), Pillow, сеть (тайлы Esri).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from aero_geoloc.basemap import ESRI_WORLD_IMAGERY, TileBasemap, TileCache  # noqa: E402
from aero_geoloc.drone import basemap_zoom_for, frame_at_mpp, load_drone_shot  # noqa: E402
from aero_geoloc.geo import Georef, ground_mpp, haversine_m, zoom_for_mpp  # noqa: E402
from aero_geoloc.localize import localize, normalize_gray  # noqa: E402
from aero_geoloc.matcher import LightGlueMatcher  # noqa: E402
from aero_geoloc.retrieval import DinoV2Encoder, MegaLocEncoder, TerrainIndex  # noqa: E402

_ENCODERS = {"dinov2": DinoV2Encoder, "megaloc": MegaLocEncoder}


def _offset_lonlat(lat, lon, distance_m, bearing_deg):
    """Сдвинуть точку на distance_m по азимуту bearing (плоское приближение)."""
    d_north = distance_m * math.cos(math.radians(bearing_deg))
    d_east = distance_m * math.sin(math.radians(bearing_deg))
    lat2 = lat + d_north / 111320.0
    lon2 = lon + d_east / (111320.0 * math.cos(math.radians(lat)))
    return lat2, lon2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", required=True)
    parser.add_argument("--offset-km", type=float, default=2.0, help="на сколько сдвинуть приор от истины")
    parser.add_argument("--bearing", type=float, default=45.0, help="азимут сдвига приора, °")
    parser.add_argument("--sigma-km", type=float, default=0.0, help="σ приора (0 → offset·1.5)")
    parser.add_argument("--index-cell-m", type=float, default=125.0, help="footprint клетки индекса, м")
    parser.add_argument("--index-mpp", type=float, default=0.75, help="разрешение клеток индекса (грубее = меньше клеток)")
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--margin-km", type=float, default=0.6, help="запас региона сверх сдвига")
    parser.add_argument("--encoder", default="dinov2", choices=sorted(_ENCODERS),
                        help="ядро Этажа 1: dinov2 (сырой) или megaloc (VPR)")
    parser.add_argument("--min-inliers", type=int, default=10,
                        help="порог similarity-инлайеров точного уровня (низкая высота/кросс-дата → 6)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="сколько клеток-кандидатов брать у ретривала (большой регион → ранг верной растёт)")
    parser.add_argument("--pca-dim", type=int, default=0, help="сжать дескриптор PCA→N (0 = без PCA)")
    parser.add_argument("--whiten", action="store_true", help="PCA с whitening (измерено: обычно вредит нашему дескриптору)")
    parser.add_argument("--faiss", action="store_true", help="поиск через FAISS/HNSW вместо numpy")
    parser.add_argument("--faiss-kind", default="hnsw", choices=["hnsw", "flat"])
    parser.add_argument("--ef-search", type=int, default=128, help="efSearch для HNSW (точность/скорость)")
    parser.add_argument("--dem", action="store_true")
    parser.add_argument("--declination", type=float, default=0.0)
    parser.add_argument("--cache", default="tiles")
    args = parser.parse_args()

    mz = ESRI_WORLD_IMAGERY.max_zoom
    tm = {}  # тайминги стадий, с
    t0 = time.perf_counter()
    shot = load_drone_shot(args.image, use_dem=args.dem, magnetic_declination_deg=args.declination)
    if not shot.is_nadir:
        print(f"кадр не надирный (наклон {shot.pitch_from_nadir_deg:.0f}°) — вне модели")
        return 1
    basemap = TileBasemap(cache=TileCache(args.cache))

    # Кадр в разрешении подложки (для точного уровня и retrieval-запроса).
    z_fine = basemap_zoom_for(shot, max_zoom=mz)
    frame, camera = frame_at_mpp(shot, ground_mpp(shot.true_lat, z_fine))
    tm["загрузка снимка + рендер кадра"] = time.perf_counter() - t0

    # Приор сдвигаем от истины на offset-км.
    offset_m = args.offset_km * 1000.0
    prior_lat, prior_lon = _offset_lonlat(shot.true_lat, shot.true_lon, offset_m, args.bearing)
    sigma_m = (args.sigma_km * 1000.0) if args.sigma_km > 0 else offset_m * 1.5
    prior = shot.prior(sigma_m=sigma_m)
    prior = type(prior)(lat=prior_lat, lon=prior_lon, sigma_m=sigma_m, altitude_m=prior.altitude_m,
                        altitude_sigma_m=prior.altitude_sigma_m, yaw_deg=prior.yaw_deg,
                        pitch_deg=prior.pitch_deg, roll_deg=prior.roll_deg)

    # Индекс региона вокруг ПРИОРА, покрывающий истину. Клетки на грубом зуме, но
    # с тем же footprint, что и кадр, — чтобы эмбеддинги были сопоставимы.
    z_index = zoom_for_mpp(args.index_mpp, prior_lat, max_zoom=mz)
    mpp_index = ground_mpp(prior_lat, z_index)
    cell_px = max(32, round(args.index_cell_m / mpp_index))
    radius_m = offset_m + args.margin_km * 1000.0
    region_px = int(2 * radius_m / mpp_index)
    region = Georef(prior_lon, prior_lat, z_index, region_px, region_px)

    print(f"{Path(args.image).name}: истина ({shot.true_lat:.5f},{shot.true_lon:.5f}), "
          f"приор сдвинут на {args.offset_km} км@{args.bearing:.0f}°, σ={sigma_m/1000:.1f}км")
    print(f"индекс[{args.encoder}]: регион {region_px}px @z{z_index} (mpp {mpp_index:.2f}), клетка {cell_px}px≈{cell_px*mpp_index:.0f}м, "
          f"~{int((region_px/(cell_px*(1-args.overlap)))**2)} клеток")

    # --- ОФЛАЙН: построение индекса (одноразово, кэшируемо на диск) ---
    t0 = time.perf_counter()
    index = TerrainIndex(_ENCODERS[args.encoder]()).build(
        basemap, region, cell_size_px=cell_px, overlap=args.overlap, rotations_deg=(0.0,)
    )
    tm[f"кодирование индекса ({len(index)} клеток)"] = time.perf_counter() - t0
    per_cell_ms = 1000.0 * tm[f"кодирование индекса ({len(index)} клеток)"] / max(len(index), 1)
    extra = ""
    if args.pca_dim > 0:
        t0 = time.perf_counter()
        index.compress(args.pca_dim, whiten=args.whiten)
        tm[f"PCA-сжатие 8448→{index._reducer.dim}"] = time.perf_counter() - t0
        extra += f", PCA→{index._reducer.dim}{'/whiten' if args.whiten else ''}"
    if args.faiss:
        index.use_faiss(kind=args.faiss_kind, ef_search=args.ef_search)
        t0 = time.perf_counter()
        index._ensure_ann()  # построить FAISS-граф сейчас, а не лениво в первом запросе
        tm[f"построение FAISS/{args.faiss_kind}"] = time.perf_counter() - t0
        extra += f", FAISS/{args.faiss_kind}"
    print(f"индекс построен: {len(index)} клеток{extra}")

    # --- ОНЛАЙН: ретривал-запрос (Этаж 1) ---
    t0 = time.perf_counter()
    rr = index.query(normalize_gray(frame), k=max(5, args.top_k), prerotate_deg=-shot.yaw_deg)
    tm["ретривал-запрос (кадр→top-K)"] = time.perf_counter() - t0
    dists = [haversine_m(shot.true_lat, shot.true_lon, c.center_lat, c.center_lon) for c in rr.cells]
    print(f"retrieval top-5 расстояний до ИСТИНЫ: {[round(d) for d in dists[:5]]} м, уникальность={rr.uniqueness:.3f}")

    # --- ОНЛАЙН: точный уровень (Этаж 2) — матчинг по top-K кандидатам ---
    t0 = time.perf_counter()
    result = localize(frame, camera, prior, basemap, index=index, matcher=LightGlueMatcher(),
                      prerotate=True, max_zoom=mz, min_ncc=0.12, min_inliers=args.min_inliers,
                      retrieval_top_k=args.top_k, ransac_threshold_px=6.0)
    localize_s = time.perf_counter() - t0
    # localize() внутри повторяет ретривал (~как выше) и гоняет точный уровень по top-K;
    # вычитая одиночный запрос, получаем чистую стоимость Этажа 2.
    tm[f"точный уровень (Этаж 2, top-{args.top_k})"] = max(0.0, localize_s - tm["ретривал-запрос (кадр→top-K)"])

    if result.is_localized:
        err = haversine_m(shot.true_lat, shot.true_lon, result.center_lat, result.center_lon)
        print(f"\n→ {result.status.value}: ОШИБКА vs ИСТИНА = {err:.1f} м "
              f"(приор был в {offset_m:.0f} м!), инлайеров={result.diagnostics.get('n_inliers')}, "
              f"эллипс={result.error_ellipse_m[0]:.1f}м")
    else:
        print(f"\n→ {result.status.value}: {result.diagnostics.get('reason')}")

    # --- разбивка таймингов: офлайн (индекс) vs онлайн (запрос+поза) ---
    offline_keys = [k for k in tm if "индекс" in k or "PCA" in k or "FAISS" in k]
    online_keys = [k for k in tm if k not in offline_keys and "загрузка" not in k]
    print("\n=== ТАЙМИНГИ ===")
    print(f"  [подготовка] {'загрузка снимка + рендер кадра':<38}: {tm['загрузка снимка + рендер кадра']:6.1f} с")
    print(f"  --- ОФЛАЙН (индекс строится один раз, кэшируется) ---")
    for k in offline_keys:
        note = f"  ({per_cell_ms:.0f} мс/клетка)" if "кодирование" in k else ""
        print(f"  {k:<48}: {tm[k]:6.1f} с{note}")
    print(f"  {'= офлайн итого':<48}: {sum(tm[k] for k in offline_keys):6.1f} с")
    print(f"  --- ОНЛАЙН (на каждый кадр, реальное время) ---")
    for k in online_keys:
        val = tm[k]
        print(f"  {k:<48}: {val*1000:6.0f} мс" if val < 1 else f"  {k:<48}: {val:6.1f} с")
    print(f"  {'= онлайн итого (запрос + поза)':<48}: {sum(tm[k] for k in online_keys):6.1f} с")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
