"""Харнесс оценки: одна команда → таблица чисел по всему набору + CSV + оверлеи.

Ради этого затевался трек ([EVAL_PLAN.md](../docs/EVAL_PLAN.md), этап D). Прогон
идёт **по манифесту** (`datasets/*.yaml`), а не по содержимому папки: какие снимки
годны и откуда у них истина — свойство данных, а не аргументов запуска.

Главное отличие от прежних скриптов — **разделение вины этажей** (урок
[JOURNAL.md](../docs/JOURNAL.md)): по одному числу «локализовано / нет» нельзя
понять, что чинить. Поэтому меряются отдельно:

* **Этаж 1 (retrieval)** — ранг клетки, ближайшей к истине, её расстояние,
  уникальность. Ранг в хвосте = дескриптор не довёл до Этажа 2.
* **Этаж 2 (поза)** — клетка доставлена, но поза не сошлась = вина матчера.
* **Гейт качества** — поза есть и место верное, но связка (инлайеры ∧ NCC ∧
  эллипс) её не пропустила. Это НЕ провал поиска, а цена порогов, и чинится она
  калибровкой, а не матчером, — поэтому вынесена в отдельного виновника.
* **Сквозное** — поза найдена / принято гейтом, доля ложных, медиана/p90, время.

Ошибка меряется при **любой найденной позе**, а не только у принятых: иначе
верная локализация, забракованная гейтом, выглядела бы как «не нашли», и чинили
бы не то. В таблице такие случаи помечаются «ГЕЙТ отверг ВЕРНУЮ».

Кейсы без истины (снимки без GPS) прогоняются наравне: ошибку по ним не измерить,
но оверлей позволяет владельцу подтвердить место глазами и заморозить его в
манифесте как ``truth: manual`` (``scripts/annotate_gt.py``).

    python scripts/eval_dataset.py --manifest datasets/test_images.yaml
    python scripts/eval_dataset.py --cases 00049,Ufa2 --radius-km 1.5

Нужны: torch + MegaLoc + LightGlue, faiss, Pillow, PyYAML, сеть (тайлы Esri).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from aero_geoloc.basemap import ESRI_WORLD_IMAGERY, TileBasemap, TileCache  # noqa: E402
from aero_geoloc.dataset import EvalCase, load_dataset  # noqa: E402
from aero_geoloc.geo import Georef, ground_mpp, haversine_m, zoom_for_mpp  # noqa: E402
from aero_geoloc.localize import localize, normalize_gray  # noqa: E402
from aero_geoloc.matcher import LightGlueMatcher  # noqa: E402
from aero_geoloc.quality import MIN_INLIERS_HARD, MIN_NCC  # noqa: E402
from aero_geoloc.retrieval import MegaLocEncoder, TerrainIndex  # noqa: E402
from aero_geoloc.types import Status  # noqa: E402
from aero_geoloc.viz import save_localization_overlay  # noqa: E402

FIELDS = [
    "case", "truth_source", "trust_yaw", "gsd_m", "footprint_m", "cells", "rotations",
    "status", "accepted", "error_m", "correct", "blame",
    "true_cell_rank", "true_cell_m", "top1_to_truth_m", "uniqueness",
    "n_inliers", "photometric_ncc", "ellipse_m",
    "found_lat", "found_lon", "offline_s", "online_s", "reason",
]


def _index_geometry(case: EvalCase, radius_m: float, cell_px_target: int, max_zoom: int):
    """Зум и нарезка индекса под КОНКРЕТНЫЙ кейс.

    Клетка обязана быть ≈ отпечатку кадра, иначе эмбеддинги несопоставимы по
    масштабу. Отпечатки в наборе гуляют от ~90 м (52 м AGL) до ~520 м (400-600 м
    AGL), поэтому зум индекса выбирается **на кейс**: такой, чтобы клетка вышла
    примерно ``cell_px_target`` пикселей. Фиксированный зум либо раздул бы число
    клеток на больших отпечатках, либо потребовал бы качать тайлы гигабайтами.
    """
    footprint_m = max(case.camera.footprint_m(case.prior.altitude_m))
    mpp_target = footprint_m / cell_px_target
    z_index = zoom_for_mpp(mpp_target, case.prior.lat, mode="coarser", max_zoom=max_zoom)
    mpp_index = ground_mpp(case.prior.lat, z_index)
    cell_px = max(32, round(footprint_m / mpp_index))
    region_px = int(2 * radius_m / mpp_index)
    region = Georef(case.prior.lon, case.prior.lat, z_index, region_px, region_px)
    return region, cell_px, mpp_index, footprint_m


def _build_or_load_index(case, region, cell_px, rotations, encoder, basemap, args):
    """Офлайн-карта: с диска, если есть, иначе построить и сохранить.

    Ключ — **геометрия**, а не имя кейса: снимки одной серии (Саратов ×4,
    Волгоград ×4) делят приор и нарезку, и по имени карта пересобиралась бы на
    каждый кадр заново.
    """
    tag = (f"{region.center_lat:.4f}_{region.center_lon:.4f}_z{region.zoom}"
           f"_r{int(args.radius_km * 1000)}_c{cell_px}_rot{len(rotations)}")
    path = Path(args.maps_dir) / f"eval_{tag}.npz"
    if path.exists() and not args.rebuild:
        index = TerrainIndex.load(path, encoder)
        index.use_faiss(kind="hnsw", ef_search=args.ef_search)
        return index, 0.0, path

    t0 = time.perf_counter()
    index = TerrainIndex(encoder).build(
        basemap, region, cell_size_px=cell_px, overlap=args.overlap, rotations_deg=rotations
    )
    index.compress(args.pca_dim, whiten=False)
    offline_s = time.perf_counter() - t0
    Path(args.maps_dir).mkdir(parents=True, exist_ok=True)
    index.save(path)
    index.use_faiss(kind="hnsw", ef_search=args.ef_search)
    return index, offline_s, path


def _transform_into(result, camera, georef) -> np.ndarray | None:
    """Преобразование «кадр → МОЁ окно подложки» из отпечатка результата.

    ``localize`` считает позу относительно своего внутреннего окна, которое наружу
    не отдаётся, — а оверлей рисуется против окна, которое тянет уже харнесс.
    Восстанавливаем точно: углы отпечатка известны в координатах, переводим их в
    пиксели своего окна и решаем задачу подобия по четырём соответствиям.
    """
    if not result.footprint_lonlat:
        return None
    src = np.array(
        [[0.0, 0.0], [camera.image_width - 1.0, 0.0],
         [camera.image_width - 1.0, camera.image_height - 1.0], [0.0, camera.image_height - 1.0]],
        dtype=np.float32,
    )
    dst = []
    for lon, lat in result.footprint_lonlat:
        px, py = georef.lonlat_to_pixel(lon, lat)
        dst.append([float(px), float(py)])
    matrix, _ = cv2.estimateAffinePartial2D(src, np.array(dst, np.float32), method=cv2.LMEDS)
    return matrix


def _retrieval_diagnostics(index, frame, case, prerotate_deg):
    """Ранг клетки, ближайшей к истине, + top-1 и уникальность (вина Этажа 1)."""
    result = index.query(normalize_gray(frame), k=len(index), prerotate_deg=prerotate_deg)
    if not result.cells:
        return None, None, None, 0.0
    if not case.has_truth:
        return None, None, None, result.uniqueness
    distances = [
        haversine_m(case.truth_lat, case.truth_lon, c.center_lat, c.center_lon)
        for c in result.cells
    ]
    nearest = min(range(len(distances)), key=lambda i: distances[i])
    return nearest + 1, distances[nearest], distances[0], result.uniqueness


def _failed_gate(diag: dict, args) -> str:
    """Какое условие связки качества не пропустило позу (см. quality.assess)."""
    reasons = []
    inliers = diag.get("n_inliers")
    ncc = diag.get("photometric_ncc")
    semi_major = diag.get("semi_major_m")
    if inliers is not None and inliers < MIN_INLIERS_HARD:
        reasons.append(f"инлайеры {inliers}<{MIN_INLIERS_HARD}")
    if ncc is not None and float(ncc) < MIN_NCC:
        reasons.append(f"NCC {float(ncc):.3f}<{MIN_NCC}")
    if semi_major is not None and float(semi_major) > args.max_ellipse_m:
        reasons.append(f"эллипс {float(semi_major):.2f}>{args.max_ellipse_m}")
    return ", ".join(reasons) if reasons else "связка качества"


def _blame(result, case, rank, error_m, accepted, pose_found, args) -> str:
    """Кому предъявлять претензию: Этажу 1, Этажу 2 или гейту качества."""
    diag = result.diagnostics or {}
    if not pose_found:
        if rank is not None and rank > args.top_k:
            return f"Этаж 1 (верная клетка на {rank}, top-K={args.top_k})"
        if rank is not None:
            return "Этаж 2 (клетка доставлена, поза не сошлась)"
        # Истины нет, ранг не посчитать — но причина отказа всё равно называет
        # виновного: если точный уровень перебрал кандидатов, Этаж 1 отработал.
        reason = str(diag.get("reason", ""))
        if "точный уровень не сошёлся" in reason:
            return "Этаж 2 (кандидаты были, поза не сошлась)"
        if "retrieval" in reason or "уникальност" in reason:
            return f"Этаж 1 ({reason})"
        return f"поза не найдена ({reason or 'причина не записана'})"

    right_place = error_m is not None and error_m <= args.correct_m
    if accepted:
        if error_m is None:
            return "принято, истины нет — проверить оверлей"
        return "ok" if right_place else "ЛОЖНОЕ (принято, но не то место)"
    # Поза есть, но связка качества её не пропустила.
    gate = _failed_gate(diag, args)
    if error_m is None:
        return f"поза есть, гейт: {gate} — проверить оверлей"
    return (f"ГЕЙТ отверг ВЕРНУЮ ({gate})" if right_place
            else f"гейт отверг неверную — верно ({gate})")


def evaluate_case(case: EvalCase, args, encoder, basemap, max_zoom) -> dict:
    row = {f: "" for f in FIELDS}
    row.update(case=case.name, truth_source=case.truth_source, trust_yaw=int(case.trust_yaw))

    z_fine = case.basemap_zoom(max_zoom=max_zoom)
    frame, camera = case.frame_at_mpp(ground_mpp(case.prior.lat, z_fine))
    radius_m = args.radius_km * 1000.0
    region, cell_px, mpp_index, footprint_m = _index_geometry(
        case, radius_m, args.cell_px, max_zoom
    )
    # Курс неизвестен → предповорот невозможен, и индекс приходится аугментировать
    # повёрнутыми копиями клеток (EVAL_PLAN, Б3). Это плата за незнание курса.
    rotations = (
        tuple(float(d) for d in range(0, 360, args.rotation_step))
        if not case.trust_yaw else (0.0,)
    )
    row.update(gsd_m=round(case.gsd_m, 4), footprint_m=round(footprint_m), rotations=len(rotations))

    index, offline_s, _ = _build_or_load_index(
        case, region, cell_px, rotations, encoder, basemap, args
    )
    row.update(cells=len(index), offline_s=round(offline_s, 1))

    prerotate_deg = -case.prior.yaw_deg if case.trust_yaw else 0.0
    rank, cell_m, top1_m, uniqueness = _retrieval_diagnostics(index, frame, case, prerotate_deg)
    row.update(
        true_cell_rank=rank if rank is not None else "",
        true_cell_m=round(cell_m, 1) if cell_m is not None else "",
        top1_to_truth_m=round(top1_m, 1) if top1_m is not None else "",
        uniqueness=round(uniqueness, 4),
    )

    t0 = time.perf_counter()
    result = localize(
        frame, camera, case.prior, basemap, index=index, matcher=LightGlueMatcher(),
        # prerotate=True всегда: флаг значит «матчер не инвариантен к повороту»
        # (LightGlue такой), а не «курс известен». При trust_yaw=False угол берётся
        # у совпавшей клетки индекса — без этого аугментация чинит только Этаж 1.
        trust_yaw=case.trust_yaw, prerotate=True, max_zoom=max_zoom,
        min_inliers=args.min_inliers, retrieval_top_k=args.top_k, ransac_threshold_px=6.0,
    )
    row["online_s"] = round(time.perf_counter() - t0, 1)

    diag = result.diagnostics or {}
    row.update(
        status=result.status.value,
        n_inliers=diag.get("n_inliers", ""),
        photometric_ncc=(round(float(diag["photometric_ncc"]), 4)
                         if diag.get("photometric_ncc") is not None else ""),
        ellipse_m=(round(result.error_ellipse_m[0], 3) if result.error_ellipse_m else ""),
        found_lat=round(result.center_lat, 6) if result.center_lat is not None else "",
        found_lon=round(result.center_lon, 6) if result.center_lon is not None else "",
        reason=diag.get("reason", ""),
    )

    # Разводим два разных исхода, которые легко спутать:
    #   поза НАЙДЕНА (LOCALIZED либо LOW_CONFIDENCE) — координаты есть;
    #   поза ПРИНЯТА (только LOCALIZED) — связка качества её пропустила.
    # Ошибку меряем при любой найденной позе: иначе не видно, что пайплайн нашёл
    # верное место, а забраковал его гейт, — а это разные починки.
    pose_found = result.is_localized
    accepted = result.status is Status.LOCALIZED
    row["accepted"] = int(accepted)
    error_m = None
    if pose_found and case.has_truth:
        error_m = haversine_m(case.truth_lat, case.truth_lon, result.center_lat, result.center_lon)
        row["error_m"] = round(error_m, 1)
        row["correct"] = int(error_m <= args.correct_m)

    row["blame"] = _blame(result, case, rank, error_m, accepted, pose_found, args)

    if not args.no_overlay:
        centre = (result.center_lat, result.center_lon) if result.center_lat is not None else (
            case.prior.lat, case.prior.lon
        )
        mpp_fine = ground_mpp(centre[0], z_fine)
        window = int(2.2 * footprint_m / mpp_fine)
        ref, gref = basemap(centre[1], centre[0], z_fine, window, window)
        matrix = _transform_into(result, camera, gref)
        shown = dataclasses.replace(result, transform=matrix) if matrix is not None else result
        save_localization_overlay(
            Path(args.out_dir) / f"{case.name}.png", frame, ref, gref, shown,
            title=f"{case.name} ({case.truth_source})",
            truth_lat=case.truth_lat, truth_lon=case.truth_lon,
        )
    return row


def report(rows: list[dict], excluded, args) -> None:
    print(f"\n{'='*104}\nОЦЕНКА: {len(rows)} кейсов (порог верности {args.correct_m:.0f} м)\n{'='*104}")
    head = f"{'кейс':<14}{'статус':<16}{'ошибка':>9}  {'ранг':>6}{'инл':>5}{'NCC':>7}  {'вина / примечание':<34}{'онлайн':>8}"
    print(head)
    print("-" * 104)
    for r in rows:
        err = f"{r['error_m']} м" if r["error_m"] != "" else "—"
        rank = r["true_cell_rank"] if r["true_cell_rank"] != "" else "—"
        ncc = r["photometric_ncc"] if r["photometric_ncc"] != "" else "—"
        print(f"{r['case']:<14}{r['status']:<16}{err:>9}  {str(rank):>6}{str(r['n_inliers']):>5}"
              f"{str(ncc):>7}  {r['blame']:<34}{r['online_s']:>7}с")

    pose_rows = [r for r in rows if r["status"] in ("localized", "low_confidence")]
    accepted = [r for r in rows if r["status"] == "localized"]
    scored = [r for r in pose_rows if r["error_m"] != ""]
    correct = [r for r in scored if r["correct"]]
    false_pos = [r for r in scored if r["accepted"] == 1 and not r["correct"]]
    gate_missed = [r for r in scored if r["accepted"] == 0 and r["correct"]]
    errors = [float(r["error_m"]) for r in correct]
    print("-" * 104)
    print(f"поза найдена: {len(pose_rows)}/{len(rows)}   принято гейтом (LOCALIZED): "
          f"{len(accepted)}/{len(rows)}   с истиной: {len(scored)}")
    print(f"верных мест: {len(correct)}/{len(scored)}   **ЛОЖНЫХ ПРИНЯТО: {len(false_pos)}**   "
          f"верных отвергнуто гейтом: {len(gate_missed)}")
    if errors:
        p90 = statistics.quantiles(errors, n=10)[-1] if len(errors) > 1 else errors[0]
        print(f"ошибка верных: медиана {statistics.median(errors):.1f} м, p90 {p90:.1f} м, "
              f"макс {max(errors):.1f} м")
    if gate_missed:
        print("  ВНИМАНИЕ: гейт качества отверг верные локализации — "
              + ", ".join(f"{r['case']}({r['error_m']} м)" for r in gate_missed))
    online = [float(r["online_s"]) for r in rows if r["online_s"] != ""]
    offline = [float(r["offline_s"]) for r in rows if r["offline_s"] not in ("", 0.0)]
    if online:
        print(f"время: онлайн медиана {statistics.median(online):.1f} с"
              + (f", офлайн (сборка карт) всего {sum(offline):.0f} с" if offline else
                 ", карты взяты с диска"))
    blame1 = [r for r in rows if r["blame"].startswith("Этаж 1")]
    blame2 = [r for r in rows if r["blame"].startswith("Этаж 2")]
    print(f"вина: Этаж 1 — {len(blame1)}, Этаж 2 — {len(blame2)}")
    if excluded:
        print(f"\nисключено манифестом ({len(excluded)}):")
        for e in excluded:
            print(f"  {e.name:<14} {e.reason[:80]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default="datasets/test_images.yaml")
    parser.add_argument("--cases", default="", help="через запятую: прогнать только эти кейсы")
    parser.add_argument("--prior", default="",
                        help="переопределить приор всех кейсов: 'lat,lon' — когда опорная "
                             "точка манифеста промахивается мимо места съёмки")
    parser.add_argument("--sigma-m", type=float, default=0.0,
                        help="переопределить σ приора, м (0 = как в манифесте)")
    parser.add_argument("--radius-km", type=float, default=2.0, help="радиус региона индексации")
    parser.add_argument("--cell-px", type=int, default=350, help="целевой размер клетки индекса, px")
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--pca-dim", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--min-inliers", type=int, default=6)
    parser.add_argument("--rotation-step", type=int, default=45,
                        help="шаг ротационной аугментации для кейсов без курса, °")
    parser.add_argument("--ef-search", type=int, default=128)
    parser.add_argument("--correct-m", type=float, default=50.0, help="порог «верно», м")
    parser.add_argument("--max-ellipse-m", type=float, default=3.0,
                        help="порог эллипса в связке качества (для расшифровки вины)")
    parser.add_argument("--maps-dir", default="maps")
    parser.add_argument("--out-dir", default="eval_out")
    parser.add_argument("--cache", default="tiles")
    parser.add_argument("--rebuild", action="store_true", help="пересобрать карты")
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument("--out", default="", help="CSV с сырьём (по умолчанию out-dir/eval.csv)")
    args = parser.parse_args()

    dataset = load_dataset(args.manifest)
    cases = dataset.cases
    if args.prior or args.sigma_m > 0:
        lat, lon = ((float(v) for v in args.prior.split(",")) if args.prior
                    else (None, None))
        patched = []
        for c in cases:
            prior = dataclasses.replace(
                c.prior,
                **({"lat": lat, "lon": lon} if args.prior else {}),
                **({"sigma_m": args.sigma_m} if args.sigma_m > 0 else {}),
            )
            patched.append(dataclasses.replace(c, prior=prior))
        cases = patched
        print(f"приор переопределён: {args.prior or 'центр как в манифесте'}"
              f"{f', σ={args.sigma_m:.0f} м' if args.sigma_m > 0 else ''}")
    if args.cases:
        wanted = {c.strip() for c in args.cases.split(",") if c.strip()}
        cases = [c for c in cases if c.name in wanted]
        if not cases:
            print(f"нет таких кейсов: {sorted(wanted)}")
            return 1

    max_zoom = ESRI_WORLD_IMAGERY.max_zoom
    basemap = TileBasemap(cache=TileCache(args.cache))
    encoder = MegaLocEncoder()
    print(f"набор {dataset.name}: {len(cases)} кейсов, радиус {args.radius_km} км, "
          f"top-K {args.top_k}, порог инлайеров {args.min_inliers}")

    rows: list[dict] = []
    for case in cases:
        print(f"\n[{case.name}] GSD={case.gsd_m:.4f} курс={'да' if case.trust_yaw else 'НЕТ'} "
              f"истина={case.truth_source}", flush=True)
        try:
            row = evaluate_case(case, args, encoder, basemap, max_zoom)
        except Exception as exc:  # noqa: BLE001 — один кейс не должен рушить прогон
            print(f"  ОШИБКА: {type(exc).__name__}: {exc}", flush=True)
            row = {f: "" for f in FIELDS}
            row.update(case=case.name, status="ошибка", blame=f"{type(exc).__name__}",
                       reason=str(exc)[:120])
        rows.append(row)
        print(f"  → {row['status']}  ошибка={row['error_m'] or '—'}  {row['blame']}", flush=True)

    out_csv = Path(args.out) if args.out else Path(args.out_dir) / "eval.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    report(rows, dataset.excluded, args)
    print(f"\nсырьё → {out_csv}")
    if not args.no_overlay:
        print(f"оверлеи → {Path(args.out_dir).resolve()}  (панели 1-2: то ли это место; "
              f"3: точность совмещения; 4: где в районе)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
