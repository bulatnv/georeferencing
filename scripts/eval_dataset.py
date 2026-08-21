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
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from aero_geoloc.basemap import ESRI_WORLD_IMAGERY, TileBasemap, TileCache  # noqa: E402
from aero_geoloc.dataset import EvalCase, load_dataset  # noqa: E402
from aero_geoloc.geo import ground_mpp, haversine_m  # noqa: E402
from aero_geoloc.localize import MAX_FINE_WINDOW_PX, localize, normalize_gray  # noqa: E402
from aero_geoloc.matcher import create_matcher  # noqa: E402
from aero_geoloc.quality import MIN_INLIERS_HARD, MIN_NCC  # noqa: E402
from aero_geoloc.region import build_or_load, plan_region  # noqa: E402
from aero_geoloc.regression import CONFIG_KEYS  # noqa: E402
from aero_geoloc.retrieval import MegaLocEncoder  # noqa: E402
from aero_geoloc.types import Status  # noqa: E402
from aero_geoloc.viz import save_localization_overlay  # noqa: E402

FIELDS = [
    "case", "matcher", "truth_source", "trust_yaw", "gsd_m", "footprint_m", "cells", "rotations", "overlap",
    "status", "accepted", "error_m", "tolerance_m", "correct", "blame",
    "true_cell_rank", "true_cell_m", "top1_to_truth_m", "uniqueness",
    "n_inliers", "photometric", "photometric_kind", "certainty_mean", "certainty_cover",
    "ellipse_m",
    "found_lat", "found_lon", "heading_deg", "offline_s", "online_s", "reason",
]


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


def _tolerance_m(case: EvalCase, footprint_m: float, args) -> float:
    """Допуск «верно» — зависит от ТОЧНОСТИ ИСТИНЫ, а не только от алгоритма.

    ``exif``: GPS борта точен до единиц метров и указывает именно центр кадра —
    годится жёсткий порог (``--correct-m``).

    ``manual``: владелец отмечает на карте **узнанный объект** (мост, кран,
    причал), а не геометрический центр кадра. Расхождение в десятки метров — это
    свойство разметки, а не промах пайплайна: на кадре с отпечатком 517 м точка,
    поставленная где угодно в центральной половине, даёт до ~130 м. Жёсткие 50 м
    здесь метили ВЕРНУЮ локализацию как ложную (измерено на Saratov2: место
    совпадает визуально, NCC 0.68, а ошибка 53 м). Поэтому допуск привязан к
    отпечатку кадра.
    """
    if case.truth_source == "manual":
        return max(args.correct_m, args.manual_tol_frac * footprint_m)
    return args.correct_m


def _failed_gate(diag: dict, args) -> str:
    """Какое условие связки качества не пропустило позу (см. quality.assess)."""
    reasons = []
    inliers = diag.get("n_inliers")
    ncc = diag.get("photometric")
    kind = diag.get("photometric_kind", "ncc")
    threshold = diag.get("photometric_threshold", MIN_NCC)
    semi_major = diag.get("semi_major_m")
    if inliers is not None and inliers < MIN_INLIERS_HARD:
        reasons.append(f"инлайеры {inliers}<{MIN_INLIERS_HARD}")
    if ncc is not None and float(ncc) < float(threshold):
        reasons.append(f"{kind} {float(ncc):.3f}<{threshold}")
    if semi_major is not None and float(semi_major) > args.max_ellipse_m:
        reasons.append(f"эллипс {float(semi_major):.2f}>{args.max_ellipse_m}")
    return ", ".join(reasons) if reasons else "связка качества"


def _blame(result, case, rank, error_m, accepted, pose_found, tolerance_m, args) -> str:
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

    # Допуск тот же, что и у поля correct, иначе таблица противоречит сводке.
    right_place = error_m is not None and error_m <= tolerance_m
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


def run_config(args) -> dict:
    """Конфигурация прогона — то, без чего его результат нельзя ни с чем сравнивать.

    Пишется рядом с CSV и замораживается вместе с золотом
    (:mod:`aero_geoloc.regression`): прогон с другим радиусом или другим матчером
    — это другой эксперимент, а по одной таблице чисел это не видно.
    """
    config = {key: getattr(args, key) for key in CONFIG_KEYS if hasattr(args, key)}
    # Часовые значения наружу не выпускаем: в замороженной конфигурации «-2.0»
    # читалось бы как порог, а это «взять калиброванный дефолт меры».
    if config.get("min_photometric", 0.0) < -1.0:
        config["min_photometric"] = None
    return config


def _cases_slug(cases_arg: str, limit: int = 60) -> str:
    """Детерминированное имя для частичного прогона из списка кейсов.

    Небезопасные для имени файла символы заменяются; длинный список сворачивается
    в «первый кейс + md5 + счёт», чтобы имя не выросло за пределы разумного и
    при этом два разных списка не столкнулись.
    """
    import hashlib
    import re as _re

    names = [n.strip() for n in cases_arg.split(",") if n.strip()]
    slug = "-".join(_re.sub(r"[^\w.-]", "_", n) for n in names)
    if len(slug) > limit:
        digest = hashlib.md5(slug.encode("utf-8")).hexdigest()[:8]
        slug = f"{names[0]}_and_{len(names) - 1}_more_{digest}"
    return slug


def output_csv_path(out: str, out_dir: str, cases_arg: str) -> Path:
    """Куда писать таблицу прогона (FIX_EVAL_ARTIFACT_LEAK, Ф1).

    Канонический ``eval.csv`` — вход других экспериментов (``--poses`` у
    оракульных проб), и частичный прогон, молча перезаписывающий его одной
    строкой, подменяет вход всем потребителям. Поэтому частичный прогон без
    явного ``--out`` пишет в производное имя ``eval_cases_<slug>.csv``.
    Молча попасть в ``eval.csv`` частичным прогоном нельзя.
    """
    if out:
        return Path(out)
    if cases_arg:
        return Path(out_dir) / f"eval_cases_{_cases_slug(cases_arg)}.csv"
    return Path(out_dir) / "eval.csv"


def region_radius_m(case: EvalCase, args) -> float:
    """Радиус района под **этот** кейс: из его приора, как это делает инструмент.

    Раньше радиус был один на весь набор (2 км по умолчанию) и от приора кейса не
    зависел. Пока все кадры снимались с полукилометра, разницы не было; с
    появлением кадров с 90 м она стала решающей: клетка ≈ отпечатку, то есть
    ~120 м, и район в 2 км режется на десять тысяч почти одинаковых клеток —
    режим, в котором Этаж 1 измеренно перестаёт находить верную (веха 2026-07-29).

    Формула та же, что в ``scripts/locate.py``: район обязан накрывать диск ±3σ,
    но раздувать его сверх нужды вредно. ``--radius-km`` остаётся ручным
    переопределением на весь набор — им пользуются свипы и замороженная
    конфигурация золота.
    """
    if args.radius_km > 0:
        return args.radius_km * 1000.0
    return max(500.0, min(5000.0, case.prior.sigma_m * 1.5))


def evaluate_case(case: EvalCase, args, encoder, basemap, max_zoom) -> dict:
    row = {f: "" for f in FIELDS}
    row.update(case=case.name, matcher=args.matcher,
               truth_source=case.truth_source, trust_yaw=int(case.trust_yaw))

    z_fine = case.basemap_zoom(max_zoom=max_zoom)
    frame, camera = case.frame_at_mpp(ground_mpp(case.prior.lat, z_fine))
    # Геометрия района и кэш карты — общие с инструментом владельца
    # (aero_geoloc/region.py). Здесь их быть не должно: два потребителя одной
    # логики разъезжаются, если каждый держит свою копию.
    plan = plan_region(
        case.camera, case.prior,
        radius_m=region_radius_m(case, args), max_zoom=max_zoom, fine_zoom=z_fine,
        trust_yaw=case.trust_yaw, rotation_step_deg=args.rotation_step,
        cell_px_target=args.cell_px,
        overlap=args.overlap if args.overlap > 0 else None,
        pca_dim=args.pca_dim, max_fine_window_px=args.max_fine_window_px,
        cache_dir=args.maps_dir, prefix="eval",
    )
    row.update(gsd_m=round(case.gsd_m, 4), footprint_m=round(plan.footprint_m),
               rotations=len(plan.rotations_deg), overlap=round(plan.overlap, 2))

    index, offline_s = build_or_load(
        plan, basemap, encoder, rebuild=args.rebuild, ef_search=args.ef_search)
    row.update(cells=len(index), offline_s=round(offline_s, 1))

    prerotate_deg = -case.prior.yaw_deg if case.trust_yaw else 0.0
    rank, cell_m, top1_m, uniqueness = _retrieval_diagnostics(index, frame, case, prerotate_deg)
    row.update(
        true_cell_rank=rank if rank is not None else "",
        true_cell_m=round(cell_m, 1) if cell_m is not None else "",
        top1_to_truth_m=round(top1_m, 1) if top1_m is not None else "",
        uniqueness=round(uniqueness, 4),
    )

    # Переопределения порогов ядра — для перепроверок (LOFTR_RECHECK Э3);
    # боевые дефолты ядер не меняются, флаги действуют только на этот прогон.
    matcher_kwargs = {"max_side": args.matcher_max_side}
    if args.coarse_thr is not None:
        matcher_kwargs["coarse_thr"] = args.coarse_thr
    if args.min_conf_pairs is not None:
        matcher_kwargs["min_conf"] = args.min_conf_pairs
    for name in ("filter_threshold", "depth_confidence", "width_confidence",
                 "detection_threshold"):
        value = getattr(args, name)
        if value is not None:
            matcher_kwargs[name] = value

    t0 = time.perf_counter()
    result = localize(
        frame, camera, case.prior, basemap, index=index,
        matcher=create_matcher(args.matcher, **matcher_kwargs),
        # prerotate=True всегда: флаг значит «матчер не инвариантен к повороту»
        # (LightGlue такой), а не «курс известен». При trust_yaw=False угол берётся
        # у совпавшей клетки индекса — без этого аугментация чинит только Этаж 1.
        trust_yaw=case.trust_yaw, prerotate=True, max_zoom=max_zoom,
        min_inliers=args.min_inliers, retrieval_top_k=args.top_k, ransac_threshold_px=6.0,
        max_fine_window_px=args.max_fine_window_px,
        photometric_kind=args.photometric,
        min_photometric=None if args.min_photometric < -1.0 else args.min_photometric,
    )
    row["online_s"] = round(time.perf_counter() - t0, 1)

    diag = result.diagnostics or {}
    row.update(
        status=result.status.value,
        n_inliers=diag.get("n_inliers", ""),
        photometric=(round(float(diag["photometric"]), 4)
                     if diag.get("photometric") is not None else ""),
        photometric_kind=diag.get("photometric_kind", ""),
        certainty_mean=(round(float(diag["certainty_mean"]), 5)
                        if diag.get("certainty_mean") is not None else ""),
        certainty_cover=(round(float(diag["certainty_cover"]), 5)
                         if diag.get("certainty_cover") is not None else ""),
        ellipse_m=(round(result.error_ellipse_m[0], 3) if result.error_ellipse_m else ""),
        found_lat=round(result.center_lat, 6) if result.center_lat is not None else "",
        found_lon=round(result.center_lon, 6) if result.center_lon is not None else "",
        # Курс нужен не отчёту, а анализу сигналов (scripts/e1_signals.py): у
        # кадров без EXIF это единственный способ построить оракульное
        # выравнивание — из позы, подтверждённой владельцем.
        heading_deg=round(result.heading_deg, 3) if result.heading_deg is not None else "",
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
    tolerance_m = _tolerance_m(case, plan.footprint_m, args)
    row["tolerance_m"] = round(tolerance_m)
    if pose_found and case.has_truth:
        error_m = haversine_m(case.truth_lat, case.truth_lon, result.center_lat, result.center_lon)
        row["error_m"] = round(error_m, 1)
        row["correct"] = int(error_m <= tolerance_m)

    row["blame"] = _blame(result, case, rank, error_m, accepted, pose_found, tolerance_m, args)

    if not args.no_overlay:
        centre = (result.center_lat, result.center_lon) if result.center_lat is not None else (
            case.prior.lat, case.prior.lon
        )
        mpp_fine = ground_mpp(centre[0], z_fine)
        window = int(2.2 * plan.footprint_m / mpp_fine)
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
        ncc = r["photometric"] if r["photometric"] != "" else "—"
        print(f"{r['case']:<14}{r['status']:<16}{err:>9}  {str(rank):>6}{str(r['n_inliers']):>5}"
              f"{str(ncc):>7}  {r['blame']:<40}{r['online_s']:>7}с")

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
    parser.add_argument("--matcher", default="minima_roma",
                        help="ядро матчинга. Дефолт `minima_roma` — измеренно лучший "
                             "на наборе (16/16 поз, 0 ложных, берёт кросс-сезон); "
                             "`lightglue` втрое быстрее и без тяжёлых зависимостей. "
                             "Матчер сменный по инварианту архитектуры — здесь это "
                             "ровно один флаг, и он же попадает в конфигурацию прогона, "
                             "чтобы регрессия не сравнила разные ядра между собой")
    parser.add_argument("--coarse-thr", type=float, default=None,
                        help="переопределить внутренний порог LoFTR (coarse_matching.thr) "
                             "на этот прогон; None = дефолт ядра")
    parser.add_argument("--min-conf-pairs", type=float, default=None,
                        help="переопределить min_conf ядра на этот прогон; None = дефолт")
    parser.add_argument("--filter-threshold", type=float, default=None,
                        help="lightglue-семейство: порог выхода назначения (на прогон)")
    parser.add_argument("--depth-confidence", type=float, default=None,
                        help="lightglue-семейство: ранняя остановка, -1 = выкл (на прогон)")
    parser.add_argument("--width-confidence", type=float, default=None,
                        help="lightglue-семейство: прунинг точек, -1 = выкл (на прогон)")
    parser.add_argument("--detection-threshold", type=float, default=None,
                        help="lightglue-семейство: порог score SuperPoint (на прогон)")
    parser.add_argument("--matcher-max-side", type=int, default=0,
                        help="рабочее разрешение матчера, px (0 = полное). Плотным "
                             "ядрам (LoFTR и далее RoMa/MINIMA) полное разрешение "
                             "подавать нельзя — обе картинки уменьшаются ОДНИМ "
                             "коэффициентом, см. ResizedMatcher")
    parser.add_argument("--photometric", default="dino",
                        help="чем мерить согласие кадра и подложки в связке качества: "
                             "dino (косинус плотных патч-токенов DINOv2 — измеренно "
                             "лучше, E1/E3 в docs/JOURNAL.md) или ncc (дёшево, без "
                             "torch, прежняя база калибровки). Здесь дефолт dino, а в "
                             "самой библиотеке остаётся ncc: пакет не должен тянуть "
                             "torch ради оценки качества, синтетический стенд ходит "
                             "на SIFT без него")
    parser.add_argument("--min-photometric", type=float, default=-2.0,
                        help="порог меры в связке; < -1 = калиброванный дефолт меры")
    parser.add_argument("--cases", default="", help="через запятую: прогнать только эти кейсы")
    parser.add_argument("--prior", default="",
                        help="переопределить приор всех кейсов: 'lat,lon' — когда опорная "
                             "точка манифеста промахивается мимо места съёмки")
    parser.add_argument("--sigma-m", type=float, default=0.0,
                        help="переопределить σ приора, м (0 = как в манифесте)")
    parser.add_argument("--offset-km", type=float, default=0.0,
                        help="ЗАГРУБИТЬ приор: сдвинуть его от истины на столько км. "
                             "GPS кадра остаётся только эталоном — так проверяется "
                             "работа из грубой области, а не из точной точки")
    parser.add_argument("--bearing", type=float, default=45.0, help="азимут сдвига приора, °")
    parser.add_argument("--radius-km", type=float, default=0.0,
                        help="радиус региона индексации на ВЕСЬ набор; 0 = вывести "
                             "из приора каждого кейса, как это делает инструмент. "
                             "Один радиус на набор с разными высотами съёмки даёт "
                             "низким кадрам заведомо переширокий район")
    parser.add_argument("--cell-px", type=int, default=350, help="целевой размер клетки индекса, px")
    parser.add_argument("--overlap", type=float, default=0.0,
                        help="перекрытие клеток индекса; 0 = АВТО по отпечатку кадра "
                             "(localize.required_cell_overlap). Крупным кадрам нужно "
                             "~0.75, мелким хватает 0.5 — фиксированные 0.75 для всех "
                             "давали вчетверо больше клеток там, где это не нужно")
    parser.add_argument("--max-fine-window-px", type=int, default=MAX_FINE_WINDOW_PX,
                        help="потолок окна точного уровня, px. Свойство ядра матчинга; "
                             "меняется ТОЛЬКО вместе с перекрытием сетки — оба берутся "
                             "из одного числа, и рассогласование уже стоило кейсов")
    parser.add_argument("--pca-dim", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=15,
                        help="сколько клеток ретривала отдавать Этажу 2. Измерено на "
                             "наборе: верная клетка лежит на ранге <= 11 у 11 кейсов из "
                             "12, поэтому прежние 25 были перерасходом — кандидаты с "
                             "12-го не пригодились ни разу. Снижение до 14 дало 27% "
                             "экономии онлайна при тех же 8/8 верных и 0 ложных; 15 "
                             "оставлено с запасом (см. docs/OPTIMIZATION_PLAN.md, O3)")
    parser.add_argument("--min-inliers", type=int, default=6)
    parser.add_argument("--rotation-step", type=int, default=45,
                        help="шаг ротационной аугментации для кейсов без курса, °")
    parser.add_argument("--ef-search", type=int, default=128)
    parser.add_argument("--correct-m", type=float, default=50.0, help="порог «верно», м")
    parser.add_argument("--manual-tol-frac", type=float, default=0.25,
                        help="допуск для ручной истины как доля отпечатка кадра "
                             "(владелец метит объект, а не центр кадра)")
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

    if args.offset_km > 0:
        # Приор уезжает от ИСТИНЫ — это и есть «загрубление»: система должна найти
        # место, зная лишь область. Без истины сдвигать не от чего.
        shifted = []
        for c in cases:
            if not c.has_truth:
                shifted.append(c)
                continue
            d = args.offset_km * 1000.0
            dn = d * math.cos(math.radians(args.bearing))
            de = d * math.sin(math.radians(args.bearing))
            lat = c.truth_lat + dn / 111320.0
            lon = c.truth_lon + de / (111320.0 * math.cos(math.radians(c.truth_lat)))
            sigma = args.sigma_m if args.sigma_m > 0 else max(d * 1.5, c.prior.sigma_m)
            shifted.append(dataclasses.replace(
                c, prior=dataclasses.replace(c.prior, lat=lat, lon=lon, sigma_m=sigma)))
        cases = shifted
        print(f"приор ЗАГРУБЛЁН: сдвинут от истины на {args.offset_km} км @ {args.bearing:.0f}°")
    if args.cases:
        wanted = {c.strip() for c in args.cases.split(",") if c.strip()}
        cases = [c for c in cases if c.name in wanted]
        if not cases:
            print(f"нет таких кейсов: {sorted(wanted)}")
            return 1

    max_zoom = ESRI_WORLD_IMAGERY.max_zoom
    basemap = TileBasemap(cache=TileCache(args.cache))
    encoder = MegaLocEncoder()
    radius_note = (f"радиус {args.radius_km} км на весь набор" if args.radius_km > 0
                   else "радиус из приора каждого кейса")
    print(f"набор {dataset.name}: {len(cases)} кейсов, {radius_note}, "
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

    out_csv = output_csv_path(args.out, args.out_dir, args.cases)
    if args.cases and not args.out:
        print(f"\nчастичный прогон ({len(rows)} кейсов) → {out_csv}; "
              f"канонический eval.csv не тронут")
    elif args.cases and out_csv.name == "eval.csv":
        print("\nВНИМАНИЕ: частичный прогон пишет в канонический eval.csv по явному "
              "--out — вход оракульных проб (--poses) будет подменён")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    # Конфигурация — рядом с таблицей и под тем же именем: таблица без неё не
    # сравнима ни с чем (scripts/regress.py читает именно этот файл).
    with open(out_csv.with_suffix(".config.json"), "w", encoding="utf-8") as fh:
        json.dump(run_config(args), fh, ensure_ascii=False, indent=2, sort_keys=True)

    report(rows, dataset.excluded, args)
    print(f"\nсырьё → {out_csv}")
    if not args.no_overlay:
        print(f"оверлеи → {Path(args.out_dir).resolve()}  (панели 1-2: то ли это место; "
              f"3: точность совмещения; 4: где в районе)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
