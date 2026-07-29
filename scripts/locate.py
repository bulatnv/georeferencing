"""Локализация снимка на карте: одна команда → отчёт, который можно открыть.

Инструмент владельца ([TOOL_PLAN.md](../docs/TOOL_PLAN.md), этап T4). На вход —
снимок и то, что о нём известно; на выходе — каталог с отчётом, оверлеем и
контуром отпечатка для Google Earth или QGIS.

    python scripts/locate.py --image DJI_0123.JPG --lat 54.81 --lon 56.09 --sigma-km 1.5
    python scripts/locate.py --image кадр.jpg --lat 51.53 --lon 46.06 --sigma-km 2 --gsd 0.065
    python scripts/locate.py --image снимок.JPG --lat 48.7 --lon 44.5 --sigma-km 1 \\
        --altitude 300 --fov 73 --yaw 120

Масштаб кадра задаётся одним из трёх способов — ``--gsd``, ``--altitude`` с
параметрами камеры, либо ``--from-exif``; правила приоритета и что делать, если
чего-то не хватает, — в :mod:`aero_geoloc.request`.

**Первая локализация в новом районе занимает минуты**: строится карта местности
(Этаж 1). Она кэшируется, и следующие снимки того же района идут за секунды.
Скрипт печатает оценку времени до начала сборки, а не после.

Нужны: torch с матчером и энкодером, сеть для тайлов подложки (кэшируются).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from aero_geoloc.basemap import (  # noqa: E402
    ESRI_WORLD_IMAGERY,
    TileBasemap,
    TileCache,
    deepest_imagery_zoom,
    probe_imagery,
)
from aero_geoloc.geo import ground_mpp  # noqa: E402
from aero_geoloc.localize import localize  # noqa: E402
from aero_geoloc.matcher import create_matcher  # noqa: E402
from aero_geoloc.region import (  # noqa: E402
    build_or_load,
    estimated_build_seconds,
    human_time,
    plan_region,
)
from aero_geoloc.report import save_report, save_summary  # noqa: E402
from aero_geoloc.request import InputError, build_request  # noqa: E402
from aero_geoloc.retrieval import MegaLocEncoder  # noqa: E402
from aero_geoloc.types import LocalizationResult, Status  # noqa: E402
from aero_geoloc.viz import render_localization  # noqa: E402

#: Ядро по умолчанию — измеренно лучшее на наборе (`docs/RESEARCH_A_RESULTS.md`):
#: 16/16 поз, 0 ложных, берёт кросс-сезон. `--fast` даёт втрое более быстрое.
DEFAULT_MATCHER = "minima_roma"
FAST_MATCHER = "lightglue"

#: Сигнал качества в связке. Библиотечный дефолт — `ncc` (пакет не должен тянуть
#: torch ради оценки качества), но боевая конфигурация — `dino`: он измеренно
#: ровнее по шкале между кадрами и перестал резать верные локализации
#: (веха E3 в `docs/JOURNAL.md`). Инструмент обязан идти на боевой, иначе владелец
#: получит не ту конфигурацию, на которой заморожено золото.
DEFAULT_PHOTOMETRIC = "dino"

#: Ниже этого уровня спускаться незачем: на z14 подложка грубее 2.5 м/пиксель, и
#: сопоставлять кадр уже не с чем. Дойти сюда — значит, что съёмки в районе нет
#: вовсе, и отказать честнее, чем продолжать.
MIN_IMAGERY_ZOOM = 14


def _say(step: str, text: str) -> None:
    print(f"[{step}] {text}", flush=True)


def imagery_zoom(basemap, request, args, z_wanted: int) -> tuple[int | None, dict]:
    """Уровень, на котором в районе есть настоящая съёмка, и что про это известно.

    Зачем это вообще ([JOURNAL.md](../docs/JOURNAL.md), веха про пустую подложку).
    ``max_zoom`` провайдера — предел пирамиды, а не гарантия покрытия: вне городов
    Esri на глубоких уровнях отдаёт заглушку вместо снимка. Раньше пайплайн этого
    не замечал и честно сопоставлял кадр с чистым листом, выдавая позу без смысла.

    Проба стоит девять тайлов. Если на желаемом уровне съёмка есть — а так на всех
    районах набора, — не меняется ровно ничего: ни зум, ни потолок. Спуск включается
    только там, где иначе работа шла бы по пустоте.

    Returns:
        ``(zoom, диагностика)``; ``zoom`` равен ``None``, когда съёмки нет и на
        :data:`MIN_IMAGERY_ZOOM` — локализовать в этом районе нечем.
    """
    radius_m = args.radius_km * 1000.0
    probe = probe_imagery(request.prior.lon, request.prior.lat, z_wanted,
                          radius_m=radius_m, provider=basemap.provider,
                          cache=basemap.cache, allow_network=basemap.allow_network)
    if probe.has_imagery:
        return z_wanted, {"imagery_zoom": z_wanted, "imagery_probe": probe.describe()}

    _say("  !", probe.describe() + " — на этом уровне подложки нет съёмки")
    deeper, probes = deepest_imagery_zoom(
        request.prior.lon, request.prior.lat, radius_m=radius_m,
        max_zoom=z_wanted - 1, min_zoom=MIN_IMAGERY_ZOOM, provider=basemap.provider,
        cache=basemap.cache, allow_network=basemap.allow_network)
    diagnostics = {
        "imagery_zoom": deeper,
        "imagery_zoom_wanted": z_wanted,
        "imagery_probe": probe.describe(),
        "imagery_probes": [p.describe() for p in [probe, *probes]],
    }
    if deeper is None:
        return None, diagnostics
    _say("  ✓", f"спускаюсь на zoom {deeper}: {probes[-1].describe()}. "
                f"Подложка будет грубее кадра — точность от этого пострадает")
    return deeper, diagnostics


def _overlay_window(basemap, result, request, plan, z_fine, camera):
    """Окно подложки для оверлея + преобразование кадра в его пиксели.

    Окно берётся вокруг найденного центра (а при отказе — вокруг приора) и шире
    отпечатка, чтобы на панели контекста было видно окружение, а не только сам
    кадр впритык.
    """
    centre = ((result.center_lat, result.center_lon) if result.center_lat is not None
              else (request.prior.lat, request.prior.lon))
    mpp = ground_mpp(centre[0], z_fine)
    size = max(256, int(2.2 * plan.footprint_m / mpp))
    ref, georef = basemap(centre[1], centre[0], z_fine, size, size)
    if not result.footprint_lonlat:
        return ref, georef, None
    src = np.array([[0.0, 0.0], [camera.image_width - 1.0, 0.0],
                    [camera.image_width - 1.0, camera.image_height - 1.0],
                    [0.0, camera.image_height - 1.0]], dtype=np.float32)
    dst = np.array([georef.lonlat_to_pixel(lon, lat)
                    for lon, lat in result.footprint_lonlat], dtype=np.float32)
    matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    return ref, georef, matrix


def _ellipse_words(result) -> str:
    """«эллипс 0.0 м» читается как «ошибки нет», хотя это округление до нуля."""
    if not result.error_ellipse_m:
        return "—"
    major = result.error_ellipse_m[0]
    return "<0.1 м" if major < 0.05 else f"{major:.2f} м"


def summary_row(name: str, result, timings: dict[str, float]) -> dict:
    """Строка снимка для сводки и для консоли.

    Вынесена отдельно намеренно: это **контракт** между :func:`locate_one` и
    :func:`aero_geoloc.report.save_summary`, и он должен проверяться тестом без
    torch. Однажды он уже разъехался — ``locate_one`` возвращал кортеж вместо
    словаря, сводка по пачке падала на каждом снимке, а тесты этого не видели,
    потому что проверяли ``save_summary`` в отрыве от её единственного источника.
    """
    diag = result.diagnostics or {}
    ellipse = _ellipse_words(result)
    if result.center_lat is not None:
        line = (f"{result.status.value}  {result.center_lat:.6f} {result.center_lon:.6f}"
                f"  курс {result.heading_deg:.0f}°"
                + (f", эллипс {ellipse}" if ellipse != "—" else ""))
    else:
        line = f"{result.status.value}  {diag.get('reason', '')}"
    return {
        "name": name, "status": result.status.value,
        "lat": result.center_lat, "lon": result.center_lon,
        "ellipse": ellipse, "inliers": diag.get("n_inliers", "—"),
        "reason": diag.get("reason", ""),
        "seconds": round(sum(timings.values()), 1),
        "line": line,
    }


def locate_one(path: Path, args, basemap, encoder, matcher, max_zoom) -> dict:
    """Локализовать один снимок и записать отчёт. Возвращает строку для сводки."""
    timings: dict[str, float] = {}

    request = build_request(
        path, lat=args.lat, lon=args.lon, sigma_m=args.sigma_m, gsd_m=args.gsd,
        altitude_m=args.altitude, fov_deg=args.fov, focal_mm=args.focal_mm,
        sensor_width_mm=args.sensor_mm, yaw_deg=args.yaw, from_exif=args.from_exif,
    )
    _say("1/4", "Вход: " + request.describe())
    for note in request.notes:
        _say("  !", note)

    started = time.perf_counter()
    z_wanted = request.basemap_zoom(max_zoom=max_zoom)
    z_fine, imagery = imagery_zoom(basemap, request, args, z_wanted)
    timings["проверка подложки"] = time.perf_counter() - started
    if z_fine is None:
        # Съёмки нет во всём диапазоне — отказ ДО сборки карты района. Строить её
        # по заглушкам значило бы потратить минуты и выдать позу без смысла.
        result = LocalizationResult.failed(
            f"у подложки нет съёмки в этом районе (проверено с zoom {z_wanted} "
            f"до {MIN_IMAGERY_ZOOM})", **imagery)
        out_dir = Path(args.out) / path.stem
        report = save_report(out_dir, request, result, matcher=args.matcher, timings=timings)
        row = summary_row(path.stem, result, timings)
        _say("4/4", f"Отчёт → {report}")
        print(f"      {row['line']}\n", flush=True)
        return row
    if z_fine < z_wanted:
        # Потолок опускаем вместе с уровнем: иначе цикл масштаба внутри localize
        # снова заберётся туда, где заглушки.
        max_zoom = z_fine

    started = time.perf_counter()
    frame, camera = request.frame_at_mpp(ground_mpp(request.prior.lat, z_fine))
    timings["подготовка кадра"] = time.perf_counter() - started

    plan = plan_region(
        request.camera, request.prior,
        radius_m=args.radius_km * 1000.0, max_zoom=max_zoom, fine_zoom=z_fine,
        trust_yaw=request.trust_yaw, cache_dir=args.maps_dir, prefix="tool",
    )
    index, offline_s = None, 0.0
    if args.no_index:
        # Окно вместо карты района: грубый уровень ищет прямо вокруг приора. Имеет
        # смысл только когда приор уже точен — иначе окно раздувается, а матчер на
        # большом окне разваливается (см. веху про окно в JOURNAL).
        _say("2/4", f"Карта района не строится (--no-index): грубый поиск идёт окном "
                    f"вокруг приора ±{request.prior.sigma_m:.0f} м")
        _say("  !", "это легаси-путь со стенда, на реальных кадрах он не подтверждён; "
                    "при отказе повторите без --no-index")
        if request.prior.sigma_m > 2.0 * plan.footprint_m:
            _say("  !", f"приор ±{request.prior.sigma_m:.0f} м намного шире отпечатка "
                        f"{plan.footprint_m:.0f} м — без карты района шансы малы")
    else:
        _say("2/4", "Карта района: " + plan.describe())
        if not plan.cached:
            _say("  …", f"сборка займёт примерно {human_time(estimated_build_seconds(plan))} "
                        f"плюс загрузка тайлов, если район новый; дальше он берётся из кэша")
        index, offline_s = build_or_load(plan, basemap, encoder)
        if offline_s:
            _say("  ✓", f"карта построена за {human_time(offline_s)} → {plan.path.name}")
    timings["карта района"] = offline_s

    _say("3/4", f"Локализация: ядро {args.matcher}, top-K {args.top_k}")
    started = time.perf_counter()
    result = localize(
        frame, camera, request.prior, basemap, index=index, matcher=matcher,
        trust_yaw=request.trust_yaw, prerotate=True, max_zoom=max_zoom,
        min_inliers=args.min_inliers, retrieval_top_k=args.top_k,
        ransac_threshold_px=6.0, photometric_kind=args.photometric,
    )
    timings["локализация"] = time.perf_counter() - started
    if result.diagnostics is None:
        result.diagnostics = {}
    result.diagnostics.update(imagery)   # на каком уровне подложки работали

    overlay = None
    if not args.no_overlay:
        try:
            ref, georef, matrix = _overlay_window(basemap, result, request, plan, z_fine, camera)
            shown = result if matrix is None else result.__class__(
                **{**result.__dict__, "transform": matrix})
            overlay = render_localization(frame, ref, georef, shown, title=path.name)
        except Exception as exc:  # noqa: BLE001 — картинка не должна ронять отчёт
            _say("  !", f"оверлей не собрался: {type(exc).__name__}: {exc}")

    out_dir = Path(args.out) / path.stem
    report = save_report(out_dir, request, result, overlay=overlay,
                         matcher=args.matcher, timings=timings,
                         region=None if args.no_index else plan.path.name)
    row = summary_row(path.stem, result, timings)
    _say("4/4", f"Отчёт → {report}")
    print(f"      {row['line']}\n", flush=True)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Масштаб кадра: --gsd ЛИБО --altitude с --fov/--focal-mm ЛИБО --from-exif.",
    )
    parser.add_argument("--image", required=True, help="снимок или папка со снимками")
    parser.add_argument("--lat", type=float, help="широта приблизительного центра")
    parser.add_argument("--lon", type=float, help="долгота приблизительного центра")
    parser.add_argument("--sigma-km", type=float, help="погрешность приора, км")
    parser.add_argument("--sigma-m", type=float, help="погрешность приора, м")
    parser.add_argument("--gsd", type=float, help="разрешение кадра на земле, м/пиксель")
    parser.add_argument("--altitude", type=float, help="высота съёмки над землёй, м")
    parser.add_argument("--fov", type=float, help="горизонтальный угол обзора, °")
    parser.add_argument("--focal-mm", type=float, help="фокусное расстояние, мм")
    parser.add_argument("--sensor-mm", type=float, help="ширина сенсора, мм")
    parser.add_argument("--yaw", type=float, help="курс кадра относительно севера, °")
    parser.add_argument("--from-exif", action="store_true",
                        help="взять приор из GPS снимка. ВНИМАНИЕ: тогда система ищет "
                             "кадр там, куда ей же указали, и успех ничего не доказывает")
    parser.add_argument("--radius-km", type=float, default=0.0,
                        help="радиус карты района; 0 = по погрешности приора")
    parser.add_argument("--max-zoom", type=int, default=0,
                        help="потолок детальности подложки; 0 = предел провайдера. "
                             "Инструмент и сам спускается там, где съёмки нет, — "
                             "флаг нужен, чтобы задать уровень вручную")
    parser.add_argument("--matcher", default=DEFAULT_MATCHER)
    parser.add_argument("--fast", action="store_true",
                        help=f"быстрое ядро {FAST_MATCHER}: втрое быстрее, чуть меньше находит")
    parser.add_argument("--photometric", default=DEFAULT_PHOTOMETRIC,
                        help="сигнал качества в связке: dino (боевой) или ncc (без torch)")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--min-inliers", type=int, default=6)
    parser.add_argument("--out", default="out", help="каталог отчётов")
    parser.add_argument("--maps-dir", default="maps", help="кэш карт районов")
    parser.add_argument("--cache", default="tiles", help="кэш тайлов подложки")
    parser.add_argument("--no-index", action="store_true",
                        help="ЛЕГАСИ: без карты района, грубый поиск окном вокруг "
                             "приора. Путь остался со стенда и на реальных кадрах "
                             "НЕ подтверждён — проверено на Volgograd3, отказ при "
                             "любом ядре и любом приоре. Обычный путь — с картой")
    parser.add_argument("--no-overlay", action="store_true")
    args = parser.parse_args()

    if args.fast:
        args.matcher = FAST_MATCHER
    if args.sigma_m is None:
        args.sigma_m = args.sigma_km * 1000.0 if args.sigma_km else None
    # Радиус карты по умолчанию идёт от приора: район обязан накрывать диск ±3σ,
    # иначе верной клетки в нём просто нет. Меньше — только если владелец сам знает.
    if args.radius_km <= 0 and args.sigma_m:
        args.radius_km = max(0.5, min(5.0, args.sigma_m * 1.5 / 1000.0))

    path = Path(args.image)
    images = ([p for p in sorted(path.iterdir())
               if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")]
              if path.is_dir() else [path])
    if not images:
        print(f"в {path} нет изображений")
        return 1

    basemap = TileBasemap(cache=TileCache(args.cache))
    max_zoom = args.max_zoom or ESRI_WORLD_IMAGERY.max_zoom
    encoder = MegaLocEncoder()
    matcher = create_matcher(args.matcher)

    failures = 0
    rows: list[dict] = []
    for image_path in images:
        try:
            rows.append(locate_one(image_path, args, basemap, encoder, matcher, max_zoom))
        except InputError as exc:
            print(f"\nНе хватает данных для {image_path.name}:\n  {exc}\n")
            failures += 1
        except Exception as exc:  # noqa: BLE001 — один снимок не должен ронять пачку
            print(f"\nОШИБКА на {image_path.name}: {type(exc).__name__}: {exc}\n")
            failures += 1
    if len(images) > 1 and rows:
        print(f"Сводка → {save_summary(args.out, rows)}", flush=True)
    return 1 if failures == len(images) else 0


if __name__ == "__main__":
    raise SystemExit(main())
