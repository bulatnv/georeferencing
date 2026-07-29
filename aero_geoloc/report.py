"""Отчёт по локализации: то, что владелец открывает и понимает без кода.

Зачем модуль ([TOOL_PLAN.md](../docs/TOOL_PLAN.md), этап T3). Результат пайплайна
— это не координата, а координата **плюс основания ей верить**. Отчёт обязан
показать и то, и другое так, чтобы решение «то ли это место» принимал человек,
глядя на картинки, а не доверяясь строке в консоли.

Что пишется рядом:

``report.html``    самодостаточный: картинки внутри base64, открывается двойным
                   кликом, ничего не качает
``result.json``    машиночитаемое — на нём потом вырастет API, если понадобится
``footprint.kml``  открыть в Google Earth
``footprint.geojson`` открыть в QGIS поверх своих данных

**Отказ формирует такой же полноценный отчёт.** ``NOT_LOCALIZED`` — легитимный
результат, а не сбой запуска: в отчёте будет причина, виновник этажа и
конкретный совет, что поменять во входных данных.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .request import LocateRequest
from .types import LocalizationResult, Status

__all__ = ["save_report", "save_summary", "result_payload",
           "footprint_geojson", "footprint_kml", "advice_for"]

_STATUS_RU = {
    Status.LOCALIZED: ("ЛОКАЛИЗОВАНО", "ok"),
    Status.LOW_CONFIDENCE: ("НИЗКОЕ ДОВЕРИЕ", "warn"),
    Status.NOT_LOCALIZED: ("НЕ ЛОКАЛИЗОВАНО", "bad"),
}


def advice_for(result: LocalizationResult, request: LocateRequest) -> list[str]:
    """Что владельцу поменять во входных данных. Отдельная функция — она главная.

    Отказ без объяснения бесполезен: человек не знает, дело в снимке, в приоре
    или в масштабе. Советы выводятся из **причины** отказа, а не выдаются
    списком на все случаи.
    """
    diag = result.diagnostics or {}
    reason = str(diag.get("reason", ""))
    tips: list[str] = []

    if result.status is Status.LOCALIZED:
        return tips

    if "уникальност" in reason or "самоподоб" in reason:
        tips.append(
            "Местность однородна (поля, лес, вода) — по такому кадру место определить "
            "нельзя в принципе. Помогает более узкий приор либо съёмка с большей высоты."
        )
    if "вне диска приора" in reason:
        tips.append(
            "Решение вышло за пределы приора: либо координаты приора неверны, либо "
            "погрешность занижена. Увеличьте --sigma-km."
        )
    if "точный уровень не сошёлся" in reason or "мало соответствий" in reason:
        tips.append(
            "Матчер не собрал позу ни на одном кандидате. Проверьте по строке выше, "
            "правдоподобен ли отпечаток кадра в метрах: неверный GSD — самая частая "
            "причина."
        )
        if not request.trust_yaw:
            tips.append("Задайте --yaw, если курс известен: это заметно упрощает задачу Этажу 2.")
        tips.append(
            "Если снимок сделан в другой сезон, чем подложка, это известная граница: "
            "плотное ядро берёт такие кадры не всегда."
        )
    if "retrieval не дал кандидатов" in reason:
        tips.append("Ни одна клетка карты не попала в диск приора — расширьте --radius-km.")
    if "нет съёмки в этом районе" in reason:
        tips.append(
            "Дело не в снимке и не в приоре: у картографической подложки в этой "
            "точке нет съёмки ни на одном пригодном уровне — сервер отдаёт "
            "заглушку вместо снимка. Проверьте координаты приора; если они верны, "
            "локализовать здесь нечем, пока не появится другая подложка."
        )

    if result.status is Status.LOW_CONFIDENCE:
        tips.append(
            "Поза найдена, но связка качества её не пропустила. Посмотрите панели "
            "совмещения: если место верное, порогам не хватает данных — сообщите об "
            "этом случае, он ценен для калибровки."
        )
    if not tips:
        tips.append(
            f"Причина: {reason or 'не записана'}. Посмотрите оверлей — по панели "
            f"контекста видно, куда именно смотрел алгоритм."
        )
    return tips


def result_payload(
    request: LocateRequest,
    result: LocalizationResult,
    *,
    matcher: str,
    timings: dict[str, float] | None = None,
    region: str | None = None,
) -> dict:
    """Машиночитаемый результат. На нём вырастет API, если он понадобится."""
    diag = dict(result.diagnostics or {})
    for key, value in list(diag.items()):
        if isinstance(value, (np.floating, np.integer)):
            diag[key] = value.item()
        elif isinstance(value, np.ndarray):
            diag[key] = value.tolist()
        elif isinstance(value, tuple):
            diag[key] = list(value)
    return {
        "снимок": str(request.image_path),
        "время": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "статус": result.status.value,
        "центр": (None if result.center_lat is None
                  else {"lat": round(result.center_lat, 7), "lon": round(result.center_lon, 7)}),
        "курс_град": None if result.heading_deg is None else round(result.heading_deg, 2),
        "высота_оценка_м": (None if result.altitude_est_m is None
                            else round(result.altitude_est_m, 1)),
        "эллипс_ошибки_м": (None if result.error_ellipse_m is None
                            else [round(v, 3) for v in result.error_ellipse_m]),
        "отпечаток": ([[round(lon, 7), round(lat, 7)] for lon, lat in result.footprint_lonlat]
                      if result.footprint_lonlat else None),
        "вход": {
            "gsd_м": round(request.gsd_m, 5),
            "источник_gsd": request.gsd_source,
            "приор": {"lat": request.prior.lat, "lon": request.prior.lon,
                      "сигма_м": request.prior.sigma_m},
            "источник_приора": request.prior_source,
            "курс_известен": request.trust_yaw,
            "отпечаток_кадра_м": [round(v, 1) for v in request.footprint_m],
            "замечания": list(request.notes),
        },
        "конфигурация": {"ядро": matcher, "район": region},
        "время_стадий_с": {k: round(v, 1) for k, v in (timings or {}).items()},
        "диагностика": diag,
        "советы": advice_for(result, request),
    }


def footprint_geojson(result: LocalizationResult, *, name: str) -> dict:
    """Контур отпечатка и центр — открыть в QGIS поверх своих слоёв."""
    features = []
    if result.footprint_lonlat:
        ring = [[float(lon), float(lat)] for lon, lat in result.footprint_lonlat]
        features.append({
            "type": "Feature",
            "properties": {"name": f"{name}: отпечаток", "status": result.status.value},
            "geometry": {"type": "Polygon", "coordinates": [ring + [ring[0]]]},
        })
    if result.center_lat is not None:
        features.append({
            "type": "Feature",
            "properties": {"name": f"{name}: центр", "status": result.status.value},
            "geometry": {"type": "Point",
                         "coordinates": [float(result.center_lon), float(result.center_lat)]},
        })
    return {"type": "FeatureCollection", "features": features}


def footprint_kml(result: LocalizationResult, *, name: str) -> str:
    """То же для Google Earth — там владельцу привычнее смотреть местность."""
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
        f"<name>{name}</name>",
    ]
    if result.footprint_lonlat:
        ring = list(result.footprint_lonlat) + [result.footprint_lonlat[0]]
        coords = " ".join(f"{lon},{lat},0" for lon, lat in ring)
        parts.append(
            f"<Placemark><name>отпечаток кадра</name><Style><LineStyle>"
            f"<color>ff00ffff</color><width>3</width></LineStyle>"
            f"<PolyStyle><fill>0</fill></PolyStyle></Style>"
            f"<Polygon><outerBoundaryIs><LinearRing><coordinates>{coords}"
            f"</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>"
        )
    if result.center_lat is not None:
        parts.append(
            f"<Placemark><name>центр кадра</name><Point><coordinates>"
            f"{result.center_lon},{result.center_lat},0</coordinates></Point></Placemark>"
        )
    parts.append("</Document></kml>")
    return "\n".join(parts)


_CSS = """
:root { color-scheme: light dark; }
body { font: 15px/1.5 system-ui, sans-serif; margin: 0 auto; max-width: 1100px;
       padding: 24px; }
h1 { font-size: 22px; margin: 0 0 4px; }
.sub { opacity: .7; font-size: 13px; margin-bottom: 20px; }
.verdict { padding: 16px 20px; border-radius: 10px; margin: 0 0 20px;
           border: 2px solid; }
.verdict.ok   { border-color: #2e7d32; background: #2e7d3216; }
.verdict.warn { border-color: #ef6c00; background: #ef6c0016; }
.verdict.bad  { border-color: #c62828; background: #c6282816; }
.verdict .big { font-size: 26px; font-weight: 700; letter-spacing: .5px; }
.coord { font: 20px ui-monospace, monospace; margin: 6px 0; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 22px; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #8884; }
th { width: 34%; font-weight: 600; opacity: .8; }
img { max-width: 100%; border-radius: 8px; border: 1px solid #8884; }
.note { border-left: 3px solid #ef6c00; padding: 6px 12px; margin: 6px 0;
        background: #ef6c000d; font-size: 14px; }
.tip  { border-left: 3px solid #1565c0; padding: 6px 12px; margin: 6px 0;
        background: #1565c00d; font-size: 14px; }
h2 { font-size: 17px; margin: 26px 0 8px; }
code { font: 13px ui-monospace, monospace; background: #8881; padding: 1px 5px;
       border-radius: 4px; }
"""


def _row(label: str, value) -> str:
    return f"<tr><th>{label}</th><td>{value}</td></tr>"


def _ellipse_words(result: LocalizationResult) -> str:
    """Словами про эллипс — и молчание там, где он ввёл бы в заблуждение.

    У непринятой позы эллипс считается по тем же невязкам, что и у принятой, но
    означает совсем другое: это разброс подгонки, которая проверку не прошла.
    Субметровое число рядом со словами «НИЗКОЕ ДОВЕРИЕ» читается как «очень
    точно» — ровно наоборот смыслу. Поймано на ``DSC00045``, где связка отвергла
    позу, построенную по пустой подложке, а в шапке стояло «эллипс 0.45 м».
    """
    if result.status is not Status.LOCALIZED:
        return ("не показывается: поза не принята. Эллипс описывает разброс "
                "подгонки, а не то, насколько верно найдено место.")
    if not result.error_ellipse_m:
        return "—"
    major, minor, _ = result.error_ellipse_m
    if major < 0.05:
        return ("менее 0.1 м (1σ) — это <b>случайная</b> часть ошибки, и она здесь "
                "пренебрежимо мала. Реальная точность определяется геопривязкой самой "
                "подложки: единицы метров.")
    return (f"{major:.2f} × {minor:.2f} м (1σ) — это <b>случайная</b> часть ошибки. "
            f"Абсолютная точность дополнительно ограничена геопривязкой самой "
            f"подложки: единицы метров.")


def _embed(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _html(request: LocateRequest, result: LocalizationResult, payload: dict,
          overlay: np.ndarray | None, *, matcher: str, region: str | None) -> str:
    title, css_class = _STATUS_RU[result.status]
    parts = [f"<style>{_CSS}</style>",
             f"<h1>Локализация: {request.image_path.name}</h1>",
             f"<div class='sub'>{payload['время']} · ядро <code>{matcher}</code></div>"]

    coord = ""
    if result.center_lat is not None:
        lat, lon = result.center_lat, result.center_lon
        coord = (f"<div class='coord'>{lat:.6f}, {lon:.6f}</div>"
                 f"<a href='https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=17/{lat}/{lon}'"
                 f" target='_blank'>открыть на карте</a>")
    parts.append(
        f"<div class='verdict {css_class}'><div class='big'>{title}</div>{coord}</div>")

    if overlay is not None:
        parts.append("<h2>Проверьте глазами</h2>")
        parts.append(
            "<div class='sub'>Панели 1–2: то ли это место. Панель 3 (шахматка): "
            "насколько точно совмещено. Панель 4: где в районе.</div>")
        parts.append(f"<img src='{_embed(overlay)}' alt='оверлей'>")

    parts.append("<h2>Результат</h2><table>")
    if result.center_lat is not None:
        parts.append(_row("Координаты центра",
                          f"{result.center_lat:.6f}, {result.center_lon:.6f}"))
        parts.append(_row("Курс кадра", f"{result.heading_deg:.1f}°"
                          if result.heading_deg is not None else "—"))
        parts.append(_row("Оценка высоты", f"{result.altitude_est_m:.0f} м"
                          if result.altitude_est_m is not None else "—"))
    parts.append(_row("Эллипс ошибки", _ellipse_words(result)))
    diag = result.diagnostics or {}
    if diag.get("n_inliers") is not None:
        parts.append(_row("Инлайеров", diag["n_inliers"]))
    if diag.get("photometric") is not None:
        kind = diag.get("photometric_kind", "мера")
        parts.append(_row(f"Согласие с подложкой ({kind})", f"{float(diag['photometric']):.3f}"))
    if diag.get("reason"):
        parts.append(_row("Причина отказа", diag["reason"]))
    parts.append("</table>")

    parts.append("<h2>Что было на входе</h2><table>")
    parts.append(_row("Снимок", f"{request.camera.image_width}×{request.camera.image_height}"))
    w, h = request.footprint_m
    parts.append(_row("GSD", f"{request.gsd_m:.4f} м/пиксель ({request.gsd_source})"))
    parts.append(_row("Кадр покрывает", f"<b>{w:.0f} × {h:.0f} м</b> — проверьте, похоже ли"))
    parts.append(_row("Приор", f"{request.prior.lat:.5f}, {request.prior.lon:.5f} "
                               f"±{request.prior.sigma_m:.0f} м ({request.prior_source})"))
    parts.append(_row("Курс", f"{request.prior.yaw_deg:.0f}°" if request.trust_yaw
                              else "неизвестен"))
    if region:
        parts.append(_row("Карта района", region))
    for stage, seconds in payload["время_стадий_с"].items():
        parts.append(_row(f"Время: {stage}", f"{seconds} с"))
    parts.append("</table>")

    if request.notes:
        parts.append("<h2>На что обратить внимание</h2>")
        parts.extend(f"<div class='note'>{n}</div>" for n in request.notes)

    tips = payload["советы"]
    if tips:
        parts.append("<h2>Что можно поменять</h2>")
        parts.extend(f"<div class='tip'>{t}</div>" for t in tips)

    parts.append("<h2>Файлы рядом</h2><div class='sub'>"
                 "<code>result.json</code> — числа для скриптов · "
                 "<code>footprint.kml</code> — Google Earth · "
                 "<code>footprint.geojson</code> — QGIS · "
                 "<code>overlay.png</code> — картинка отдельно</div>")
    return "\n".join(parts)


def save_report(
    out_dir: str | Path,
    request: LocateRequest,
    result: LocalizationResult,
    *,
    overlay: np.ndarray | None = None,
    matcher: str = "",
    timings: dict[str, float] | None = None,
    region: str | None = None,
) -> Path:
    """Записать все артефакты. Возвращает путь к ``report.html``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = result_payload(request, result, matcher=matcher, timings=timings, region=region)

    (out / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    name = request.image_path.stem
    (out / "footprint.geojson").write_text(
        json.dumps(footprint_geojson(result, name=name), ensure_ascii=False, indent=2),
        encoding="utf-8")
    (out / "footprint.kml").write_text(footprint_kml(result, name=name), encoding="utf-8")
    if overlay is not None:
        cv2.imwrite(str(out / "overlay.png"), overlay)

    report = out / "report.html"
    report.write_text(
        _html(request, result, payload, overlay, matcher=matcher, region=region),
        encoding="utf-8")
    return report


def save_summary(out_dir: str | Path, rows: list[dict]) -> Path:
    """Сводка по пачке снимков: таблица со ссылками на отдельные отчёты.

    Нужна не ради красоты. При прогоне серии главный вопрос — «сколько взято и
    что с остальными», и отвечать на него, открывая двадцать отчётов по одному,
    невозможно. Строки-отказы здесь такие же полноправные, как успехи: по ним
    видно, повторяется ли причина.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    total = len(rows)
    localized = sum(1 for r in rows if r["status"] == Status.LOCALIZED.value)
    refused = total - localized

    parts = [f"<style>{_CSS}</style>", "<h1>Сводка по локализации</h1>",
             f"<div class='sub'>{total} снимков · локализовано {localized} · "
             f"не принято {refused}</div>"]
    parts.append("<table><tr><th style='width:22%'>снимок</th><th>статус</th>"
                 "<th>координаты</th><th>эллипс</th><th>инлайеры</th><th>время</th></tr>")
    for r in rows:
        css = {"localized": "ok", "low_confidence": "warn"}.get(r["status"], "bad")
        coord = (f"{r['lat']:.6f}, {r['lon']:.6f}" if r.get("lat") is not None
                 else f"<span class='sub'>{r.get('reason', '') or '—'}</span>")
        parts.append(
            f"<tr><td><a href='{r['name']}/report.html'>{r['name']}</a></td>"
            f"<td class='{css}'>{_STATUS_RU[Status(r['status'])][0]}</td>"
            f"<td>{coord}</td><td>{r.get('ellipse', '—')}</td>"
            f"<td>{r.get('inliers', '—')}</td><td>{r.get('seconds', '—')} с</td></tr>")
    parts.append("</table>")
    if refused:
        parts.append(
            "<div class='tip'>Непринятые снимки — не обязательно ошибка: отказ "
            "легитимен и лучше уверенно-неверной точки. Откройте их отчёты: там "
            "причина и что можно поменять во входных данных.</div>")

    path = out / "summary.html"
    path.write_text(chr(10).join(parts), encoding="utf-8")
    return path
