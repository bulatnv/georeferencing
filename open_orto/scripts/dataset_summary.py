"""Сводка по каталогу поставки: README.md и визуальный отчёт SUMMARY.html.

Отличие от `corpus_report.py` — тот описывает рабочие корпуса, каждый в своём
каталоге; здесь всё уже собрано в один каталог с общим манифестом, и сводка
кладётся рядом с данными, чтобы датасет объяснял себя сам: кто его читает,
может не иметь под рукой ни репозитория, ни этой переписки.

    python open_orto/scripts/dataset_summary.py --root openaerialmap_dataset
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

from report import auto_bins, hist_svg, load_pair, panel, residual_px  # noqa: E402

CORPUS_TITLE = {
    "base": "Орто ↔ спутниковая подложка",
    "ss": "Ортоплан сам на себя (same_source)",
    "pilot": "Пилотные площадки (орто ↔ подложка)",
}
CORPUS_NOTE = {
    "base": "боевой тип: сторона A — вид виртуального борта, отрендеренный из "
            "ортоплана, сторона B — кроп спутниковой подложки Esri. Учит "
            "переносу между источниками",
    "ss": "обе стороны из одного растра, разметка точна по построению. Учит "
          "инвариантности к ракурсу, наклону и сезонной перекраске",
    "pilot": "первые пары с подложкой, собранные до сеточного режима; аудит "
             "подтвердил разметку на всех трёх площадках",
}


def num(rows, key):
    out = []
    for r in rows:
        try:
            out.append(float(r.get(key, "")))
        except (TypeError, ValueError):
            continue
    return np.array(out)


def stat_line(a, fmt="{:.2f}"):
    if not len(a):
        return "—"
    return (f"{fmt.format(float(np.median(a)))} "
            f"<span class=dim>({fmt.format(float(a.min()))}–"
            f"{fmt.format(float(a.max()))})</span>")


def pick(rows, total):
    """Примеры: поровну по корпусам, внутри — по видам пар и высотам."""
    by = defaultdict(list)
    for r in rows:
        by[r["corpus"]].append(r)
    picked = []
    per_corpus = max(1, total // max(len(by), 1))
    for corpus, items in by.items():
        groups = defaultdict(list)
        for r in items:
            groups[(r.get("pair_kind", ""), r.get("вердикт", ""))].append(r)
        per = max(1, per_corpus // max(len(groups), 1))
        for _, g in sorted(groups.items(), key=lambda t: -len(t[1])):
            g = sorted(g, key=lambda r: float(r.get("height_m") or 0))
            idx = np.linspace(0, len(g) - 1, min(per, len(g))).round().astype(int)
            picked += [g[i] for i in dict.fromkeys(idx.tolist())]
        # добор до квоты корпуса: деление по группам почти всегда даёт
        # недобор, и без этого сводка теряет десяток примеров без причины
        chosen = {id(r) for r in picked}
        rest = [r for r in items if id(r) not in chosen]
        while sum(1 for r in picked if r["corpus"] == corpus) < per_corpus and rest:
            picked.append(rest.pop(len(rest) // 2))
    return picked[:total]


def readme(root: Path, rows, control) -> str:
    total_gb = sum(int(r["bytes"] or 0) for r in rows) / 2**30
    scenes = len({r["scene"] for r in rows})
    by_corpus = Counter(r["corpus"] for r in rows)
    kinds = Counter(r["pair_kind"] for r in rows)
    h = num(rows, "height_m")
    t = num(rows, "tilt_deg")

    lines = [
        "# openaerialmap_dataset",
        "",
        f"**{len(rows)} обучающих пар** с **{scenes} площадок**, {total_gb:.2f} ГБ.",
        "Пары собраны из открытых ортофотопланов и спутниковой подложки; формат",
        "записи — канонический (`docs/DATASET_SPEC_FINETUNE.md`, §2), тот же, что",
        "у конвертера OrthoLoC, поэтому корпуса можно смешивать без адаптеров.",
        "",
        "## Что внутри одной пары",
        "",
        "Каждый файл — `.npz` со следующими массивами:",
        "",
        "| ключ | что это |",
        "|---|---|",
        "| `image_a_jpeg` | сторона A, кадр виртуального борта (1024×576), JPEG в байтах |",
        "| `image_b_jpeg` | сторона B, кроп подложки или второй вид ортоплана, JPEG |",
        "| `warp_ab` | плотное соответствие A→B: для каждого пикселя A его координата в B (float16, NaN вне маски) |",
        "| `mask_ab` | маска ко-видимости: где соответствие определено |",
        "| `meta` | JSON с параметрами съёмки, привязки и происхождением |",
        "| `pinhole` | зарезервировано (`null`): камера у стороны B не пинхольная |",
        "",
        "Разметка задана **попиксельным соответствием плюс маской**, а не",
        "глубиной с матрицами камер: у ортоплана и подложки нет ни карты глубины,",
        "ни внешней ориентации, и кодировать их через Depth+K+T было бы выдумкой.",
        "",
        "## Состав",
        "",
        "| корпус | пар | площадок | что даёт |",
        "|---|---|---|---|",
    ]
    for c, n in by_corpus.most_common():
        sc = len({r["scene"] for r in rows if r["corpus"] == c})
        lines.append(f"| `{c}` — {CORPUS_TITLE.get(c, c)} | {n} | {sc} | "
                     f"{CORPUS_NOTE.get(c, '')} |")
    lines += [
        "",
        f"Виды пар: " + ", ".join(f"`{k}` — {v}" for k, v in kinds.most_common()) + ".",
        "",
        f"Высота съёмки {h.min():.0f}–{h.max():.0f} м (медиана {np.median(h):.0f}), "
        f"наклон камеры 0–{t.max():.0f}° (медиана {np.median(t):.1f}°).",
        "",
        "## Точность разметки — разная у частей, и это важно",
        "",
        "| часть | ошибка разметки | чем измерена |",
        "|---|---|---|",
        f"| `same_source` | **{control['ss']:.3f} px** | фазовой корреляцией: "
        "стороны различаются только геометрией, измеритель точен |",
        "| орто ↔ подложка | **≈3–4.5 px (около 1–1.5 м)** | независимым матчером: "
        "геопривязка ортоплана, мозаика подложки, параллакс и разный рельеф |",
        "",
        "Практическое следствие: порог инлайера 3 px, осмысленный для",
        "`same_source`, на парах с подложкой лежит **внутри собственного шума",
        "разметки**. Метрики по ним читать на 5 и 10 px.",
        "",
        "Отдельно: фазовая корреляция на парах с подложкой **не годится как",
        "контроль** — при смене фактуры (ортоплан снят зимой, подложка летняя)",
        "она вырождается в шум и показывала 37 px там, где матчер уверенно давал",
        "3 px. Поэтому качество этих пар проверял матчер, причём как измеритель:",
        "его соответствия в разметку не записывались.",
        "",
        "## Как это собрано",
        "",
        "1. **Гейт привязки.** По каждой площадке до генерации меряется поле",
        "   сдвигов «ортоплан ↔ подложка» (двухступенчатая фазовая корреляция с",
        "   контролем остатка и фильтром по соседям). Площадка без измеренной",
        "   привязки пропускается: пара с непокрытым систематическим сдвигом",
        "   выглядит нормальной и учит модель неверному соответствию.",
        "2. **Сеточная генерация.** Территория режется на непересекающиеся ячейки",
        "   400 м, в каждой берётся несколько пар — объём выводится из площади",
        "   съёмки, а не назначается числом.",
        "3. **Аудит.** Выборка пар каждой площадки проверяется независимым",
        "   матчером; площадки со сдвинутой привязкой в поставку не попадают.",
        "",
        "## Чего здесь нет",
        "",
        "- пар с площадок, где аудит нашёл сдвиг привязки (они в карантине",
        "  рабочего каталога — материал для разбора, не для обучения);",
        "- адаптеров под тренеры: датасет хранит готовый плотный warp, а",
        "  romatch и LoFTR строят его сами из depth+K+T — нужна подмена одной",
        "  функции. Что именно и где расходится с исходной спецификацией —",
        "  `METHODOLOGY.md`, §10 «Готовность к обучению»;",
        "- растров в градусных CRS: у них единицы сетки не метры, поддержка",
        "  отложена;",
        "- сплитов train/val. Делить надо **по площадке целиком** (`scene`):",
        "  одна площадка присутствует и в парах с подложкой, и в `same_source`,",
        "  и при делении по парам один участок попал бы в обе части.",
        "",
        "## Источник и лицензия",
        "",
        "Ортофотопланы — открытые снимки (OpenAerialMap и аналогичные наборы);",
        "подложка — Esri World Imagery. Условия использования определяются",
        "лицензиями исходников и провайдера подложки: **уточните их перед",
        "публикацией или передачей датасета**.",
        "",
        "## Документы рядом",
        "",
        "- `METHODOLOGY.md` — как датасет получен: что было на входе, что и",
        "  почему отсеяно, как меряется привязка, как проверялось качество и",
        "  какие пороги пришлось пересмотреть по ходу;",
        "- `RASTERS.md` и `rasters_inventory.csv` — опись исходных ортопланов:",
        "  какие пошли в дело, какие нет и по какой причине;",
        "- `MATCHERS_REPORT.html` и `MATCHERS_METRICS.md` — как на этом корпусе",
        "  работают пять матчеров (LoFTR, MINIMA-LoFTR, RoMa v1, MINIMA-RoMa,",
        "  RoMa v2): отправная точка, от которой считается выигрыш дообучения;",
        "- `PAIR_ANATOMY.html` — разбор одной пары: что лежит внутри `.npz`,",
        "  как выглядит каждый массив и как прочитать пару в коде;",
        "- `SUMMARY.html` — распределения параметров и примеры пар;",
        "- `manifest.csv` — полная опись: одна строка на пару.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="openaerialmap_dataset")
    ap.add_argument("--examples", type=int, default=90)
    ap.add_argument("--panel-width", type=int, default=1300)
    ap.add_argument("--control", type=int, default=200,
                    help="сколько пар прогнать через контроль разметки")
    args = ap.parse_args()
    from cpu_affinity import pin_to_performance
    pin_to_performance(verbose=False)

    root = Path(args.root)
    rows = list(csv.DictReader((root / "manifest.csv").open(encoding="utf-8")))
    print(f"пар в поставке: {len(rows)}", flush=True)

    # контроль разметки: на same_source он точен, на подложке — справочен
    print("контроль разметки...", flush=True)
    vals = defaultdict(list)
    files = sorted(root.glob("*.npz"))
    step = max(1, len(files) // args.control)
    for f in files[::step][: args.control]:
        try:
            pair = load_pair(f)
        except Exception:  # noqa: BLE001
            continue
        meta = pair["meta"] if isinstance(pair["meta"], dict) else json.loads(pair["meta"])
        r, peak = residual_px(pair)
        if r is None or peak < 0.004:
            continue
        vals["ss" if meta.get("pair_kind") == "same_source" else "base"].append(r)
    control = {k: float(np.median(v)) if v else float("nan") for k, v in vals.items()}
    control.setdefault("ss", float("nan"))
    control.setdefault("base", float("nan"))

    (root / "README.md").write_text(readme(root, rows, control), encoding="utf-8")
    print(f"README: {root / 'README.md'}")

    # ——— HTML-сводка
    total_gb = sum(int(r["bytes"] or 0) for r in rows) / 2**30
    parts = [f"""<h1>openaerialmap_dataset</h1>
<p class=lead><b>{len(rows)}</b> обучающих пар с <b>{len({r['scene'] for r in rows})}</b>
площадок, {total_gb:.2f} ГБ. Формат записи канонический: попиксельное
соответствие A→B плюс маска ко-видимости.</p>
<h2>Состав</h2>
<table><tr><th>корпус<th>пар<th>площадок<th>что даёт</tr>"""]
    for c, n in Counter(r["corpus"] for r in rows).most_common():
        sc = len({r["scene"] for r in rows if r["corpus"] == c})
        parts.append(f"<tr><td><b>{html.escape(CORPUS_TITLE.get(c, c))}</b><td>{n}"
                     f"<td>{sc}<td>{html.escape(CORPUS_NOTE.get(c, ''))}</tr>")
    parts.append("</table>")

    parts.append(f"""<h2>Точность разметки</h2>
<p>У частей она <b>разная</b>, и это свойство источников, а не недоделка.</p>
<table><tr><th>часть<th>ошибка разметки<th>чем измерена</tr>
<tr><td>same_source<td><b>{control['ss']:.3f} px</b><td>фазовой корреляцией:
стороны различаются только геометрией</tr>
<tr><td>орто ↔ подложка<td><b>≈3–4.5 px (1–1.5 м)</b><td>независимым матчером:
геопривязка, мозаика подложки, параллакс, рельеф</tr></table>
<p class=note>Порог инлайера 3 px, осмысленный для <code>same_source</code>, на
парах с подложкой лежит внутри собственного шума разметки — читайте метрики по
ним на 5 и 10 px. Фазовая корреляция на этих парах контролем служить не может:
при смене фактуры она показывала 37 px там, где матчер уверенно давал 3 px.</p>""")

    parts.append("<h2>Параметры съёмки</h2>")
    for c in ("base", "ss", "pilot"):
        sub = [r for r in rows if r["corpus"] == c]
        if not sub:
            continue
        parts.append(f"<h3>{html.escape(CORPUS_TITLE.get(c, c))} — {len(sub)} пар</h3>")
        parts.append("<table><tr><th>параметр<th>медиана (мин–макс)</tr>")
        for label, k, fmt in (("высота съёмки, м", "height_m", "{:.0f}"),
                              ("наклон камеры, °", "tilt_deg", "{:.1f}"),
                              ("масштаб A/B", "scale_ratio", "{:.2f}"),
                              ("сторона кропа B, px", "b_px", "{:.0f}"),
                              ("ко-видимость", "covis_frac", "{:.2f}")):
            parts.append(f"<tr><td>{label}<td>{stat_line(num(sub, k), fmt)}</tr>")
        parts.append("</table>")
        for label, k, fmt in (("Высота съёмки, м", "height_m", lambda v: f"{v:.0f}"),
                              ("Наклон камеры, °", "tilt_deg", lambda v: f"{v:.0f}")):
            a = num(sub, k)
            if len(a):
                parts.append(hist_svg(a, auto_bins(a), label, fmt))
        for k, label in (("layout", "компоновка"), ("season", "сезонная перекраска"),
                         ("compensation_src", "источник компенсации привязки"),
                         ("вердикт", "вердикт аудита")):
            cnt = Counter(r.get(k, "") for r in sub if r.get(k))
            if cnt:
                parts.append(f"<p><b>{label}:</b> " + ", ".join(
                    f"{html.escape(str(v))} — {n}" for v, n in cnt.most_common()) + "</p>")

    print("галерея...", flush=True)
    picked = pick(rows, args.examples)
    parts.append(f"""<h2>Примеры</h2>
<p>В каждой панели: <b>кадр A</b> | <b>сторона B</b> | <b>наложение</b> — кадр A
натянут на B шахматкой, жёлтым обведён его след. Совпадают ли объекты на
границах клеток — и есть проверка разметки глазами.</p>""")
    n_panels = 0
    seen = None
    for r in picked:
        f = root / r["pair"]
        if not f.exists():
            continue
        try:
            pair = load_pair(f)
        except Exception:  # noqa: BLE001
            continue
        meta = pair["meta"] if isinstance(pair["meta"], dict) else json.loads(pair["meta"])
        if r["corpus"] != seen:
            seen = r["corpus"]
            parts.append(f"<h3>{html.escape(CORPUS_TITLE.get(seen, seen))}</h3>")
        note = " | ".join(x for x in [
            r["corpus"], meta.get("pair_kind", ""), str(meta.get("season_a") or ""),
            f"H {float(meta.get('height_m', 0)):.0f} m",
            f"tilt {float(meta.get('tilt_deg', 0)):.1f}",
            f"scale {float(meta.get('scale_ratio', 0)):.2f}",
            str(meta.get("compensation_src") or ""), r.get("вердикт", "")] if x and x != "None")
        img = panel(pair, note, max_width=args.panel_width)
        parts.append(f'<figure><img loading="lazy" src="data:image/jpeg;base64,{img}"/>'
                     f'<figcaption>{html.escape(r["pair"])}</figcaption></figure>')
        n_panels += 1
        if n_panels % 25 == 0:
            print(f"  {n_panels}/{len(picked)}", flush=True)

    out = root / "SUMMARY.html"
    out.write_text(f"""<!doctype html><html lang=ru><meta charset=utf-8>
<title>openaerialmap_dataset — сводка</title>
<style>body{{font:15.5px/1.55 Georgia,serif;max-width:1060px;margin:0 auto;padding:24px 18px 70px;color:#1a1a1a}}
h1{{font-size:27px;margin:0 0 6px}} h2{{margin-top:34px;border-top:1px solid #ddd;padding-top:16px}}
h3{{margin-top:22px;font-size:18px}} .lead{{font-size:17px;color:#333}}
table{{border-collapse:collapse;margin:12px 0;font-size:14.5px}}
td,th{{border:1px solid #ddd;padding:5px 10px;text-align:left;vertical-align:top}}
th{{background:#f4f4f4}} .dim{{color:#888}}
.note{{color:#555;font-size:14px;background:#f8f8f6;border-left:3px solid #ccc;padding:8px 12px}}
figure{{margin:22px 0}} img{{max-width:100%;border:1px solid #ccc}}
figcaption{{color:#777;font-size:12.5px;margin-top:4px;font-family:monospace}}
code{{background:#f2f2f0;padding:1px 4px}}</style>
{"".join(parts)}
</html>""", encoding="utf-8")
    print(f"сводка: {out} ({out.stat().st_size / 2**20:.1f} МБ), примеров {n_panels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
