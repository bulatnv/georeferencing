"""Визуальная сводка поставки OrthoLoC: SUMMARY.html рядом с данными.

Датасет должен объяснять себя сам: тот, кто его открыл, может не иметь под
рукой ни репозитория, ни переписки. Поэтому здесь всё, что нужно, чтобы
понять корпус за один заход — состав, съёмочный конверт с гистограммами,
происхождение и точность разметки, сплиты, базовая линия ядер, разрезы метрик
и галерея живых примеров каждого вида пар.

Панели галереи строятся тем же кодом, что у своего корпуса
(`open_orto/scripts/report.py`): кадр A, сторона B и шахматка — сторона B,
натянутая на геометрию кадра по разметке. Совпадение объектов на границах
клеток и есть проверка разметки глазами.

    python scripts/ortholoc_report.py --root ortholoc_dataset
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
sys.path.insert(0, str(_HERE.parents[1] / "open_orto" / "scripts"))

from report import auto_bins, hist_svg, load_pair, panel, residual_px  # noqa: E402

KIND_TITLE = {
    "frame_xdop": "Кадр ↔ чужое ортофото",
    "frame_dop": "Кадр ↔ своё ортофото",
    "rect_ortho": "Ортофото ↔ ортофото (кадр ректифицирован)",
}
KIND_NOTE = {
    "frame_xdop": "боевой тип: сторона B снята другим источником в другую дату, "
                  "поэтому к геометрии добавляется разрыв во внешнем виде",
    "frame_dop": "тот же источник, что у кадра: остаётся только трудность "
                 "наклонной съёмки против ортографической проекции",
    "rect_ortho": "контроль забывания: кадр орто-ректифицирован на сетку "
                  "ортофото, обе стороны на одной геометрии",
}


def num(rows, key):
    v = []
    for r in rows:
        try:
            x = float(r[key])
        except (TypeError, ValueError, KeyError):
            continue
        if np.isfinite(x):
            v.append(x)
    return np.array(v)


def stat_line(a, fmt="{:.2f}"):
    if not len(a):
        return "—"
    return (f"{fmt.format(float(np.median(a)))} "
            f"({fmt.format(float(a.min()))}–{fmt.format(float(a.max()))})")


def pick_examples(rows, per_kind: int, seed: int = 20260901):
    """Примеры по видам пар: разные сцены, разный наклон.

    Случайная выборка кучкуется на крупных сценах, а показать надо разнообразие
    корпуса, поэтому сначала перебираются сцены, и только внутри сцены берётся
    случайная пара.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for kind in KIND_TITLE:
        sub = [r for r in rows if r["pair_kind"] == kind]
        by_scene = defaultdict(list)
        for r in sub:
            by_scene[r["scene"]].append(r)
        scenes = sorted(by_scene)
        rng.shuffle(scenes)
        chosen, i = [], 0
        while len(chosen) < per_kind and scenes:
            pool = by_scene[scenes[i % len(scenes)]]
            if pool:
                chosen.append(pool.pop(int(rng.integers(len(pool)))))
            i += 1
            if i > 8 * per_kind:
                break
        out[kind] = chosen
    return out


def control_residual(root: Path, rows, limit_per_kind: int):
    """Остаток разметки фазовой корреляцией — по видам пар.

    На `rect_ortho` измеритель точен (обе стороны ортографичны и различаются
    только сдвигом сетки), на парах с наклонным кадром он видит и параллакс, и
    рельеф, поэтому там его число — верхняя оценка, а не ошибка разметки.
    """
    vals = defaultdict(list)
    for kind in KIND_TITLE:
        sub = [r for r in rows if r["pair_kind"] == kind]
        step = max(1, len(sub) // max(limit_per_kind, 1))
        for r in sub[::step][:limit_per_kind]:
            try:
                pair = load_pair(root / r["pair"])
            except Exception:  # noqa: BLE001
                continue
            res, peak = residual_px(pair)
            if res is None or peak < 0.004:
                continue
            vals[kind].append(res)
    return {k: (float(np.median(v)), len(v)) for k, v in vals.items() if v}


def bench_slices(bench_csv: Path, rows) -> list:
    """Разрезы базовой линии: где ядру трудно и почему."""
    if not bench_csv.exists():
        return []
    kind_of = {r["pair"].replace(".npz", ""): r["pair_kind"] for r in rows}
    tilt_of = {r["pair"].replace(".npz", ""): float(r["tilt_deg"]) for r in rows}
    height_of = {}
    for r in rows:
        try:
            height_of[r["pair"].replace(".npz", "")] = float(r["height_m"])
        except (TypeError, ValueError):
            pass
    data = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(bench_csv.open(encoding="utf-8")):
        if r["matcher"] == "matcher":
            continue
        name = r["pair"]
        try:
            inl5 = float(r["inl5_frac"])
        except (TypeError, ValueError):
            continue
        kind, tilt = kind_of.get(name), tilt_of.get(name)
        h = height_of.get(name)
        if kind is None:
            continue
        data[r["matcher"]]["всё"].append(inl5)
        data[r["matcher"]][KIND_TITLE.get(kind, kind)].append(inl5)
        if tilt is not None:
            data[r["matcher"]]["наклон ≤ 15°" if tilt <= 15 else "наклон > 15°"].append(inl5)
        if h is not None:
            data[r["matcher"]]["высота ≤ 100 м" if h <= 100 else "высота > 100 м"].append(inl5)
    order = ["всё", KIND_TITLE["frame_xdop"], KIND_TITLE["frame_dop"],
             "наклон ≤ 15°", "наклон > 15°", "высота ≤ 100 м", "высота > 100 м"]
    matchers = sorted(data, key=lambda m: -float(np.median(data[m]["всё"] or [0])))
    return [(row, [(m, float(np.median(data[m][row])) if data[m][row] else None,
                    len(data[m][row])) for m in matchers]) for row in order]


def matchers_md(bench_csv: Path, rows, slices) -> str:
    """Таблица метрик пяти ядер на приёмке — в машиночитаемом виде рядом с данными."""
    kind_of = {r["pair"].replace(".npz", ""): r["pair_kind"] for r in rows}
    data = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(bench_csv.open(encoding="utf-8")):
        if r["matcher"] == "matcher":
            continue
        kind = kind_of.get(r["pair"])
        if kind is None:
            continue
        for key in ("epe_med_px", "inl1_frac", "inl3_frac", "inl5_frac",
                    "inl10_frac", "sec"):
            try:
                data[(r["matcher"], kind)][key].append(float(r[key]))
                data[(r["matcher"], "всё")][key].append(float(r[key]))
            except (TypeError, ValueError, KeyError):
                pass
        try:
            hit = 1.0 if float(r["inl5_frac"]) >= 0.5 else 0.0
            data[(r["matcher"], kind)]["success"].append(hit)
            data[(r["matcher"], "всё")]["success"].append(hit)
        except (TypeError, ValueError, KeyError):
            pass

    matchers = sorted({m for m, _ in data},
                      key=lambda m: float(np.median(data[(m, "всё")]["epe_med_px"])))
    out = [
        "# Пять ядер на приёмке OrthoLoC", "",
        "Прогон без дообучения на парах `heldout` (сцены L08, L50, L51 — их нет",
        "ни в обучении, ни в валидации). Метрики считаются против плотного GT",
        "пары: медиана по парам внутри сэмпла, затем медиана по сэмплам.", "",
        "Сырьё: `eval_out/ortholoc_bench_heldout.csv`. Правила чтения и потолки —",
        "`ACCEPTANCE.md`.", "",
        "## Боевой тип: кадр против чужого ортофото", "",
        "| ядро | EPE, px | inl1 | inl3 | inl5 | inl10 | успех | с/пару |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def line(m, kind):
        d = data[(m, kind)]
        g = lambda k: float(np.median(d[k])) if d[k] else float("nan")  # noqa: E731
        return (f"| {m} | {g('epe_med_px'):.2f} | {g('inl1_frac'):.2f} | "
                f"{g('inl3_frac'):.2f} | {g('inl5_frac'):.2f} | "
                f"{g('inl10_frac'):.2f} | {float(np.mean(d['success'])):.2f} | "
                f"{g('sec'):.2f} |")

    for m in matchers:
        out.append(line(m, "frame_xdop"))
    out += ["", "## Тот же кадр против своего ортофото", "",
            "Разница между этой таблицей и предыдущей — цена смены источника:",
            "геометрия та же, разметка той же природы, меняется только то, чем",
            "снята сторона B.", "",
            "| ядро | EPE, px | inl1 | inl3 | inl5 | inl10 | успех | с/пару |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in matchers:
        out.append(line(m, "frame_dop"))

    best = matchers[0]
    own = float(np.median(data[(best, "frame_dop")]["epe_med_px"]))
    foreign = float(np.median(data[(best, "frame_xdop")]["epe_med_px"]))
    out += ["",
            f"Лучшему ядру (**{best}**) смена источника стоит **×{foreign/own:.1f}** "
            f"по EPE: {own:.2f} → {foreign:.2f} px. Сдвиг привязки между",
            "источниками при этом всего 1.5 px, то есть трудность создаёт не",
            "геометрия, а разрыв во внешнем виде.", ""]

    if slices:
        out += ["## Разрезы (медиана `inl5`)", "",
                "| разрез | n | " + " | ".join(m for m, _, _ in slices[0][1]) + " |",
                "|---|---:|" + "---:|" * len(slices[0][1])]
        for label, cells in slices:
            n_ = cells[0][2] if cells else 0
            vals = " | ".join("—" if v is None else f"{v:.2f}" for _, v, _ in cells)
            out.append(f"| {label} | {n_} | {vals} |")
        out += ["",
                "**Разрезы по режиму съёмки здесь читать нельзя, и это проверено.**",
                "В приёмке всего три сцены, и режим совпадает с территорией:",
                "наклон ≤ 15° — это ровно сцена L50 (114 пар из 114), высоты",
                "выше 100 м — почти только L51 и L50, а 1301 пара из 1397 с",
                "наклоном > 15° приходится на L08. Поэтому строки «наклон» и",
                "«высота» сравнивают не ракурс и не высоту, а местность: где-то",
                "застройка, где-то поля. Настоящий разрез по режиму нужно считать",
                "**внутри сцены** и на `train`/`val`, где сцен сорок восемь.", "",
                "Единственный разрез, свободный от этой оговорки, — по виду пары:",
                "он делит одни и те же кадры одних и тех же территорий, меняя",
                "только источник стороны B.", ""]
    return "\n".join(out) + "\n"


CSS = """body{font:15.5px/1.62 Georgia,serif;max-width:1180px;margin:0 auto;
padding:24px 18px 80px;color:#1a1a1a;background:#fcfcfb}
h1{font-size:29px;margin:0 0 10px} h2{margin-top:40px;border-top:1px solid #ddd;
padding-top:16px;font-size:23px} h3{margin-top:26px;font-size:18px}
.lead{font-size:17.5px;color:#333}
table{border-collapse:collapse;margin:14px 0;font-size:14.5px;width:100%}
td,th{border:1px solid #ddd;padding:6px 10px;text-align:left;vertical-align:top}
th{background:#f4f4f4} td.n{font-family:ui-monospace,monospace;font-size:13.5px;
white-space:nowrap;text-align:right}
.note{color:#444;font-size:14.5px;background:#f8f8f6;border-left:3px solid #c9c9c0;
padding:10px 14px;margin:14px 0}
figure{margin:18px 0} img{max-width:100%;border:1px solid #ccc;display:block}
figcaption{color:#666;font-size:13px;margin-top:6px;line-height:1.45}
svg{max-width:100%;height:auto;margin:6px 0 18px}
code{background:#f0f0ec;padding:1px 4px;border-radius:3px;font-size:13.5px;
font-family:ui-monospace,monospace}
.grid{display:flex;flex-wrap:wrap;gap:10px 24px;margin:10px 0}
.kpi{border:1px solid #ddd;padding:10px 14px;min-width:150px}
.kpi b{display:block;font-size:22px;font-family:ui-monospace,monospace}
.kpi span{color:#666;font-size:13px}"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="ortholoc_dataset")
    ap.add_argument("--bench", default="eval_out/ortholoc_bench_heldout.csv")
    ap.add_argument("--audit", default="eval_out/ortholoc_audit.csv")
    ap.add_argument("--examples", type=int, default=6, help="панелей на вид пары")
    ap.add_argument("--control", type=int, default=60, help="пар на вид для контроля")
    ap.add_argument("--panel-width", type=int, default=1400)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    try:
        from cpu_affinity import pin_to_performance
        pin_to_performance(verbose=False)
    except Exception:  # noqa: BLE001
        pass

    root = Path(args.root)
    rows = list(csv.DictReader((root / "manifest.csv").open(encoding="utf-8")))
    gb = sum(int(r["bytes"] or 0) for r in rows) / 2**30
    kinds = Counter(r["pair_kind"] for r in rows)
    print(f"пар: {len(rows)}, {gb:.2f} ГБ", flush=True)

    print("контроль разметки...", flush=True)
    control = control_residual(root, rows, args.control)

    parts = [f"""<h1>ortholoc_dataset</h1>
<p class=lead><b>{len(rows)}</b> обучающих пар с <b>{len({r['scene'] for r in rows})}</b>
сцен, {gb:.2f} ГБ. Собрано из внешнего датасета <b>OrthoLoC</b> в том же
каноническом формате, что и свой корпус: попиксельное соответствие A→B плюс
маска ко-видимости.</p>
<div class=grid>
  <div class=kpi><b>{kinds['frame_xdop']}</b><span>кадр ↔ чужое ортофото</span></div>
  <div class=kpi><b>{kinds['frame_dop']}</b><span>кадр ↔ своё ортофото</span></div>
  <div class=kpi><b>{kinds['rect_ortho']}</b><span>контроль забывания</span></div>
  <div class=kpi><b>1.52 px</b><span>измеренный сдвиг привязки</span></div>
</div>

<h2>Состав</h2>
<table><tr><th>вид пары<th>пар<th>сцен<th>роль</tr>"""]
    for kind, title in KIND_TITLE.items():
        sub = [r for r in rows if r["pair_kind"] == kind]
        parts.append(f"<tr><td><b>{html.escape(title)}</b><td class=n>{len(sub)}"
                     f"<td class=n>{len({r['scene'] for r in sub})}"
                     f"<td>{html.escape(KIND_NOTE[kind])}</tr>")
    parts.append("</table>")

    # ——— конверт съёмки
    h, t = num(rows, "height_m"), num(rows, "tilt_deg")
    parts.append(f"""<h2>Съёмочный конверт</h2>
<p>Здесь принципиальное отличие от своего корпуса: съёмка <b>ниже и наклоннее</b>.
Высота {np.nanmin(h):.0f}–{np.nanmax(h):.0f} м против наших 175–300, наклон до
{np.nanmax(t):.0f}° против наших 20. Низкий режим, который мы откладывали как
«кропы однородны, учить нечему», здесь закрыт настоящей съёмкой; плата —
ракурсы, которых у борта в надирном полёте не бывает.</p>
<table><tr><th>параметр<th>медиана (мин–макс)</tr>""")
    for label, k, fmt in (("высота съёмки, м", "height_m", "{:.0f}"),
                          ("наклон камеры, °", "tilt_deg", "{:.1f}"),
                          ("ко-видимость", "covis_frac", "{:.2f}"),
                          ("масштаб A/B", "scale_ratio", "{:.2f}"),
                          ("сторона кропа B, px", "b_px", "{:.0f}")):
        parts.append(f"<tr><td>{label}<td class=n>{stat_line(num(rows, k), fmt)}</tr>")
    parts.append("</table>")
    for label, k, fmt in (("Высота съёмки, м", "height_m", lambda v: f"{v:.0f}"),
                          ("Наклон камеры, °", "tilt_deg", lambda v: f"{v:.0f}"),
                          ("Ко-видимость", "covis_frac", lambda v: f"{v:.2f}")):
        a = num(rows, k)
        if len(a):
            parts.append(hist_svg(a, auto_bins(a), label, fmt))

    # ——— разметка
    audit = Path(args.audit)
    shifts = []
    if audit.exists():
        shifts = [float(r["shift_px"]) for r in csv.DictReader(audit.open(encoding="utf-8"))
                  if r["status"] == "измерено" and r["shift_px"]]
    parts.append("""<h2>Откуда взята точность разметки</h2>
<p>Разметка здесь не наша: она приходит из <code>point_map</code> самого
OrthoLoC. Но у пар с <b>чужим</b> ортофото есть дефект, которого никто не
мерил. У вариантов «своё» и «чужое» общий кадр, общий <code>point_map</code> и
одинаковый масштаб, поэтому GT-карта у них <b>одна и та же</b> — формула молча
считает, что оба ортофото привязаны к мировой сетке идеально. Если чужое
смещено, весь сдвиг уходит прямо в разметку и глазами не виден.</p>""")
    if shifts:
        sh = np.array(shifts)
        parts.append(f"""<p>Замер: два ортофото одной территории сводятся
сильнейшим ядром, по его соответствиям строится 4-DoF подобие (обе стороны
ортографичны, модель применима), и сдвиг этой модели есть систематическая
ошибка разметки. <b>{len(sh)}</b> сведений по 28 сценам: медиана
<b>{np.median(sh):.2f} px</b> ({0.2*np.median(sh):.2f} м при типичном GSD 0.2 м),
90-й перцентиль {np.percentile(sh, 90):.2f} px, по сценам от 0.64 до 5.14 px.
Поворот 0.05°, масштаб 0.12 % — расхождение оказалось <b>чистым переносом</b>,
ровно как у нас между ортопланом и мозаикой Esri.</p>""")
        parts.append(hist_svg(sh, auto_bins(sh), "Сдвиг привязки источников, px",
                              lambda v: f"{v:.1f}"))
    parts.append("<table><tr><th>класс<th>пар<th>вес<th>ошибка, px<th>откуда</tr>")
    src_note = {"измерено": "сдвиг привязки, замеренный на этой сцене",
                "оценка": "прямого измерения нет — верхняя оценка",
                "аналитическая": "warp есть целочисленный сдвиг сеток"}
    for cls in ("registered", "approx", "exact"):
        sub = [r for r in rows if r["gt_class"] == cls]
        if not sub:
            continue
        srcs = Counter(r["gt_sigma_src"] for r in sub)
        sig = num(sub, "gt_sigma_px")
        parts.append(f"<tr><td><code>{cls}</code><td class=n>{len(sub)}"
                     f"<td class=n>{sub[0]['weight']}<td class=n>{stat_line(sig)}"
                     f"<td>" + "; ".join(
                         f"{html.escape(src_note.get(s, s))} — {n}"
                         for s, n in srcs.most_common()) + "</tr>")
    parts.append("</table>")
    if control:
        parts.append("""<h3>Контроль остатка</h3>
<p>Независимая проверка: сторона B натягивается на кадр по нашей же разметке, и
фазовая корреляция ищет остаточный сдвиг. На пары с ректифицированным кадром
измеритель годится полностью, на наклонных кадрах он видит ещё и параллакс с
рельефом — там это верхняя оценка, а не ошибка разметки.</p>
<table><tr><th>вид пары<th>остаток, px<th>пар в замере</tr>""")
        for kind, (val, n) in control.items():
            parts.append(f"<tr><td>{html.escape(KIND_TITLE.get(kind, kind))}"
                         f"<td class=n>{val:.2f}<td class=n>{n}</tr>")
        parts.append("</table>")

    # ——— сплиты
    parts.append("""<h2>Сплиты</h2>
<p>Деление <b>своё, а не из датасета</b>. В OrthoLoC <code>train</code> и
<code>val</code> — одни и те же 48 сцен, а <code>test_inPlace</code> входит в
<code>train</code>: по правилу «делить по территориям» это утечка. Честный
held-out здесь ровно один — сцены <code>test_outPlace</code>, не встречающиеся
больше нигде. Исходная метка сохранена колонкой <code>src_split</code>.</p>
<table><tr><th>сплит<th>пар<th>боевых<th>сцен<th>назначение</tr>""")
    for sp, note in (("train", "обучение"), ("val", "выбор чекпоинта"),
                     ("heldout", "приёмка, расходуется один раз")):
        sub = [r for r in rows if r["split"] == sp]
        nb = sum(1 for r in sub if r["pair_kind"] in ("frame_xdop", "frame_dop"))
        parts.append(f"<tr><td><code>{sp}</code><td class=n>{len(sub)}"
                     f"<td class=n>{nb}<td class=n>{len({r['scene'] for r in sub})}"
                     f"<td>{note}</tr>")
    parts.append("</table>")

    # ——— базовая линия и разрезы
    slices = bench_slices(Path(args.bench), rows)
    if slices:
        matchers = [m for m, _, _ in slices[0][1]]
        parts.append("""<h2>Базовая линия и разрезы</h2>
<p>Пять ядер без дообучения на парах приёмки. В ячейках — медиана
<code>inl5</code>: доля соответствий, легших ближе 5 px от истины. Разрезы
отвечают на вопрос, что именно даётся ядрам тяжело.</p>
<table><tr><th>разрез<th>n""")
        for m in matchers:
            parts.append(f"<th>{html.escape(m)}")
        parts.append("</tr>")
        for label, cells in slices:
            n = cells[0][2] if cells else 0
            parts.append(f"<tr><td>{html.escape(label)}<td class=n>{n}")
            best = max((v for _, v, _ in cells if v is not None), default=None)
            for _, v, _ in cells:
                if v is None:
                    parts.append("<td class=n>—")
                elif best is not None and abs(v - best) < 1e-9:
                    parts.append(f"<td class=n><b>{v:.2f}</b>")
                else:
                    parts.append(f"<td class=n>{v:.2f}")
            parts.append("</tr>")
        parts.append("</table>")
        parts.append("""<p class=note><b>Разрезы по наклону и высоте здесь
читать нельзя</b>, и это проверено: в приёмке всего три сцены, и режим съёмки
совпадает с территорией — наклон ≤ 15° это ровно сцена L50 (114 пар из 114),
а 1301 из 1397 пар с наклоном &gt; 15° приходится на L08. Такие строки
сравнивают местность, а не ракурс. Свободен от оговорки только разрез по виду
пары: он делит одни и те же кадры одних и тех же территорий, меняя лишь
источник стороны B.</p>
<p class=note>Главное число корпуса: смена источника
ортофото стоит лучшему ядру <b>×6 по EPE</b> (0.97 px на своём против 5.86 на
чужом) при том, что сдвиг привязки между источниками — всего 1.5 px. Трудность
создаёт не геометрия, а разрыв во внешнем виде: тот же вывод, что на своём
корпусе, теперь на данных, где обе стороны — ортофото.</p>""")

    # ——— галерея
    print("галерея...", flush=True)
    picks = pick_examples(rows, args.examples)
    parts.append("""<h2>Примеры пар</h2>
<p>Слева кадр, в середине сторона B, справа шахматка: сторона B, натянутая на
геометрию кадра по разметке. Если на границах клеток объекты продолжаются —
разметка верна.</p>""")
    for kind, title in KIND_TITLE.items():
        chosen = picks.get(kind, [])
        if not chosen:
            continue
        parts.append(f"<h3>{html.escape(title)} — {html.escape(KIND_NOTE[kind])}</h3>")
        for r in chosen:
            try:
                pair = load_pair(root / r["pair"])
            except Exception:  # noqa: BLE001
                continue
            note = (f"сцена {r['scene']}, наклон {float(r['tilt_deg']):.0f}°, "
                    f"высота {float(r['height_m']):.0f} м, "
                    f"ко-видимость {float(r['covis_frac']):.2f}, "
                    f"ошибка разметки {float(r['gt_sigma_px']):.2f} px "
                    f"({r['gt_sigma_src']})")
            # panel отдаёт base64 картинки, а не готовый тег
            img = panel(pair, note, max_width=args.panel_width)
            parts.append(f'<figure><img loading="lazy" '
                         f'src="data:image/jpeg;base64,{img}"/>'
                         f'<figcaption>{html.escape(note)} · '
                         f'<code>{html.escape(r["pair"])}</code></figcaption></figure>')
    parts.append("""<h2>Ограничения</h2>
<ul>
<li><b>Чужой домен.</b> Наклонная съёмка против ортофото — не наш боевой режим
(надир против спутниковой подложки). Доля этого корпуса в смеси — решение, а
не данность.</li>
<li><b>Разметка точнее нашей</b> (около 1 px против 4), поэтому на общих
метриках корпус будет выглядеть «лучше»: сравнивать корпуса между собой
бессмысленно, у каждого своя приёмка и свои потолки.</li>
<li><b>Лицензия CC BY-NC-SA 4.0</b> — некоммерческое использование; ограничение
переходит на всё, что из корпуса собрано.</li>
<li><b>Сезонной оси нет</b>: синтетические перекраски отброшены — замерено, что
разрыва они почти не создают.</li>
</ul>
<p style="color:#777;font-size:13px;margin-top:36px">Состав и правила —
<code>README.md</code>, метрики приёмки — <code>ACCEPTANCE.md</code>, как
корпус получен — <code>METHODOLOGY.md</code>, разбор одной пары —
<code>PAIR_ANATOMY.html</code>.</p>""")

    bench_path = Path(args.bench)
    if slices and bench_path.exists() and not args.out:
        md = root / "MATCHERS_METRICS.md"
        md.write_text(matchers_md(bench_path, rows, slices), encoding="utf-8")
        print(f"метрики ядер: {md}")

    out = Path(args.out) if args.out else root / "SUMMARY.html"
    out.write_text(f"""<!doctype html><html lang=ru><meta charset=utf-8>
<title>ortholoc_dataset — сводка</title>
<style>{CSS}</style>
{"".join(parts)}
</html>""", encoding="utf-8")
    print(f"сводка: {out} ({out.stat().st_size/2**20:.1f} МБ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
