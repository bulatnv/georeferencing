"""Сводный отчёт по всему собранному датасету: состав, распределения, примеры.

`report.py` описывает один каталог; здесь сводятся все корпуса разом —
`same_source`, пары с подложкой, пилотные три площадки и карантин, — потому что
судить о датасете надо целиком: у частей разная точность разметки и разное
назначение, и смешивать их при обучении можно только зная это.

Галерея набирается **стратифицированно**: по корпусам, видам пар, вердиктам
аудита и высотам съёмки. Случайная выборка из 5 тысяч пар показала бы одно и
то же — типичный кадр; интересны как раз края распределения.

    python open_orto/scripts/corpus_report.py --examples 150 \\
        --out open_orto/CORPUS_REPORT.html
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

from report import auto_bins, hist_svg, load_pair, panel, residual_px  # noqa: E402

#: Корпуса: (ключ, каталог, заголовок, назначение).
CORPORA = [
    ("base", "open_orto/dataset_base", "Пары «орто ↔ подложка»",
     "боевой тип: сторона A — вид виртуального борта из ортоплана, сторона B — "
     "кроп спутниковой подложки Esri. Учит переносу между источниками"),
    ("ss", "open_orto/dataset_ss", "Пары «ортоплан сам на себя» (same_source)",
     "обе стороны из одного растра: разметка точна по построению. Учит "
     "инвариантности к ракурсу, наклону и сезонной перекраске"),
    ("pilot", "open_orto/dataset", "Пилотный корпус трёх площадок",
     "первые пары с подложкой, снятые до сеточного режима; оставлены как "
     "исторический срез"),
    ("quar", "open_orto/dataset_base_quarantine", "Карантин",
     "пары площадок, где аудит нашёл сдвиг привязки. В обучение не идут"),
]


def read_manifest(root: Path):
    mf = root / "manifest.csv"
    if not mf.exists():
        return []
    return list(csv.DictReader(mf.open(encoding="utf-8")))


def num(rows, key):
    out = []
    for r in rows:
        v = r.get(key, "")
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return np.array(out)


def stat_line(a: np.ndarray, fmt="{:.2f}") -> str:
    if not len(a):
        return "—"
    return (f"{fmt.format(float(np.median(a)))} "
            f"<span class=dim>({fmt.format(float(a.min()))}–"
            f"{fmt.format(float(a.max()))})</span>")


def pick_examples(corpora: dict, total: int):
    """Стратифицированная выборка пар для галереи.

    Доли между корпусами — по их размеру, но с гарантированным минимумом на
    каждый: карантин мал, а показать его надо (без примера брака непонятно,
    от чего защищает аудит). Внутри корпуса — по виду пары и вердикту, а
    внутри группы — по высоте, чтобы попали и низкие, и высокие.
    """
    sizes = {k: len(v["rows"]) for k, v in corpora.items() if v["rows"]}
    if not sizes:
        return []
    floor = min(8, total // max(len(sizes), 1))
    left = total - floor * len(sizes)
    scale = left / max(sum(sizes.values()), 1)
    quota = {k: floor + int(n * scale) for k, n in sizes.items()}

    picked = []
    for key, q in quota.items():
        rows = corpora[key]["rows"]
        groups = defaultdict(list)
        for r in rows:
            groups[(r.get("pair_kind", ""), r.get("вердикт", ""))].append(r)
        per = max(1, q // max(len(groups), 1))
        for gk, items in sorted(groups.items(), key=lambda t: -len(t[1])):
            items = sorted(items, key=lambda r: float(r.get("height_m") or 0))
            take = min(per, len(items))
            if take:
                idx = np.linspace(0, len(items) - 1, take).round().astype(int)
                for i in dict.fromkeys(idx.tolist()):
                    picked.append((key, items[i]))
        # добор до квоты, если групп было мало
        rest = [r for r in rows if all(r is not p for _, p in picked)]
        while sum(1 for k, _ in picked if k == key) < q and rest:
            picked.append((key, rest.pop(len(rest) // 2)))
    return picked[:total]


def note_for(key: str, r: dict, meta: dict) -> str:
    bits = [key, meta.get("pair_kind", "")]
    if meta.get("season_a"):
        bits.append(str(meta["season_a"]))
    bits.append(f"H {float(meta.get('height_m', 0)):.0f} m")
    bits.append(f"tilt {float(meta.get('tilt_deg', 0)):.1f}")
    if meta.get("delta_yaw_deg") is not None:
        try:
            bits.append(f"dyaw {float(meta['delta_yaw_deg']):.0f}")
        except (TypeError, ValueError):
            pass
    bits.append(f"scale {float(meta.get('scale_ratio', 0)):.2f}")
    if meta.get("compensation_src"):
        bits.append(str(meta["compensation_src"]))
    if r.get("вердикт"):
        bits.append(r["вердикт"])
    return " | ".join(b for b in bits if b)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--examples", type=int, default=150)
    ap.add_argument("--panel-width", type=int, default=1300,
                    help="ширина панели, px: при 150 примерах натуральные 2000 "
                         "дают 60+ МБ, которые браузер не тянет")
    ap.add_argument("--residual-checks", type=int, default=250,
                    help="сколько пар прогнать через контроль разметки")
    ap.add_argument("--audit", default="open_orto/work/audit_basemap.csv")
    ap.add_argument("--out", default="open_orto/CORPUS_REPORT.html")
    args = ap.parse_args()
    from cpu_affinity import pin_to_performance
    pin_to_performance(verbose=False)

    corpora = {}
    for key, path, title, purpose in CORPORA:
        root = Path(path)
        rows = read_manifest(root)
        corpora[key] = dict(root=root, title=title, purpose=purpose, rows=rows,
                            files=len(list(root.glob("*.npz"))) if root.exists() else 0)
        print(f"{title}: пар {len(rows)}", flush=True)

    # ——— сводка
    total_pairs = sum(len(c["rows"]) for c in corpora.values())
    total_gb = sum(float(num(c["rows"], "bytes").sum()) for c in corpora.values()) / 2**30
    scenes_all = set()
    for c in corpora.values():
        scenes_all |= {r["scene"] for r in c["rows"]}

    parts = []
    parts.append(f"""<h1>Датасет open_orto: что собрано</h1>
<p class=lead>Всего <b>{total_pairs}</b> пар канонического формата
(DATASET_SPEC_FINETUNE §2) с <b>{len(scenes_all)}</b> площадок, {total_gb:.2f} ГБ.
Ниже — состав по корпусам, распределения параметров съёмки, измеренное качество
разметки и @EXAMPLES@ примеров.</p>

<h2>1. Корпуса и их назначение</h2>
<p>Датасет собран из частей с <b>разной точностью разметки</b>. Это не
недоделка, а свойство источников, и при обучении это надо учитывать: смешивать
их можно, но порог инлайера, осмысленный для одной части, для другой лежит
внутри её собственного шума.</p>
<table><tr><th>корпус<th>пар<th>площадок<th>ГБ<th>назначение</tr>""")
    for key, c in corpora.items():
        if not c["rows"]:
            continue
        gb = float(num(c["rows"], "bytes").sum()) / 2**30
        parts.append(f"<tr><td><b>{html.escape(c['title'])}</b><td>{len(c['rows'])}"
                     f"<td>{len({r['scene'] for r in c['rows']})}<td>{gb:.2f}"
                     f"<td>{html.escape(c['purpose'])}</tr>")
    parts.append("</table>")

    # ——— качество
    parts.append("""<h2>2. Качество разметки</h2>
<p>Двумя разными измерителями, и это не перестраховка: у них разные области
применимости.</p>
<p><b>Контроль по фазовой корреляции</b> берёт кадр A и натянутую на него
подложку B и ищет остаточный сдвиг. На парах <code>same_source</code> он точен
до сотых пикселя. На парах с подложкой он <b>врёт при смене фактуры</b>: если
ортоплан снят зимой, а подложка летняя, корреляция по градиентам вырождается в
шум — замерено 37 px там, где матчер уверенно даёт 3 px.</p>
<p><b>Аудит матчером</b> (MINIMA-RoMa) сравнивает нашу разметку с независимой
моделью соответствий. Матчер здесь <b>измеритель, а не источник разметки</b>:
его соответствия никуда не записываются, иначе модель обучалась бы на
собственных предсказаниях.</p>""")

    audit_rows = []
    ap_path = Path(args.audit)
    if ap_path.exists():
        audit_rows = list(csv.DictReader(ap_path.open(encoding="utf-8")))
    if audit_rows:
        tally = Counter(r["вердикт"] for r in audit_rows)
        diffs = np.array([float(r["gt_diff_med"]) for r in audit_rows
                          if r["вердикт"] == "разметка подтверждена" and r["gt_diff_med"]])
        parts.append("<table><tr><th>вердикт аудита<th>площадок<th>что значит</tr>")
        expl = {
            "разметка подтверждена": "матчер согласован и с собой, и с нашей разметкой",
            "привязка сдвинута": "матчер согласован с собой, но расходится с нами > 12 px",
            "без структуры": "матчер не согласован сам с собой: поле, лес, вода",
            "проверено частично": "подтверждений меньше трети проверенных пар",
        }
        for v, n in tally.most_common():
            parts.append(f"<tr><td>{html.escape(v)}<td>{n}"
                         f"<td>{html.escape(expl.get(v, ''))}</tr>")
        parts.append("</table>")
        if len(diffs):
            parts.append(f"<p>Расхождение с матчером на подтверждённых площадках: "
                         f"<b>{np.median(diffs):.2f} px</b> (p90 "
                         f"{np.percentile(diffs, 90):.2f}). В метрах это около "
                         f"{np.median(diffs) * 0.3:.1f} м при типичном GSD подложки "
                         f"0.3 м/пкс.</p>")

    # контроль разметки по корпусам
    print("контроль разметки...", flush=True)
    control = {}
    for key in ("base", "ss", "pilot", "quar"):
        c = corpora.get(key)
        if not c or not c["rows"]:
            continue
        files = sorted(c["root"].glob("*.npz"))
        per_corpus = max(args.residual_checks // 4, 1)
        step = max(1, len(files) // per_corpus)
        vals_ss, vals_bm = [], []
        for f in files[::step][:per_corpus]:
            try:
                pair = load_pair(f)
            except Exception:  # noqa: BLE001
                continue
            meta = pair["meta"] if isinstance(pair["meta"], dict) else json.loads(pair["meta"])
            r, peak = residual_px(pair)
            if r is None or peak < 0.004:
                continue
            (vals_ss if meta.get("pair_kind") == "same_source" else vals_bm).append(r)
        control[key] = (np.array(vals_ss), np.array(vals_bm))

    parts.append("<table><tr><th>корпус<th>контроль same_source, px"
                 "<th>фазовый замер на парах с подложкой, px</tr>")
    for key, (ss, bm) in control.items():
        t = corpora[key]["title"]
        s1 = f"{np.median(ss):.3f} <span class=dim>(n={len(ss)})</span>" if len(ss) else "—"
        s2 = f"{np.median(bm):.2f} <span class=dim>(n={len(bm)})</span>" if len(bm) else "—"
        parts.append(f"<tr><td>{html.escape(t)}<td>{s1}<td>{s2}</tr>")
    parts.append("""</table>
<p class=note><b>Второй столбец — не ошибка разметки, а показание измерителя.</b>
Нагляднее всего это видно на карантине: фазовый замер даёт там около 3 px, то
есть «лучше» пилотного корпуса, — тогда как матчер на тех же площадках находит
сдвиг <b>18.2 px (5.0 м)</b>, и именно за это они и отправлены в карантин.
Судить о парах с подложкой по фазовой корреляции нельзя; она оставлена в
отчёте лишь как контроль <code>same_source</code>, где точна.</p>
<table><tr><th>измеритель<th>карантин<th>подтверждённые площадки</tr>
<tr><td>матчер (арбитр)<td><b>18.2 px</b><td>4.1 px</tr>
<tr><td>фазовая корреляция<td>3.1 px<td>0.5–2.4 px</tr></table>""")

    # ——— распределения
    parts.append("<h2>3. Параметры съёмки</h2>")
    for key in ("base", "ss"):
        c = corpora.get(key)
        if not c or not c["rows"]:
            continue
        rows = c["rows"]
        parts.append(f"<h3>{html.escape(c['title'])}</h3>")
        parts.append("<table><tr><th>параметр<th>медиана (мин–макс)</tr>")
        for label, k, fmt in (("высота съёмки, м", "height_m", "{:.0f}"),
                              ("наклон камеры, °", "tilt_deg", "{:.1f}"),
                              ("курс кадра, °", "yaw_deg", "{:.0f}"),
                              ("масштаб A/B", "scale_ratio", "{:.2f}"),
                              ("сторона кропа B, px", "b_px", "{:.0f}"),
                              ("ко-видимость", "covis_frac", "{:.2f}"),
                              ("размер пары, МБ", "bytes", "{:.1f}")):
            a = num(rows, k)
            if k == "bytes":
                a = a / 2**20
            parts.append(f"<tr><td>{label}<td>{stat_line(a, fmt)}</tr>")
        parts.append("</table>")
        for label, k, fmt in (("Высота съёмки, м", "height_m", lambda v: f"{v:.0f}"),
                              ("Наклон камеры, °", "tilt_deg", lambda v: f"{v:.0f}"),
                              ("Ко-видимость", "covis_frac", lambda v: f"{v:.2f}")):
            a = num(rows, k)
            if len(a):
                parts.append(hist_svg(a, auto_bins(a), label, fmt))
        for k, label in (("layout", "компоновка"), ("pair_kind", "вид пары"),
                         ("season", "сезонная перекраска"),
                         ("compensation_src", "источник компенсации привязки")):
            if any(k in r for r in rows):
                cnt = Counter(r.get(k, "") for r in rows)
                cells = ", ".join(f"{html.escape(str(v) or '—')} — {n}"
                                  for v, n in cnt.most_common())
                parts.append(f"<p><b>{label}:</b> {cells}</p>")

    # ——— галерея
    print("галерея...", flush=True)
    picked = pick_examples(corpora, args.examples)
    parts.append(f"""<h2>4. Примеры ({len(picked)})</h2>
<p>В каждой панели: <b>кадр A</b> (вид виртуального борта) | <b>сторона B</b>
(подложка или второй вид того же ортоплана) | <b>наложение</b>: кадр A натянут
на B шахматкой, жёлтым обведён его след. Совпадают ли дороги и контуры на
границах клеток — и есть проверка разметки глазами. Стороны показаны в
натуральном размере: кроп B крупнее кадра A, так и должно быть.</p>
<p class=note>Подпись сверху: корпус | вид пары | сезон | высота | наклон |
разница курсов | масштаб | источник компенсации | вердикт аудита.</p>""")
    gal, n_panels = [], 0
    seen_key = None
    for i, (key, r) in enumerate(picked, 1):
        if key != seen_key:
            seen_key = key
            n_here = sum(1 for k, _ in picked if k == key)
            c = corpora[key]
            gal.append(f"<h3>{html.escape(c['title'])} — {n_here} примеров</h3>"
                       f"<p class=note>{html.escape(c['purpose'])}</p>")
        f = corpora[key]["root"] / r["pair"]
        if not f.exists():
            continue
        try:
            pair = load_pair(f)
        except Exception as exc:  # noqa: BLE001
            print(f"  пропущена {f.name}: {exc}", flush=True)
            continue
        meta = pair["meta"] if isinstance(pair["meta"], dict) else json.loads(pair["meta"])
        meta = dict(meta)
        meta.setdefault("compensation_src", r.get("compensation_src", ""))
        note = note_for(key, r, meta)
        img = panel(pair, note, max_width=args.panel_width)
        n_panels += 1
        gal.append(f'<figure><img loading="lazy" src="data:image/jpeg;base64,{img}"/>'
                   f'<figcaption>{html.escape(r["pair"])} — '
                   f'{html.escape(r["scene"][:16])}</figcaption></figure>')
        if i % 25 == 0:
            print(f"  {i}/{len(picked)}", flush=True)
    parts.append("\n".join(gal))

    out = Path(args.out)
    # число примеров известно только после галереи: часть пар могла не открыться
    # считаем панели, а не элементы списка: в нём есть ещё заголовки корпусов
    parts = [x.replace("@EXAMPLES@", str(n_panels)) for x in parts]
    out.write_text(f"""<!doctype html><html lang=ru><meta charset=utf-8>
<title>Датасет open_orto — сводный отчёт</title>
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
    print(f"\nотчёт: {out} ({out.stat().st_size / 2**20:.1f} МБ), примеров {len(gal)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
