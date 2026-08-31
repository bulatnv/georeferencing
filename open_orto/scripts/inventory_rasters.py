"""Опись исходных ортофотопланов: что пошло в датасет, что нет и почему.

Датасет собирался в несколько прогонов, и судьба каждого растра разбросана по
манифестам, отказным спискам и результатам аудита. Здесь всё сводится в одну
таблицу — по строке на растр, — чтобы при следующем пополнении набора не
пришлось выяснять заново, почему конкретный ортоплан не дал ни одной пары.

Судьба растра складывается из двух независимых веток: пары «сам на себя» ему
доступны всегда, а пары с подложкой — только если удалось измерить привязку.
Поэтому растр может быть использован наполовину, и в описи это видно.

    python open_orto/scripts/inventory_rasters.py --data-dir E:/open_ortophoto_data
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

#: Отбор по паспорту — те же пороги, что в конвейере.
MIN_KM2 = 0.5
MAX_RES = 0.1

#: Отказные списки прогонов, от раннего к позднему: поздний уточняет ранний
#: (пороги гейта по ходу пересматривались, и часть отказов была снята).
REJECT_LOGS = [
    "open_orto/work/basemap/rejected_step_by_area.csv",
    "open_orto/work/basemap/rejected.csv",
    "open_orto/work/basemap_G/rejected.csv",
    "open_orto/work/basemap_N/rejected.csv",
]

FIELDS = ["name", "статус", "причина", "пар_ss", "пар_base", "пар_карантин",
          "вердикт_аудита", "km2", "res", "bands", "crs", "lon", "lat"]


def load(p):
    f = Path(p)
    return list(csv.DictReader(f.open(encoding="utf-8"))) if f.exists() else []


def passport_index(scans):
    """Паспорта из имеющихся сканов: имя → строка скана."""
    out = {}
    for s in scans:
        for r in load(s):
            out[r["name"]] = r
    return out


def paper_reason(p) -> str | None:
    """Почему растр не годен по паспорту (None — годен)."""
    if not p:
        return "паспорт не прочитан (файл не открывается)"
    km2, res = float(p["km2"]), float(p["res"])
    if km2 <= 0 or res <= 0:
        return "неметрический CRS: единицы сетки не метры"
    if km2 < MIN_KM2:
        return f"площадь {km2:.2f} км² меньше {MIN_KM2}"
    if res > MAX_RES:
        return f"разрешение {res:.2f} м/пкс грубее {MAX_RES}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default="E:/open_ortophoto_data")
    ap.add_argument("--scans", nargs="*", default=["open_orto/work/rasters_scan.csv",
                                                   "open_orto/work/rasters_scan_new.csv"])
    ap.add_argument("--out-csv", default="openaerialmap_dataset/rasters_inventory.csv")
    ap.add_argument("--out-md", default="openaerialmap_dataset/RASTERS.md")
    args = ap.parse_args()

    files = sorted(p.stem for p in Path(args.data_dir).glob("*.tif"))
    paper = passport_index(args.scans)

    pairs = defaultdict(Counter)
    for corpus, path in (("ss", "open_orto/dataset_ss"),
                         ("base", "open_orto/dataset_base"),
                         ("base", "open_orto/dataset"),
                         ("quar", "open_orto/dataset_base_quarantine")):
        for r in load(Path(path) / "manifest.csv"):
            pairs[r["scene"]][corpus] += 1

    audit = {r["scene"]: r["вердикт"] for r in load("open_orto/work/audit_all.csv")}
    rejects = {}
    for p in REJECT_LOGS:
        for r in load(p):
            rejects[r["scene"]] = r["причина"]

    rows, tally = [], Counter()
    for name in files:
        p = paper.get(name)
        n_ss = pairs[name]["ss"]
        n_base = pairs[name]["base"]
        n_quar = pairs[name]["quar"]
        reason_paper = paper_reason(p)

        if n_ss or n_base:
            if n_base and n_ss:
                status, reason = "использован полностью", ""
            elif n_ss:
                status = "использован частично"
                if n_quar:
                    # пары с подложкой построены, но арбитр их забраковал —
                    # это другая судьба, чем «привязку не удалось измерить»
                    reason = f"подложка: {n_quar} пар забраковано аудитом (привязка сдвинута)"
                elif name in rejects:
                    reason = "подложка: " + rejects[name]
                else:
                    reason = "подложка: пар не вышло"
            else:
                status, reason = "использован (только с подложкой)", ""
        elif n_quar:
            status = "забракован аудитом"
            reason = f"привязка сдвинута ({audit.get(name, '')})".strip()
        elif reason_paper:
            status, reason = "не годен по паспорту", reason_paper
        elif name in rejects:
            status, reason = "не использован", "привязка: " + rejects[name]
        else:
            status, reason = "не использован", "годен, но ни одной пары не вышло"

        tally[status] += 1
        rows.append(dict(name=name, статус=status, причина=reason,
                         пар_ss=n_ss, пар_base=n_base, пар_карантин=n_quar,
                         вердикт_аудита=audit.get(name, ""),
                         km2=p["km2"] if p else "", res=p["res"] if p else "",
                         bands=p["bands"] if p else "", crs=p["crs"] if p else "",
                         lon=p.get("lon", "") if p else "",
                         lat=p.get("lat", "") if p else ""))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    # ——— отчёт
    used = [r for r in rows if r["статус"].startswith("использован")]
    total_pairs = sum(int(r["пар_ss"]) + int(r["пар_base"]) for r in rows)
    md = [
        "# Опись исходных ортофотопланов",
        "",
        f"Всего в каталоге **{len(rows)}** растров. В датасет вошли пары с "
        f"**{len(used)}** из них ({100*len(used)/max(len(rows),1):.0f} %), "
        f"суммарно {total_pairs} пар.",
        "",
        "Судьба растра складывается из двух независимых веток: пары «ортоплан",
        "сам на себя» доступны любому годному растру, а пары с подложкой —",
        "только тем, у кого удалось измерить привязку. Поэтому растр может быть",
        "использован наполовину.",
        "",
        "| статус | растров | что это значит |",
        "|---|---|---|",
    ]
    meaning = {
        "использован полностью": "дал пары обоих видов — и с подложкой, и «сам на себя»",
        "использован частично": "дал только пары «сам на себя»: с подложкой пары либо не построились, либо забракованы аудитом",
        "использован (только с подложкой)": "пары с подложкой есть, «сам на себя» не вышли",
        "забракован аудитом": "пары построены, но арбитр нашёл сдвиг привязки — они в карантине",
        "не годен по паспорту": "отсеян до обработки: CRS, площадь или разрешение",
        "не использован": "годен по паспорту, но пар не дал",
    }
    for st, n in tally.most_common():
        md.append(f"| {st} | {n} | {meaning.get(st, '')} |")

    md += ["", "## Почему растры не пошли в дело", ""]
    for st in ("не годен по паспорту", "не использован", "забракован аудитом"):
        sub = [r for r in rows if r["статус"] == st]
        if not sub:
            continue
        md += [f"### {st} — {len(sub)}", ""]
        # причины схлопываем по виду, числа внутри причины у каждого свои
        c = Counter(re.sub(r"[0-9]+([.,][0-9]+)?", "N", r["причина"]) for r in sub)
        md += ["| причина | растров |", "|---|---|"]
        for reason, n in c.most_common():
            md.append(f"| {reason or '—'} | {n} |")
        md.append("")

    part = [r for r in rows if r["статус"] == "использован частично"]
    if part:
        md += [
            "## Использованы наполовину",
            "",
            f"{len(part)} растров дали пары «сам на себя», но не дали пар с",
            "подложкой. Причины — те же, что у полного отказа гейта: привязку не",
            "на чем измерить (кросс-сезонная съёмка, сплошной лес или поле,",
            "перестроенная территория) либо рабочая зона слишком мала.",
            "",
        ]

    # что имеет смысл прогнать заново, если пороги изменятся
    retry = [r for r in rows if r["статус"] in ("использован частично", "не использован")
             and "привязк" in r["причина"]]
    km2 = sum(float(r["km2"] or 0) for r in retry)
    near = [r for r in retry if "узлов привязки" in r["причина"]]
    md += [
        "## Резерв: кого прогнать заново при смене порогов",
        "",
        f"{len(retry)} растров ({km2:.0f} км²) не дали пар с подложкой из-за "
        f"того, что привязку не удалось измерить. Из них **{len(near)} были "
        f"близки к порогу**: валидные узлы нашлись, но их не набралось пять.",
        "",
        "Это не приговор: пороги гейта пересматривались дважды, и каждый раз",
        "часть отказов снималась — сначала шаг сетки стали считать по рабочей",
        "зоне, а не по габариту растра (вернулось 128 площадок из 256), затем",
        "окно замера подогнали под размер площадки (ещё 57 из 128). Если",
        "появится третья такая правка, начинать надо с этих строк описи.",
        "",
        "## Как читать таблицу",
        "",
        "Полная опись — `rasters_inventory.csv`, по строке на растр:",
        "",
        "| колонка | смысл |",
        "|---|---|",
        "| `статус`, `причина` | судьба растра и почему |",
        "| `пар_ss`, `пар_base` | сколько пар каждого вида он дал |",
        "| `пар_карантин` | сколько его пар забраковано аудитом |",
        "| `вердикт_аудита` | что сказал независимый арбитр о его привязке |",
        "| `km2`, `res`, `bands`, `crs` | паспорт растра |",
        "| `lon`, `lat` | центр в WGS84 — по нему ищутся дубли при пополнении набора |",
        "",
        "## Зачем это хранится",
        "",
        "При следующем пополнении набора опись отвечает на два вопроса сразу:",
        "не обрабатывался ли этот растр раньше (сверка по `lon`/`lat`, а не по",
        "имени — имена меняются) и не отбраковывался ли он уже, и по какой",
        "причине. Часть отказов снималась пересмотром порогов: если пороги",
        "меняются снова, по этой таблице видно, кого имеет смысл прогнать",
        "заново.",
        "",
        f"Методика сборки целиком — `METHODOLOGY.md`.",
    ]
    Path(args.out_md).write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"растров: {len(rows)}")
    for st, n in tally.most_common():
        print(f"  {st:34} {n:5}")
    print(f"\nопись: {out_csv}\nотчёт: {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
