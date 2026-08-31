"""Файл приёмки: база, потолки и правила чтения метрик — до первого обучения.

Зачем это пишется заранее. У корпуса пять метрик и четыре разреза — больше
сорока чисел; после любого обучения какое-нибудь из них вырастет само, и его
будет соблазнительно назвать результатом. И отдельно: движение внутри
статистического шума легко принять за прибавку. Оба способа обмануть себя
закрываются тем, что решающая метрика, потолок и порог значимости названы
**до** запуска и потом не меняются.

Потолки не выдуманы: у самой разметки есть измеренное расхождение с независимым
арбитром. Если считать ошибку разметки изотропной, её модуль распределён по
Рэлею, и из медианы выводится всё остальное — какую долю инлайеров покажет даже
идеальная модель.

    python open_orto/scripts/acceptance.py --bench open_orto/work/bench_heldout.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

MATCHERS = ["roma", "minima_roma", "minima_loftr", "loftr", "romav2"]
TITLE = {"roma": "RoMa v1 (ванильная)", "minima_roma": "MINIMA-RoMa",
         "minima_loftr": "MINIMA-LoFTR", "loftr": "LoFTR", "romav2": "RoMa v2"}
#: Медиана модуля рэлеевского вектора: med = sigma * sqrt(2 ln 2).
RAYLEIGH_MED = math.sqrt(2 * math.log(2))


def ceilings(median_px: float) -> dict:
    """Потолки метрик при ошибке разметки с такой медианой.

    Ошибка разметки складывается с ошибкой модели, поэтому даже идеальная
    модель не покажет EPE ниже медианы расхождения, а доля инлайеров упрётся
    в вероятность того, что сама разметка легла ближе порога.
    """
    sigma = median_px / RAYLEIGH_MED
    inl = {t: 1 - math.exp(-(t ** 2) / (2 * sigma ** 2)) for t in (1, 3, 5, 10)}
    return dict(sigma_px=sigma, epe=median_px,
                inl1=inl[1], inl3=inl[3], inl5=inl[5], inl10=inl[10])


def num(r, k):
    try:
        return float(r[k])
    except (TypeError, ValueError, KeyError):
        return float("nan")


def stat(rows):
    if not rows:
        return {}
    g = lambda k: float(np.nanmedian([num(r, k) for r in rows]))
    success = float(np.nanmean([1.0 if (num(r, "inl5_frac") or 0) >= 0.5 else 0.0
                                for r in rows]))
    return dict(n=len(rows), epe=g("epe_med_px"), inl3=g("inl3_frac"),
                inl5=g("inl5_frac"), inl10=g("inl10_frac"), success=success,
                sec=g("sec"))


def discriminable(n: int, p: float) -> float:
    """Порог различимости доли при парном сравнении на n парах."""
    se = math.sqrt(p * (1 - p) / n)
    return 1.96 * se * math.sqrt(2) / 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench", default="open_orto/work/bench_heldout.csv")
    ap.add_argument("--manifest", default="openaerialmap_dataset/manifest.csv")
    ap.add_argument("--audit", default="open_orto/work/audit_all.csv")
    ap.add_argument("--out-md", default="openaerialmap_dataset/ACCEPTANCE.md")
    ap.add_argument("--out-json", default="openaerialmap_dataset/baseline.json")
    args = ap.parse_args()

    man = list(csv.DictReader(Path(args.manifest).open(encoding="utf-8")))
    ho = [r for r in man if r["split"] == "heldout"]
    ho_scenes = {r["scene"] for r in ho}
    combat = [r for r in ho if r["pair_kind"] != "same_source"]

    # фактическое расхождение с арбитром — на площадках именно этого сплита
    diffs = []
    for r in csv.DictReader(Path(args.audit).open(encoding="utf-8")):
        if r["scene"] in ho_scenes and r.get("gt_diff_med") and \
                r["вердикт"] == "разметка подтверждена":
            try:
                diffs.append(float(r["gt_diff_med"]))
            except ValueError:
                pass
    gt_med = float(np.median(diffs)) if diffs else 4.13
    ceil = ceilings(gt_med)
    print(f"held-out: {len(ho)} пар, боевых {len(combat)}, площадок {len(ho_scenes)}")
    print(f"расхождение с арбитром на них: медиана {gt_med:.2f} px (n={len(diffs)})")
    print(f"потолки: EPE {ceil['epe']:.2f}, inl3 {ceil['inl3']:.3f}, "
          f"inl5 {ceil['inl5']:.3f}, inl10 {ceil['inl10']:.3f}")

    rows = list(csv.DictReader(Path(args.bench).open(encoding="utf-8")))
    by = defaultdict(list)
    for r in rows:
        # боевые — строго orto_basemap: кросс-датные пары в базу не входят,
        # у них другой источник стороны B и своя точность разметки
        by[(r["matcher"], r["pair_kind"])].append(r)
    base = {m: stat(by[(m, "orto_basemap")]) for m in MATCHERS}
    ctrl = {m: stat(by[(m, "same_source")]) for m in MATCHERS}
    xdate = {m: stat(by[(m, "cross_date")]) for m in MATCHERS}
    ref = base.get("roma") or {}
    p = ref.get("success", 0.39)
    n = ref.get("n", len(combat))
    delta = discriminable(int(n), p)

    def table(stats, keys, heads, fmts):
        out = ["| ядро | " + " | ".join(heads) + " |", "|---" * (len(heads) + 1) + "|"]
        for m in MATCHERS:
            d = stats.get(m) or {}
            cells = [fmt.format(d[k]) if k in d else "—" for k, fmt in zip(keys, fmts)]
            out.append(f"| {TITLE[m]} | " + " | ".join(cells) + " |")
        return out

    md = [
        "# Приёмка: база, потолки и правила чтения метрик",
        "",
        f"Составлено {date.today().isoformat()} **до** первого дообучения. Числа "
        "получены на сплите `heldout`, который до приёмки больше не трогается.",
        "",
        "## Решающие метрики",
        "",
        "| пункт | значение |",
        "|---|---|",
        "| **решающие** | `inl5` и «успех» на боевых парах (`pair_kind = orto_basemap`) сплита `heldout` |",
        "| **справочные** | EPE, `inl10`, разрезы по высоте, наклону, разнице курсов |",
        f"| **не использовать** | `inl3`: его потолок {ceil['inl3']:.2f} — он меряет преимущественно шум разметки |",
        "| **контроль забывания** | `same_source` не ниже `inl3` 0.99 |",
        "| **инвариант** | ноль ложных срабатываний на стенде из 30 кейсов — не смягчается ни при какой прибавке |",
        f"| **порог значимости** | парное сравнение на одних парах: различимо от **{delta:.3f}** по доле успеха при n = {int(n)} |",
        "",
        "## База: пять ядер на held-out, без дообучения",
        "",
        f"Боевые пары, n = {int(n)}.",
        "",
    ]
    md += table(base, ["epe", "inl3", "inl5", "inl10", "success", "sec"],
                ["EPE, px", "inl3", "inl5", "inl10", "успех", "с/пару"],
                ["{:.2f}", "{:.2f}", "{:.2f}", "{:.2f}", "{:.2f}", "{:.2f}"])
    md += ["", "Контрольная ось `same_source` того же сплита "
           f"(n = {ctrl.get('roma', {}).get('n', 0)}):", ""]
    md += table(ctrl, ["epe", "inl3", "inl5", "success"],
                ["EPE, px", "inl3", "inl5", "успех"],
                ["{:.2f}", "{:.2f}", "{:.2f}", "{:.2f}"])
    if xdate.get("roma", {}).get("n"):
        md += ["", f"Кросс-датные пары `heldout` (n = {xdate['roma']['n']}) — "
               "отдельная ось: обе стороны ортофото, но снятые в разные даты. "
               "В базу не входят, меряются справочно:", ""]
        md += table(xdate, ["epe", "inl3", "inl5", "success"],
                    ["EPE, px", "inl3", "inl5", "успех"],
                    ["{:.2f}", "{:.2f}", "{:.2f}", "{:.2f}"])

    md += [
        "",
        "## Потолки: что вообще достижимо на этой разметке",
        "",
        f"Расхождение разметки с независимым арбитром на площадках `heldout` — "
        f"**{gt_med:.2f} px** (медиана по {len(diffs)} подтверждённым площадкам). "
        "Если считать ошибку разметки изотропной, её модуль распределён по Рэлею "
        f"с параметром σ = {ceil['sigma_px']:.2f} px, и отсюда следуют потолки: "
        "даже идеальная модель не покажет больше.",
        "",
        "| метрика | база (RoMa v1) | потолок | остаток хода |",
        "|---|---|---|---|",
    ]
    for key, label, better_low in (("epe", "EPE, px", True), ("inl3", "inl3", False),
                                   ("inl5", "inl5", False), ("inl10", "inl10", False)):
        b, c = ref.get(key, float("nan")), ceil[key]
        gap = (f"−{100*(b-c)/b:.0f} %" if better_low and b else
               f"×{c/b:.1f}" if b else "—")
        md.append(f"| {label} | {b:.2f} | **{c:.2f}** | {gap} |")
    md.append(f"| успех | {p:.2f} | ~1.0 | ×{1/p:.1f} |")

    true_err = math.sqrt(max(ref.get("epe", 0) ** 2 - gt_med ** 2, 0))
    md += [
        "",
        f"Отсюда же оценка истинной ошибки модели: измеренное складывается с "
        f"ошибкой разметки, и при независимости `{ref.get('epe', 0):.2f}² = x² + "
        f"{gt_med:.2f}²`, откуда **x ≈ {true_err:.2f} px**. Модель ближе к пределу "
        "корпуса, чем выглядит по сырому числу.",
        "",
        "## Как считать прогресс",
        "",
        "Нормированно: `(стало − база) / (потолок − база)`. Без этой шкалы числа "
        "вводят в заблуждение в обе стороны — «EPE упала с "
        f"{ref.get('epe', 0):.2f} до {ref.get('epe', 0) - 1:.2f}» звучит скромно, "
        "а это заметная доля всего достижимого; рост `inl5` на пару сотых выглядит "
        "движением, хотя лежит внутри порога различимости.",
        "",
        "| `inl5` стало | пройдено доступного хода |",
        "|---|---|",
    ]
    b5, c5 = ref.get("inl5", 0.0), ceil["inl5"]
    for v in (b5 + 0.04, b5 + 0.07, b5 + 0.12, b5 + 0.17, b5 + 0.22):
        if v < c5:
            md.append(f"| {v:.2f} | {100*(v-b5)/(c5-b5):.0f} % |")
    md += [
        "",
        "## Что запрещено",
        "",
        "- **Трогать `heldout` до приёмки.** Он расходуется один раз; выбор "
        "чекпоинта — по `val`.",
        "- **Оптимизировать `inl3`** на боевых парах: его потолок "
        f"{ceil['inl3']:.2f}.",
        "- **Менять решающую метрику после того, как увидели результат.**",
        "- **Сравнивать с базой, снятой на другой выборке.** База в этом файле "
        "снята ровно на `heldout`.",
        "",
        f"Сырьё: `{args.bench}`. Состав сплитов: `splits.json`. "
        "Методика корпуса: `METHODOLOGY.md`.",
        "",
    ]
    Path(args.out_md).write_text("\n".join(md) + "\n", encoding="utf-8")

    Path(args.out_json).write_text(json.dumps({
        "date": date.today().isoformat(),
        "split": "heldout",
        "combat_pairs": int(n),
        "gt_median_px": round(gt_med, 3),
        "gt_scenes": len(diffs),
        "ceilings": {k: round(v, 4) for k, v in ceil.items()},
        "decisive_metrics": ["inl5", "success"],
        "do_not_use": ["inl3"],
        "discriminable_success": round(delta, 4),
        "baseline": {m: {k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in base[m].items()} for m in MATCHERS if base[m]},
        "control_same_source": {m: {k: (round(v, 4) if isinstance(v, float) else v)
                                    for k, v in ctrl[m].items()} for m in MATCHERS if ctrl[m]},
        "cross_date": {m: {k: (round(v, 4) if isinstance(v, float) else v)
                           for k, v in xdate[m].items()} for m in MATCHERS if xdate[m]},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nприёмка: {args.out_md}\nбаза: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
