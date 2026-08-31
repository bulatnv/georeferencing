"""Разметка пар по качеству разметки и вес в обучающей смеси.

Корпус неоднороден: у части пар разметка выведена аналитически, у части
измерена и подтверждена независимым арбитром, а у части опирается на константу
площадки вместо замера по месту. Загрузчику это надо знать, иначе он будет
учить одинаково на всём — и на том, что заведомо точнее его собственной цели.

Три класса:

- `exact` — `same_source`: обе стороны из одного растра, соответствие выводится
  аналитически, ошибка порядка сотых пикселя;
- `registered` — боевая пара с подтверждённой привязкой и компенсацией,
  измеренной по месту;
- `approx` — остальное: компенсация взята константой площадки либо арбитр не
  подтвердил площадку.

Веса решают отдельную проблему: `same_source` — половина корпуса, и линия RoMa
решает его полностью (inl3 = 1.00). Половина шагов обучения уходила бы в нулевой
градиент. Понижение веса оставляет ось как контроль забывания, не делая её
основным сигналом.

    python open_orto/scripts/enrich_manifest.py
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

#: Вердикты аудита, подтверждающие привязку площадки.
CONFIRMING = {
    "разметка подтверждена",
    "подтверждена (гейт пересмотрен)",
    "подтверждена (повтор; прицельный замер)",
    # кросс-датные пары: обе стороны — ортофото, привязка между вылетами
    # измерена и подтверждена арбитром. Их разметка точнее большинства боевых
    # пар (контроль остатка 0.94 px против 2.4–6.5 у пар с подложкой)
    "кросс-датная (аудит подтвердил)",
}

#: Ошибка разметки по классам, px. Для `registered` берётся измеренное на
#: площадке расхождение с арбитром, а это значение — запасное.
SIGMA_EXACT = 0.02
#: У кросс-датных пар обе стороны ортофото: расхождение измерено арбитром
#: по их собственным площадкам, запасное значение — измеренный контроль.
SIGMA_CROSS_DATE = 1.0
SIGMA_REGISTERED = 4.13
SIGMA_APPROX = 8.0

#: Веса в смеси. Цель — довести долю `same_source` до 15 % и ниже, не удаляя
#: данные: она остаётся контролем забывания и якорем геометрии.
WEIGHTS = {"exact": 0.15, "registered": 1.0, "approx": 0.3}

#: Пары с площадок, где «ортофото» на самом деле карта рельефа (визуализация
#: теневой отмывкой). Геометрия у них верна, но содержание — не фотоснимок:
#: матчер учился бы сопоставлять отмывку с настоящей съёмкой, а это другая
#: задача. Вес ноль вместо удаления: материал редкий и как отдельная
#: кросс-модальная ось может пригодиться.
RELIEF_WEIGHT = 0.0
RELIEF_SLICE = "relief_xmodal"


def classify(row) -> str:
    if row["pair_kind"] == "same_source":
        return "exact"
    verdict = row.get("вердикт", "")
    # пилотные площадки аудировались отдельно и подтверждены все три, но в
    # общей аудитной таблице их нет: без этой оговорки восьмая часть боевого
    # корпуса уехала бы в пониженный вес без причины
    confirmed = verdict in CONFIRMING or row.get("corpus") == "pilot"
    by_place = row.get("compensation_src") != "global"
    return "registered" if confirmed and by_place else "approx"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default="openaerialmap_dataset/manifest.csv")
    ap.add_argument("--audit", default="open_orto/work/audit_all.csv")
    ap.add_argument("--modality", default="openaerialmap_dataset/scene_modality.csv")
    ap.add_argument("--audit-xdate", default="open_orto/work/audit_xdate.csv",
                    help="аудит кросс-датных пар: у них своё расхождение, "
                         "измеренное на них же, а не на парах с подложкой")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    man = list(csv.DictReader(Path(args.manifest).open(encoding="utf-8")))
    modality = {}
    mp = Path(args.modality)
    if mp.exists():
        modality = {r["scene"]: r["modality"] for r in csv.DictReader(mp.open(encoding="utf-8"))}
    sigma_by_scene = {}
    ap_path = Path(args.audit)
    if ap_path.exists():
        for r in csv.DictReader(ap_path.open(encoding="utf-8")):
            if r.get("gt_diff_med"):
                try:
                    sigma_by_scene[r["scene"]] = float(r["gt_diff_med"])
                except ValueError:
                    pass

    sigma_xdate = {}
    xp = Path(args.audit_xdate)
    if xp.exists():
        for r in csv.DictReader(xp.open(encoding="utf-8")):
            if r.get("gt_diff_med"):
                try:
                    sigma_xdate[r["scene"]] = float(r["gt_diff_med"])
                except ValueError:
                    pass

    tally, weight_sum = Counter(), Counter()
    for r in man:
        cls = classify(r)
        if cls == "exact":
            sigma = SIGMA_EXACT
        elif cls == "registered":
            if r["pair_kind"] == "cross_date":
                # у площадки может быть измерено расхождение и на парах с
                # подложкой — но к кросс-датным парам оно не относится
                sigma = sigma_xdate.get(r["scene"], SIGMA_CROSS_DATE)
            else:
                sigma = sigma_by_scene.get(r["scene"], SIGMA_REGISTERED)
        else:
            sigma = SIGMA_APPROX
        mod = modality.get(r["scene"], "photo")
        r["modality"] = mod
        r["gt_class"] = cls
        r["gt_sigma_px"] = f"{sigma:.3f}"
        w = RELIEF_WEIGHT if mod == "relief" else WEIGHTS[cls]
        r["weight"] = f"{w:.2f}"
        r["slice"] = RELIEF_SLICE if mod == "relief" else r.get("slice", "")
        tally[cls] += 1
        weight_sum[cls] += w

    total_w = sum(weight_sum.values())
    print(f"{'класс':12} {'пар':>6} {'доля сейчас':>12} {'вес':>5} {'доля в смеси':>13}")
    for cls in ("registered", "approx", "exact"):
        n = tally[cls]
        print(f"{cls:12} {n:6} {100*n/len(man):11.0f}% {WEIGHTS[cls]:5.2f} "
              f"{100*weight_sum[cls]/total_w:12.0f}%")
    ss_share = 100 * weight_sum["exact"] / total_w
    print(f"\nдоля same_source в смеси: {ss_share:.0f}% "
          f"(было {100*tally['exact']/len(man):.0f}% по числу пар)")

    # из чего состоит approx — полезно знать, прежде чем понижать вес
    rel = [r for r in man if r["modality"] == "relief"]
    if rel:
        print(f"площадки-карты рельефа: {len({r['scene'] for r in rel})}, "
              f"их пар {len(rel)} — вес {RELIEF_WEIGHT}, помечены `{RELIEF_SLICE}`")
    pan = [r for r in man if r["modality"] == "pan"]
    if pan:
        print(f"панхромные площадки: {len({r['scene'] for r in pan})}, их пар {len(pan)} "
              f"— остаются в смеси: это снимки, просто без цвета")

    approx = [r for r in man if r["gt_class"] == "approx"]
    print(f"состав approx ({len(approx)}): "
          f"компенсация global — {sum(1 for r in approx if r['compensation_src'] == 'global')}, "
          f"вердикт не подтверждает — "
          f"{sum(1 for r in approx if r['вердикт'] not in CONFIRMING and r['corpus'] != 'pilot')}")

    if args.dry_run:
        return 0

    head = list(man[0].keys())
    with Path(args.manifest).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=head)
        w.writeheader()
        w.writerows(man)
    print(f"\nманифест: {len(man)} строк, {len(head)} колонок")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
