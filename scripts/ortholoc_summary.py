"""Сводка, приёмка и батарея проверок поставки OrthoLoC.

Пишет рядом с данными три вещи: `README.md` (что внутри и как этим
пользоваться), `ACCEPTANCE.md` с `baseline.json` (решающие метрики, база пяти
ядер на held-out, потолки — записываются **до** первого дообучения) и печатает
батарею проверок целостности.

Потолки считаются из измеренной ошибки разметки: она складывается с ошибкой
модели, поэтому даже идеальное ядро не покажет EPE ниже неё, а доля инлайеров
упирается в вероятность того, что сама разметка легла ближе порога.

    python scripts/ortholoc_summary.py --root ortholoc_dataset \\
        --bench eval_out/ortholoc_bench_heldout.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

#: Медиана рэлеевского распределения в единицах σ: med = σ·√(2 ln 2).
RAYLEIGH_MED = math.sqrt(2 * math.log(2))

#: Боевые виды пар этого корпуса — кадр против ортофото. `rect_ortho` из базы
#: исключён: там обе стороны на одной сетке, это контроль, а не задача.
BATTLE_KINDS = ("frame_xdop", "frame_dop")
CONTROL_KIND = "rect_ortho"

#: На чём стоит приёмка. Только `frame_xdop`: это боевой тип (чужое ортофото,
#: другая дата) **и единственный, у которого ошибка разметки измерена, а не
#: оценена**. Потолок, посчитанный из оценки сверху, был бы фикцией: у
#: `frame_dop` оценка включает ошибку самой модели, и потолок получался бы
#: выше базы почти без запаса. Пары со своим ортофото идут справочной строкой.
ACCEPT_KIND = "frame_xdop"


def ceilings(median_px: float) -> dict:
    """Потолки метрик при ошибке разметки с такой медианой."""
    sigma = median_px / RAYLEIGH_MED
    inl = {t: 1 - math.exp(-(t ** 2) / (2 * sigma ** 2)) for t in (1, 3, 5, 10)}
    return dict(sigma_px=sigma, epe=median_px, inl1=inl[1], inl3=inl[3],
                inl5=inl[5], inl10=inl[10])


def discriminable(n: int, p: float) -> float:
    """Порог различимости доли при парном сравнении на n парах."""
    return 1.96 * math.sqrt(p * (1 - p) / n) * math.sqrt(2) / 2


def num(r, k):
    try:
        return float(r[k])
    except (TypeError, ValueError, KeyError):
        return float("nan")


def stat(rows):
    if not rows:
        return {}
    g = lambda k: float(np.nanmedian([num(r, k) for r in rows]))   # noqa: E731
    success = float(np.nanmean([1.0 if (num(r, "inl5_frac") or 0) >= 0.5 else 0.0
                                for r in rows]))
    return dict(n=len(rows), epe=g("epe_med_px"), inl3=g("inl3_frac"),
                inl5=g("inl5_frac"), inl10=g("inl10_frac"), success=success,
                sec=g("sec"))


def checks(rows, root: Path, sample_crc: int = 60) -> list:
    """Батарея проверок: что можно сломать молча — то и проверяется."""
    out = []
    by_scene = defaultdict(set)
    for r in rows:
        by_scene[r["scene"]].add(r["split"])
    bad = [s for s, sp in by_scene.items() if len(sp) > 1]
    out.append(("V1", "сцена не встречается в двух сплитах",
                not bad, f"нарушений {len(bad)}" if bad else f"{len(by_scene)} сцен"))

    held = {r["scene"] for r in rows if r["split"] == "heldout"}
    other = {r["scene"] for r in rows if r["split"] != "heldout"}
    out.append(("V2", "территории приёмки не встречаются в обучении",
                not (held & other), f"{sorted(held)}"))

    files = {p.name for p in root.glob("*.npz")}
    named = {r["pair"] for r in rows}
    out.append(("V3", "строк манифеста = файлов", files == named,
                f"{len(named)} строк, {len(files)} файлов"))

    rng = np.random.default_rng(7)
    idx = rng.permutation(len(rows))[:sample_crc]
    broken = []
    for i in idx:
        p = root / rows[int(i)]["pair"]
        try:
            with zipfile.ZipFile(p) as z:
                if z.testzip() is not None:
                    broken.append(p.name)
        except Exception:  # noqa: BLE001
            broken.append(p.name)
    out.append(("V4", f"CRC читается (выборка {len(idx)})", not broken,
                "ошибок нет" if not broken else f"битых {len(broken)}"))

    for tag, split, need in (("V5", "heldout", 640), ("V6", "val", 400)):
        n = sum(1 for r in rows if r["split"] == split
                and r["pair_kind"] in BATTLE_KINDS)
        out.append((tag, f"боевых в `{split}` ≥ {need}", n >= need, f"{n}"))

    ctrl_scenes = {r["scene"] for r in rows if r["pair_kind"] == CONTROL_KIND}
    out.append(("V7", "контроль забывания покрывает все сцены",
                ctrl_scenes == set(by_scene),
                f"{len(ctrl_scenes)} из {len(by_scene)}"))

    w = sum(float(r["weight"]) for r in rows)
    ctrl = sum(float(r["weight"]) for r in rows if r["pair_kind"] == CONTROL_KIND)
    share = ctrl / w if w else 0
    out.append(("V8", "доля контрольной оси в смеси ≤ 15 %", share <= 0.15,
                f"{100*share:.1f} %"))

    no_sigma = [r for r in rows if not r["gt_sigma_px"]
                or float(r["gt_sigma_px"]) <= 0]
    out.append(("V9", "у каждой пары задана ожидаемая ошибка разметки",
                not no_sigma, f"без неё {len(no_sigma)}"))

    seasoned = [r for r in rows if r["season"]]
    out.append(("V10", "синтетических перекрасок в поставке нет",
                not seasoned, f"{len(seasoned)}"))
    return out


def plural(n: int, one: str, few: str, many: str) -> str:
    """Число с согласованным существительным: «12 831 пара», «51 сцена»."""
    tail, hund = n % 10, n % 100
    word = (one if tail == 1 and hund != 11 else
            few if 2 <= tail <= 4 and not 12 <= hund <= 14 else many)
    return f"{n:,}".replace(",", "\u2009") + f" {word}"


def readme(rows, bench, ceil, root: Path) -> str:
    kinds = Counter(r["pair_kind"] for r in rows)
    gb = sum(int(r["bytes"]) for r in rows) / 2**30
    h = np.array([num(r, "height_m") for r in rows])
    t = np.array([num(r, "tilt_deg") for r in rows])
    c = np.array([num(r, "covis_frac") for r in rows])
    hmin, hmax, hmed = np.nanmin(h), np.nanmax(h), np.nanmedian(h)
    tmin, tmax, tmed = np.nanmin(t), np.nanmax(t), np.nanmedian(t)
    cmed = np.nanmedian(c)
    battle = [r for r in rows if r["pair_kind"] in BATTLE_KINDS]
    w = sum(float(r["weight"]) for r in rows)
    lines = [
        "# ortholoc_dataset", "",
        f"Корпус из внешнего датасета **OrthoLoC**, приведённый к тому же виду,",
        f"что `openaerialmap_dataset`: тот же формат пары, те же колонки манифеста,",
        f"те же классы качества и веса. Обучаться и валидироваться можно так же —",
        f"загрузчик из `TRAINING.md` соседнего корпуса работает без изменений.", "",
        f"**{plural(len(rows), 'пара', 'пары', 'пар')}, {gb:.2f} ГБ, "
        f"{plural(len({r['scene'] for r in rows}), 'сцена', 'сцены', 'сцен')}.**", "",
        "## Состав", "",
        "| вид пары | пар | сторона B | роль |",
        "|---|---:|---|---|",
        f"| `frame_xdop` | {kinds['frame_xdop']} | ортофото **другого источника** | "
        "боевой тип: настоящий разрыв во внешнем виде |",
        f"| `frame_dop` | {kinds['frame_dop']} | своё ортофото территории | "
        "лёгкий боевой: тот же источник, но кадр наклонный |",
        f"| `rect_ortho` | {kinds['rect_ortho']} | ортофото, кадр ректифицирован | "
        "контроль забывания: обе стороны на одной сетке |", "",
        "## Съёмочный конверт", "",
        f"Высота {hmin:.0f}–{hmax:.0f} м (медиана **{hmed:.0f}**), наклон "
        f"{tmin:.0f}–{tmax:.0f}° (медиана **{tmed:.0f}°**), ко-видимость медиана "
        f"{cmed:.2f}.", "",
        "Это важнее, чем кажется: **корпуса дополняют друг друга по высоте**. У",
        "`openaerialmap_dataset` высоты 175–300 м, здесь — вдвое ниже, и низкий",
        "режим (тот самый, где кропы однородны и признаков мало) закрыт настоящей",
        "съёмкой, а не синтетикой. Плата — наклон: медиана 17° против наших 5°, а",
        "хвост доходит до 87°, то есть до почти горизонтальных кадров. Разрезы по",
        "`height_m` и `tilt_deg` в манифесте позволяют брать из корпуса только тот",
        "конверт, который нужен.", "",
        "## Сплиты", "",
        "Колонка `split`. Деление **своё, а не из датасета**: в OrthoLoC `train` и",
        "`val` — одни и те же 48 сцен, а `test_inPlace` входит в `train`. Честный",
        "held-out здесь один: сцены `test_outPlace`, не встречающиеся больше нигде.",
        "Исходная метка сохранена в `src_split`.", "",
        "| сплит | пар | боевых | сцен | назначение |",
        "|---|---:|---:|---:|---|",
    ]
    for sp, why in (("train", "обучение"), ("val", "выбор чекпоинта"),
                    ("heldout", "приёмка, расходуется один раз")):
        rs = [r for r in rows if r["split"] == sp]
        nb = sum(1 for r in rs if r["pair_kind"] in BATTLE_KINDS)
        lines.append(f"| `{sp}` | {len(rs)} | {nb} | "
                     f"{len({r['scene'] for r in rs})} | {why} |")
    lines += [
        "", "## Точность разметки", "",
        "Колонки `gt_class`, `gt_sigma_px`, `weight`.", "",
        "| класс | пар | вес | доля в смеси | откуда ошибка |",
        "|---|---:|---:|---:|---|",
    ]
    why = {"registered": "измеренное расхождение источников ортофото либо оценка сверху",
           "approx": "привязка источников на этой сцене не измерена",
           "exact": "обе стороны на одной сетке, warp — целочисленный сдвиг"}
    for cls in ("registered", "approx", "exact"):
        rs = [r for r in rows if r["gt_class"] == cls]
        if not rs:
            continue
        share = sum(float(r["weight"]) for r in rs) / w
        lines.append(f"| `{cls}` | {len(rs)} | {rs[0]['weight']} | "
                     f"{100*share:.0f} % | {why[cls]} |")
    lines += [
        "",
        "Ошибка `frame_xdop` — **измеренная**: два ортофото одной территории",
        "сводятся матчером, и сдвиг их взаимной привязки целиком уходит в",
        "разметку (`scripts/audit_ortholoc.py`). У `frame_dop` прямого измерения",
        "нет, стоит оценка сверху.", "",
        "## Чего здесь нет", "",
        "- **надира**: съёмка наклонная, медиана около 20°, а наш боевой режим —",
        "  надирный. Это чужой домен, и в смеси со своим корпусом его доля —",
        "  предмет решения, а не данность;",
        "- **своей подложки**: сторона B здесь всегда ортофото, а не спутниковая",
        "  мозаика;",
        "- **сезонной оси**: синтетические перекраски отброшены — замерено, что",
        "  разрыва они почти не создают.", "",
        "## Как смешивать со своим корпусом", "",
        "Формат пары общий, поэтому оба корпуса читаются одним `Dataset` и",
        "сэмплируются одним `WeightedRandomSampler` по колонке `weight`. Но домен",
        "разный, и доля этого корпуса в смеси — решение, а не мелочь:", "",
        "- здесь **наклонная** съёмка (медиана около 20°) против ортофото, у нас —",
        "  надир против спутниковой подложки;",
        "- разметка здесь **точнее** (около 1 px против 4 px), поэтому на общих",
        "  метриках этот корпус будет выглядеть «лучше» — сравнивать их между собой",
        "  бессмысленно, у каждого своя приёмка и свои потолки;",
        "- сплиты у корпусов независимые: смешивать можно `train` с `train`, но",
        "  **приёмку каждого проводить на своём** `heldout`.", "",
        "## Лицензия", "",
        "Исходный OrthoLoC — **CC BY-NC-SA 4.0**, то есть некоммерческое",
        "использование. Это ограничение переходит на всё, что из него собрано,",
        "включая эту поставку. Смешивая корпуса в обучении, помните: у",
        "`openaerialmap_dataset` лицензия другая.", "",
        "## Документы рядом", "",
        "- `METHODOLOGY.md` — как корпус получен: что было на входе, как строится",
        "  пара, что отброшено и откуда взялись числа точности разметки;",
        "- `ACCEPTANCE.md` и `baseline.json` — решающие метрики, база пяти ядер",
        "  на held-out, потолки и порог значимости: записаны до обучения;",
        "- `MATCHERS_METRICS.md` — пять ядер по видам пар и разрезам, с оговоркой",
        "  о том, какие разрезы читать нельзя;",
        "- `SUMMARY.html` — распределения, галерея примеров каждого вида пары и",
        "  контроль разметки глазами;",
        "- `PAIR_ANATOMY.html` — разбор одной пары: что лежит внутри `.npz`,",
        "  как выглядит каждый массив и как прочитать пару в коде;",
        "- `explore.ipynb` и `requirements.txt` — просмотрщик корпуса: загрузка,",
        "  перемешивание, случайная пара со всеми массивами и проверками;",
        "- `WORK_REPORT.html` — отчёт о работе: как корпус собирался, что измерено",
        "  и какие поправки пришлось внести по ходу;",
        "- `manifest.csv` — опись: одна строка на пару;",
        "- загрузчик, сэмплер и точки подмены GT — в `TRAINING.md` соседнего",
        "  корпуса `openaerialmap_dataset`: формат пары общий, адаптер один на оба.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="ortholoc_dataset")
    ap.add_argument("--bench", default="eval_out/ortholoc_bench_heldout.csv")
    args = ap.parse_args()

    root = Path(args.root)
    rows = list(csv.DictReader((root / "manifest.csv").open(encoding="utf-8")))
    accept = [r for r in rows if r["pair_kind"] == ACCEPT_KIND
              and r["split"] == "heldout"]
    sigma_med = float(np.median([float(r["gt_sigma_px"]) for r in accept])) if accept else 3.0
    ceil = ceilings(sigma_med)

    (root / "README.md").write_text(readme(rows, None, ceil, root), encoding="utf-8")

    bench_rows = []
    bp = Path(args.bench)
    if bp.exists():
        held = {r["pair"].replace(".npz", "") for r in rows if r["split"] == "heldout"}
        bench_rows = [r for r in csv.DictReader(bp.open(encoding="utf-8"))
                      if r["pair"] in held]

    if bench_rows:
        kind_of = {r["pair"].replace(".npz", ""): r["pair_kind"] for r in rows}
        by, by_own, by_ctrl = defaultdict(list), defaultdict(list), defaultdict(list)
        for r in bench_rows:
            kind = kind_of.get(r["pair"], "")
            (by if kind == ACCEPT_KIND else
             by_own if kind == "frame_dop" else by_ctrl)[r["matcher"]].append(r)
        stats = {m: stat(rs) for m, rs in by.items()}
        stats_own = {m: stat(rs) for m, rs in by_own.items()}
        best = min(stats.items(), key=lambda kv: kv[1]["epe"])
        n = best[1]["n"]
        thr = discriminable(n, best[1]["success"])
        lines = [
            "# Приёмка OrthoLoC: база, потолки и правила чтения метрик", "",
            "Составлено **до** первого дообучения. Числа сняты на сплите `heldout`",
            "(сцены L08, L50, L51 — их нет ни в обучении, ни в валидации).", "",
            "База стоит на парах `frame_xdop`: кадр против **чужого** ортофото.",
            "Это боевой тип корпуса и единственный, у которого ошибка разметки",
            "измерена (сдвиг привязки источников), а не оценена сверху. Пары со",
            "своим ортофото идут справочной строкой — по ним видно, сколько",
            "трудности создаёт именно смена источника.", "",
            "## Решающие метрики", "",
            "| пункт | значение |", "|---|---|",
            f"| **решающие** | `inl5` и доля успешных пар на `frame_xdop` в `heldout` |",
            f"| **справочные** | EPE, `inl10`, разрезы по наклону и ко-видимости |",
            f"| **контроль забывания** | `rect_ortho` не ниже своей базы |",
            f"| **порог значимости** | различимо от **{thr:.3f}** по доле успеха при n = {n} |",
            "", "## База: пять ядер на held-out, без дообучения", "",
            "| ядро | EPE, px | inl3 | inl5 | inl10 | успех | с/пару |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for m, s in sorted(stats.items(), key=lambda kv: kv[1]["epe"]):
            lines.append(f"| {m} | {s['epe']:.2f} | {s['inl3']:.2f} | {s['inl5']:.2f} | "
                         f"{s['inl10']:.2f} | {s['success']:.2f} | {s['sec']:.2f} |")
        if stats_own:
            lines += ["", "Справочно — те же ядра на парах со **своим** ортофото "
                      f"(`frame_dop`, n = {next(iter(stats_own.values()))['n']}):", "",
                      "| ядро | EPE, px | inl3 | inl5 | успех |", "|---|---:|---:|---:|---:|"]
            for m, s in sorted(stats_own.items(), key=lambda kv: kv[1]["epe"]):
                lines.append(f"| {m} | {s['epe']:.2f} | {s['inl3']:.2f} | "
                             f"{s['inl5']:.2f} | {s['success']:.2f} |")
            gap = (min(stats.values(), key=lambda s: s["epe"])["epe"]
                   / max(min(stats_own.values(), key=lambda s: s["epe"])["epe"], 1e-6))
            lines += ["", f"Смена источника ортофото стоит лучшему ядру **×{gap:.1f}** "
                      "по EPE — это и есть та трудность, ради которой корпус берётся."]
        lines += [
            "", "## Потолки: что достижимо на этой разметке", "",
            f"Ожидаемая ошибка разметки пар приёмки — **{sigma_med:.2f} px**:",
            "медиана измеренного сдвига привязки между источниками ортофото",
            "(`scripts/audit_ortholoc.py`). Отсюда потолки:", "",
            "| метрика | база | потолок |", "|---|---:|---:|",
            f"| EPE, px | {best[1]['epe']:.2f} | **{ceil['epe']:.2f}** |",
            f"| inl3 | {best[1]['inl3']:.2f} | {ceil['inl3']:.2f} |",
            f"| inl5 | {best[1]['inl5']:.2f} | **{ceil['inl5']:.2f}** |",
            f"| inl10 | {best[1]['inl10']:.2f} | {ceil['inl10']:.2f} |",
            "",
            "Прогресс считать нормированно: `(стало − база) / (потолок − база)`.",
            "", f"Лучшее ядро без дообучения — **{best[0]}**.", "",
            "## Что запрещено", "",
            "- трогать `heldout` до приёмки: он расходуется один раз;",
            "- сравнивать с базой, снятой на другой выборке;",
            "- менять решающую метрику после того, как увидели результат.", "",
            f"Сырьё: `{args.bench}`.",
        ]
        (root / "ACCEPTANCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (root / "baseline.json").write_text(json.dumps(
            dict(dataset="ortholoc", n_heldout_battle=n, sigma_px=sigma_med,
                 ceilings=ceil, threshold=thr,
                 baseline={m: s for m, s in stats.items()}),
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"приёмка: база по {n} парам, лучшее ядро {best[0]}")
    else:
        print("бенчмарка нет — ACCEPTANCE.md не обновлён")

    print("\nбатарея проверок:")
    ok_all = True
    for tag, name, ok, note in checks(rows, root):
        ok_all &= ok
        print(f"  {tag} {'✔' if ok else '✗'} {name:52} {note}")
    print("итог:", "все проверки пройдены" if ok_all else "ЕСТЬ НАРУШЕНИЯ")
    print(f"сводка: {root / 'README.md'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
