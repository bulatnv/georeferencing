"""Регрессия набора: «изменилось ли поведение» одной командой.

Этап F из [EVAL_PLAN.md](../docs/EVAL_PLAN.md), фаза 0 из
[ROADMAP.md](../docs/ROADMAP.md). Отвечает ровно на один вопрос: **не сломали ли
мы то, что работало**, — и отвечает поимённо, а не сводным числом.

    python scripts/regress.py                       # прогнать и сверить с золотом
    python scripts/regress.py --from-csv eval_out/loftr.csv   # сверить готовую таблицу
    python scripts/regress.py --freeze               # ЗАМОРОЗИТЬ текущее поведение

Ключевое устройство: **золото хранит конфигурацию прогона**, и по умолчанию
скрипт сам запускает ``eval_dataset.py`` с этой конфигурацией. Собрать сравнение
«прогон с радиусом 1.5 км против золота с радиусом 2 км» нельзя не потому, что
это запрещено, а потому, что параметры берутся из золота, а не из памяти автора.
Если таблица приносится снаружи (``--from-csv``, например для A/B другого
матчера), конфигурация читается из её файла-спутника и расхождение объявляется
явно — вердикт при этом не выдаётся вовсе.

Что считается регрессией — см. :mod:`aero_geoloc.regression`. Коротко: падение
класса исхода кейса, любое ложное срабатывание, авария кейса. Рост ошибки внутри
класса — предупреждение, а не провал: RANSAC недетерминирован.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aero_geoloc.regression import (  # noqa: E402
    OUTCOME_RU,
    compare,
    freeze,
    load_golden,
    save_golden,
)

DEFAULT_GOLDEN = "datasets/golden.yaml"
HEADER = """# Золотой набор: ЗАМОРОЖЕННОЕ поведение оценки на test_images/.
#
# Не редактировать руками ради того, чтобы прогон «прошёл». Файл существует ровно
# затем, чтобы изменение поведения нельзя было не заметить: если прогон разошёлся
# с золотом — либо чинить код, либо осознанно перезаморозить (--freeze) с записью
# причины в docs/JOURNAL.md.
#
# outcome — класс исхода (aero_geoloc/regression.py), именно он даёт вердикт.
# error_m / true_cell_rank / n_inliers / ncc — справочные: по ним видно, ЧТО
# изменилось, но сами по себе они шумят и сборку не рушат.
"""


def _read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _read_config(csv_path: Path) -> dict:
    side = csv_path.with_suffix(".config.json")
    if not side.exists():
        return {}
    with open(side, encoding="utf-8") as fh:
        return json.load(fh)


def _run_eval(config: dict, out_csv: Path, extra: list[str]) -> int:
    """Запустить оценку с конфигурацией ИЗ ЗОЛОТА, а не из головы."""
    argv = [sys.executable, str(Path(__file__).with_name("eval_dataset.py")),
            "--out", str(out_csv), "--out-dir", str(out_csv.parent)]
    flags = {
        "manifest": "--manifest", "matcher": "--matcher", "radius_km": "--radius-km",
        "cell_px": "--cell-px", "overlap": "--overlap", "pca_dim": "--pca-dim",
        "top_k": "--top-k", "min_inliers": "--min-inliers",
        "rotation_step": "--rotation-step", "correct_m": "--correct-m",
        "manual_tol_frac": "--manual-tol-frac", "offset_km": "--offset-km",
        "sigma_m": "--sigma-m",
    }
    for key, flag in flags.items():
        if config.get(key) is not None:
            argv += [flag, str(config[key])]
    argv += extra
    print("прогон:", " ".join(argv[1:]), flush=True)
    return subprocess.call(argv)


def _print_report(report, golden, rows) -> None:
    width = 96
    print(f"\n{'='*width}\nРЕГРЕССИЯ: {len(rows)} кейсов в прогоне, "
          f"{len(golden.cases)} в золоте\n{'='*width}")
    if report.config_diff:
        print("КОНФИГУРАЦИЯ РАЗОШЛАСЬ — сравнение недействительно:")
        for line in report.config_diff:
            print(f"  {line}")
        print()

    marks = {"регрессия": "✗", "ухудшение": "!", "улучшение": "+",
             "новый": "?", "пропал": "?", "ok": " "}
    for v in report.verdicts:
        if v.severity == "ok":
            continue
        was = OUTCOME_RU.get(v.was, v.was or "—")
        now = OUTCOME_RU.get(v.now, v.now or "—")
        line = f"{was} → {now}"
        print(f" {marks[v.severity]} {v.name:<16}{v.severity:<12}{line}")
        if v.detail and v.detail != line:
            print(f"   {'':<16}{'':<12}{v.detail}")
    unchanged = sum(1 for v in report.verdicts if v.severity == "ok")
    print("-" * width)
    print(f"без изменений: {unchanged}   улучшений: {len(report.improvements)}   "
          f"предупреждений: {len(report.warnings)}   РЕГРЕССИЙ: {len(report.failures)}")
    print("-" * width)
    if report.passed:
        print("ВЕРДИКТ: регрессий нет.")
    elif report.config_diff:
        print("ВЕРДИКТ: не выдан — прогон и золото сняты при разных параметрах.")
    else:
        print("ВЕРДИКТ: РЕГРЕССИЯ — " + ", ".join(v.name for v in report.failures))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--golden", default=DEFAULT_GOLDEN)
    parser.add_argument("--from-csv", default="",
                        help="сверить готовую таблицу вместо нового прогона "
                             "(конфигурация читается из её файла .config.json)")
    parser.add_argument("--freeze", action="store_true",
                        help="ЗАМОРОЗИТЬ поведение прогона как новое золото")
    parser.add_argument("--note", default="", help="зачем перезаморожено (пишется в золото)")
    parser.add_argument("--out", default="eval_out/regress.csv",
                        help="куда класть таблицу собственного прогона")
    parser.add_argument("--slack-m", type=float, default=10.0,
                        help="полоса шума по ошибке, м (RANSAC недетерминирован)")
    parser.add_argument("--slack-frac", type=float, default=0.5,
                        help="полоса шума по ошибке как доля золотой")
    args, extra = parser.parse_known_args()

    golden_path = Path(args.golden)
    if args.from_csv:
        csv_path = Path(args.from_csv)
        if not csv_path.exists():
            print(f"нет таблицы {csv_path}")
            return 2
    else:
        csv_path = Path(args.out)
        config = load_golden(golden_path).config if golden_path.exists() else {}
        if not config and not args.freeze:
            print(f"нет золота {golden_path} — сперва заморозить: "
                  f"python scripts/regress.py --freeze")
            return 2
        code = _run_eval(config, csv_path, extra)
        if code != 0:
            print(f"прогон завершился с кодом {code}")
            return code

    rows = _read_rows(csv_path)
    config = _read_config(csv_path)

    if args.freeze:
        golden = freeze(rows, config, note=args.note)
        save_golden(golden, golden_path, header=HEADER)
        print(f"\nзолото заморожено → {golden_path}  ({len(golden.cases)} кейсов)")
        for outcome, count in sorted(golden.summary.items(),
                                     key=lambda kv: -kv[1]):
            print(f"  {count:>3}  {OUTCOME_RU.get(outcome, outcome)}")
        return 0

    golden = load_golden(golden_path)
    report = compare(rows, golden, config,
                     slack_m=args.slack_m, slack_frac=args.slack_frac)
    _print_report(report, golden, rows)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
