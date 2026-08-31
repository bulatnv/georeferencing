"""Сборка валидных пар всех корпусов в один каталог поставки.

Корпуса собирались раздельно (`same_source`, пары с подложкой, пилотные
площадки), у них разные манифесты и пересекающиеся имена файлов: одна и та же
площадка даёт `pair_<тег>_00006_inside_ss.npz` и в корпусе с подложкой, и в
`same_source`, но это **разные** пары. Поэтому в поставке имя получает префикс
корпуса — иначе тринадцать пар молча затёрли бы друг друга.

Файлы кладутся **жёсткими ссылками**: том один, содержимое не дублируется, а
исходные каталоги остаются нетронутыми — если сборку понадобится пересобрать
иначе, терять нечего. Где ссылка невозможна (другой том), файл копируется.

Что считается валидным: всё, что прошло аудит и лежит в рабочих корпусах.
Карантин (площадки со сдвинутой привязкой) не берётся — это брак, который
выглядит нормально и потому опаснее отсутствия данных.

    python open_orto/scripts/build_dataset.py --out openaerialmap_dataset
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_parallel import _crc_ok  # noqa: E402

#: Корпуса-источники: (префикс, каталог, чем эти пары ценны).
SOURCES = [
    ("base", "open_orto/dataset_base",
     "вид виртуального борта из ортоплана против кропа спутниковой подложки"),
    ("ss", "open_orto/dataset_ss",
     "обе стороны из одного ортоплана: разметка точна по построению"),
    ("pilot", "open_orto/dataset",
     "первые пары с подложкой, собранные до сеточного режима"),
]

#: Колонки поставочного манифеста. Часть полей есть не во всех корпусах
#: (сезон — только у same_source, источник компенсации — только у пар с
#: подложкой), пустое значение здесь означает «неприменимо», а не «неизвестно».
FIELDS = ["pair", "corpus", "scene", "pair_kind", "layout", "season",
          "height_m", "tilt_deg", "yaw_deg", "scale_ratio", "b_px",
          "area_frac", "covis_frac", "compensation_src", "вердикт", "bytes"]


def link_or_copy(src: Path, dst: Path) -> str:
    """Жёсткая ссылка, а при невозможности — копия. Возвращает что вышло."""
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="openaerialmap_dataset")
    ap.add_argument("--verify", action="store_true", default=True,
                    help="проверять CRC каждого файла при переносе")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    from cpu_affinity import pin_to_performance
    pin_to_performance(verbose=False)

    out = Path(args.out)
    rows_out, stats = [], Counter()
    plan = []
    for prefix, path, _ in SOURCES:
        root = Path(path)
        mf = root / "manifest.csv"
        if not mf.exists():
            print(f"нет манифеста: {mf}")
            continue
        rows = list(csv.DictReader(mf.open(encoding="utf-8")))
        print(f"{path}: {len(rows)} пар")
        for r in rows:
            name = f"{prefix}_{r['pair']}"
            plan.append((root / r["pair"], name, prefix, r))
        stats[prefix] = len(rows)

    names = [n for _, n, _, _ in plan]
    dup = [n for n, c in Counter(names).items() if c > 1]
    if dup:
        print(f"ОСТАНОВ: имена повторяются даже с префиксом ({len(dup)}): {dup[:3]}")
        return 1
    print(f"\nвсего к сборке: {len(plan)} пар, коллизий имён нет")
    if args.dry_run:
        return 0

    out.mkdir(parents=True, exist_ok=True)
    made = Counter()
    bad = []
    for i, (src, name, prefix, r) in enumerate(plan, 1):
        dst = out / name
        if not src.exists():
            bad.append((name, "нет исходного файла"))
            continue
        if args.verify:
            err = _crc_ok(str(src))
            if err:
                bad.append((name, f"повреждён: {err}"))
                continue
        if not dst.exists():
            made[link_or_copy(src, dst)] += 1
        row = {k: r.get(k, "") for k in FIELDS}
        row["pair"], row["corpus"] = name, prefix
        # у корпусов разные наборы колонок: сезон есть только у same_source,
        # источник компенсации — только у пар с подложкой
        row["season"] = r.get("season", "")
        row["compensation_src"] = r.get("compensation_src", "")
        row["вердикт"] = r.get("вердикт", "")
        rows_out.append(row)
        if i % 1000 == 0:
            print(f"  {i}/{len(plan)}", flush=True)

    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows_out)

    files = len(list(out.glob("*.npz")))
    gb = sum(int(r["bytes"] or 0) for r in rows_out) / 2**30
    print(f"\nсобрано: {len(rows_out)} пар ({gb:.2f} ГБ), "
          f"жёстких ссылок {made['link']}, копий {made['copy']}")
    print(f"файлов в каталоге {files}, строк манифеста {len(rows_out)}"
          + ("  ← расходится!" if files != len(rows_out) else ""))
    if bad:
        print(f"пропущено {len(bad)}: {bad[:5]}")
    for prefix, _, _ in SOURCES:
        n = sum(1 for r in rows_out if r["corpus"] == prefix)
        print(f"  {prefix:6} {n:5} пар, площадок "
              f"{len({r['scene'] for r in rows_out if r['corpus'] == prefix})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
