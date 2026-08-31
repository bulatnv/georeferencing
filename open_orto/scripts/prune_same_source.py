"""Сокращение контрольной оси `same_source` в поставке.

Ось нужна — она контроль забывания и якорь геометрии, — но не в том объёме, в
каком собралась. Линия RoMa решает её полностью (inl3 = 1.00), то есть обучающего
сигнала там почти нет, а места она занимает больше половины поставки и грузится
каждую эпоху.

Понижения веса для этого мало: вес убирает пары из градиента, но не с диска.

**Что важно сохранить при сокращении:**

- покрытие территорий: хотя бы одна пара с каждой площадки, иначе контроль
  забывания перестанет видеть часть корпуса;
- оценочные сплиты: в `val` и `heldout` ось меряет деградацию, там нужна
  статистика, а не единичные пары;
- разнообразие: сезонная перекраска, наклоны, компоновки.

Удаляются **жёсткие ссылки поставки**, а не сами пары: рабочий корпус
`dataset_ss` не трогается, и сборку можно повторить в любой момент.

    python open_orto/scripts/prune_same_source.py --per-scene 1 --eval-per-scene 2
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def pick_for_scene(rows, k: int, rng) -> list:
    """k пар площадки: сначала разные сезоны, потом разные наклоны."""
    if len(rows) <= k:
        return rows
    by_season = defaultdict(list)
    for r in rows:
        by_season[r.get("season", "")].append(r)
    picked, seasons = [], sorted(by_season, key=lambda s: -len(by_season[s]))
    i = 0
    while len(picked) < k and any(by_season.values()):
        pool = by_season[seasons[i % len(seasons)]]
        if pool:
            # внутри сезона берём пару с наклоном подальше от уже взятых
            taken = [float(p["tilt_deg"]) for p in picked] or [0.0]
            pool.sort(key=lambda r: -min(abs(float(r["tilt_deg"]) - t) for t in taken))
            picked.append(pool.pop(0))
        i += 1
        if i > 4 * k + len(seasons):
            break
    return picked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="openaerialmap_dataset")
    ap.add_argument("--per-scene", type=int, default=1,
                    help="сколько пар оставить с площадки в train")
    ap.add_argument("--eval-per-scene", type=int, default=2,
                    help="сколько оставить с площадки в val и heldout: там ось "
                         "меряет деградацию, нужна статистика")
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    man = list(csv.DictReader((root / "manifest.csv").open(encoding="utf-8")))
    head = list(man[0].keys())
    rng = np.random.default_rng(args.seed)

    ss = [r for r in man if r["pair_kind"] == "same_source"]
    other = [r for r in man if r["pair_kind"] != "same_source"]
    by_scene = defaultdict(list)
    for r in ss:
        by_scene[(r["scene"], r["split"])].append(r)

    keep = []
    for (scene, split), rows in by_scene.items():
        k = args.eval_per_scene if split in ("val", "heldout") else args.per_scene
        keep += pick_for_scene(list(rows), k, rng)
    keep_names = {r["pair"] for r in keep}
    drop = [r for r in ss if r["pair"] not in keep_names]

    gb = lambda rows: sum(int(r["bytes"]) for r in rows) / 2**30
    print(f"same_source: было {len(ss)} пар ({gb(ss):.2f} ГБ), "
          f"остаётся {len(keep)} ({gb(keep):.2f} ГБ), убирается {len(drop)} "
          f"({gb(drop):.2f} ГБ)")
    print("  остаётся по сплитам:", dict(Counter(r["split"] for r in keep)))
    print(f"  площадок в остатке: {len({r['scene'] for r in keep})} "
          f"из {len({r['scene'] for r in ss})}")
    print(f"  перекрашенных: {sum(1 for r in keep if r.get('season'))}")
    total_after = len(other) + len(keep)
    print(f"поставка: {len(man)} → {total_after} пар, "
          f"{gb(man):.2f} → {gb(other) + gb(keep):.2f} ГБ")
    if args.dry_run:
        return 0

    gone = 0
    for r in drop:
        f = root / r["pair"]
        if f.exists():
            f.unlink()          # жёсткая ссылка; исходник в dataset_ss цел
            gone += 1
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=head)
        w.writeheader()
        w.writerows(other + keep)

    files = len(list(root.glob("*.npz")))
    print(f"\nудалено ссылок: {gone}")
    print(f"итог: файлов {files}, строк манифеста {total_after}"
          + ("  ← расходится!" if files != total_after else ""))
    print("исходные пары остались в open_orto/dataset_ss — сборку можно повторить")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
