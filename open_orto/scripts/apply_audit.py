"""Развод корпуса по вердикту аудита: годное — в корпус, брак — в карантин.

Аудит (`audit_basemap.py`) выносит вердикт площадке, а не паре: привязка
меряется на площадку целиком, и если она сдвинута, сдвинуты все её пары.
Здесь вердикт применяется к файлам.

Брак **перемещается, а не удаляется**: «привязка сдвинута» — это материал для
разбора (почему подложка разошлась именно здесь), а восстановить его дешевле
переносом, чем повторной генерацией. В манифест добавляется колонка вердикта,
чтобы обучение могло отделить подтверждённые площадки от непроверяемых:
«без структуры» — не брак, там просто нечем подтвердить разметку.

    python open_orto/scripts/apply_audit.py --dataset open_orto/dataset_base \\
        --audit open_orto/work/audit_basemap.csv
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

#: Вердикты, при которых пары уводятся из корпуса.
QUARANTINE = {"привязка сдвинута"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="open_orto/dataset_base")
    ap.add_argument("--audit", default="open_orto/work/audit_basemap.csv")
    ap.add_argument("--quarantine", default="", help="куда уводить брак "
                    "(по умолчанию <dataset>_quarantine)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.dataset)
    quar = Path(args.quarantine) if args.quarantine else root.with_name(root.name + "_quarantine")
    verdicts = {r["scene"]: r["вердикт"]
                for r in csv.DictReader(Path(args.audit).open(encoding="utf-8"))}

    mf = root / "manifest.csv"
    lines = mf.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    scene_i = header.index("scene")
    rows = [ln.split(",") for ln in lines[1:] if ln.strip()]

    keep, moved, unknown = [], [], 0
    for r in rows:
        v = verdicts.get(r[scene_i])
        if v is None:
            unknown += 1
            keep.append((r, "не проверялась"))
        elif v in QUARANTINE:
            moved.append((r, v))
        else:
            keep.append((r, v))

    print(f"пар в корпусе {len(rows)}: остаётся {len(keep)}, "
          f"в карантин {len(moved)}" + (f", без вердикта {unknown}" if unknown else ""))
    if args.dry_run:
        return 0

    quar.mkdir(parents=True, exist_ok=True)
    gone = 0
    for r, _ in moved:
        src = root / r[0]
        if src.exists():
            shutil.move(str(src), str(quar / r[0]))
            gone += 1
    if moved:
        qm = quar / "manifest.csv"
        new = not qm.exists()
        with qm.open("a", encoding="utf-8") as fh:
            if new:
                fh.write(",".join(header + ["вердикт"]) + "\n")
            for r, v in moved:
                fh.write(",".join(r + [v]) + "\n")

    with mf.open("w", encoding="utf-8") as fh:
        fh.write(",".join(header + ["вердикт"]) + "\n")
        for r, v in keep:
            fh.write(",".join(r + [v]) + "\n")

    files = len(list(root.glob("*.npz")))
    print(f"перемещено файлов: {gone} → {quar}")
    print(f"корпус: строк манифеста {len(keep)}, файлов {files}"
          + ("  ← расходится!" if files != len(keep) else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
