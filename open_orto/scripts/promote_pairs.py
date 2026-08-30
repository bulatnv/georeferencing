"""Перенос пар из экспериментального прогона в корпус по вердикту аудита.

Повторная генерация площадки даёт файлы с **теми же именами**, что и прошлая
(имя строится из растра и порядкового номера), поэтому при переносе им
добавляется суффикс: иначе повторные пары нельзя отличить от прежних, лежащих
в карантине, и любой возврат карантина затёр бы их.

Переносятся только площадки с нужным вердиктом, и каждый файл проверяется на
целостность перед тем, как попасть в корпус.

    python open_orto/scripts/promote_pairs.py --from open_orto/work/refix_B \\
        --audit open_orto/work/audit_refix_B.csv --suffix r2
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_parallel import _crc_ok  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--audit", required=True)
    ap.add_argument("--to", default="open_orto/dataset_base")
    ap.add_argument("--verdict", default="разметка подтверждена")
    ap.add_argument("--suffix", default="r2",
                    help="метка повторной генерации в имени файла")
    ap.add_argument("--note", default="",
                    help="что писать в колонку вердикта (по умолчанию — сам вердикт)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.to)
    keep = {r["scene"] for r in csv.DictReader(Path(args.audit).open(encoding="utf-8"))
            if r["вердикт"] == args.verdict}

    rows = list(csv.DictReader((src / "manifest.csv").open(encoding="utf-8")))
    take = [r for r in rows if r["scene"] in keep]
    print(f"площадок с вердиктом «{args.verdict}»: {len(keep)}, "
          f"их пар: {len(take)} из {len(rows)}")

    dst_head = (dst / "manifest.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    known = {ln.split(",", 1)[0]
             for ln in (dst / "manifest.csv").read_text(encoding="utf-8").splitlines()[1:]}

    plan, bad, clash = [], [], []
    for r in take:
        f = src / r["pair"]
        # пустой суффикс — для площадок, которых в корпусе ещё не было:
        # переименовывать нечего, а коллизию всё равно ловит проверка ниже
        new_name = f"{f.stem}_{args.suffix}.npz" if args.suffix else f.name
        if new_name in known or (dst / new_name).exists():
            clash.append(new_name)
            continue
        if _crc_ok(str(f)):
            bad.append(r["pair"])
            continue
        plan.append((f, new_name, r))

    if bad:
        print(f"повреждённых файлов пропущено: {len(bad)}")
    if clash:
        print(f"имена уже заняты в корпусе, пропущено: {len(clash)}")
    print(f"к переносу: {len(plan)} пар")
    if args.dry_run:
        return 0

    # запятая в примечании разорвала бы строку CSV на лишнюю колонку
    note = (args.note or args.verdict).replace(",", ";")
    with (dst / "manifest.csv").open("a", encoding="utf-8") as fh:
        for f, new_name, r in plan:
            shutil.copy2(f, dst / new_name)
            r = dict(r, pair=new_name)
            fh.write(",".join(r.get(c, "") if c != "вердикт" else note
                              for c in dst_head) + "\n")

    files = len(list(dst.glob("*.npz")))
    lines = len((dst / "manifest.csv").read_text(encoding="utf-8").splitlines()) - 1
    print(f"перенесено {len(plan)} пар → {dst}")
    print(f"корпус: файлов {files}, строк манифеста {lines}"
          + ("  ← расходится!" if files != lines else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
