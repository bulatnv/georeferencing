"""Разбор площадок, отсечённых гейтом привязки: почему замер не проходит.

Гейт отклоняет площадку, когда валидных узлов меньше порога, но *причина*
неудачи каждого узла в логи не попадает — там только счётчик. Здесь замер
повторяется с сохранением причин, и сразу в нескольких режимах, чтобы
отделить свойство местности от наших собственных настроек:

- **как есть** — те же константы, что в прогоне;
- **малое окно** — узел 200/100 м вместо 400/200: у площадки в 1 км² рабочая
  зона после эрозии втрое меньше габарита, и окно в 400 м туда просто не
  помещается. Тогда отказ говорит о размере окна, а не о местности;
- **широкий потолок сдвига** — 60 м вместо 20: если геопривязка ортоплана
  расходится с подложкой сильнее потолка, верный замер отбрасывается как
  «вне потолка», и площадка выглядит неизмеримой, хотя сдвиг у неё
  систематический и постоянный.

    python open_orto/scripts/diagnose_gate.py --scenes 12 --nodes 12
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import shift_field as SF  # noqa: E402
from rasters import BasemapSource, NoImageryError, OrthoSource  # noqa: E402

#: Режимы: имя → что переопределяем в модуле замера.
MODES = {
    "как есть": {},
    "малое окно 200/100 м": dict(NODE_COARSE_M=200.0, NODE_FINE_M=100.0),
    "потолок сдвига 60 м": dict(MAX_SHIFT_M=60.0),
    "малое окно + потолок 60 м": dict(NODE_COARSE_M=200.0, NODE_FINE_M=100.0,
                                      MAX_SHIFT_M=60.0),
}


def run_mode(ortho, base, nodes, over: dict):
    """Замер узлов при временно изменённых константах: (валидных, причины)."""
    saved = {k: getattr(SF, k) for k in over}
    for k, v in over.items():
        setattr(SF, k, v)
    try:
        ok, reasons = 0, Counter()
        for gx, gy in nodes:
            rec = SF.measure_node(ortho, base, gx, gy)
            if rec.get("ok"):
                ok += 1
            else:
                r = rec.get("reason", "?")
                # «сдвиг 34.2 м вне потолка» — величина у каждого своя,
                # для сводки важен сам вид отказа
                reasons["сдвиг вне потолка" if "вне потолка" in r else r] += 1
        return ok, reasons
    finally:
        for k, v in saved.items():
            setattr(SF, k, v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rejected", default="open_orto/work/basemap/rejected.csv")
    ap.add_argument("--data-dir", default="E:/open_ortophoto_data")
    ap.add_argument("--scenes", type=int, default=12)
    ap.add_argument("--nodes", type=int, default=12, help="узлов на площадку")
    ap.add_argument("--out", default="open_orto/work/diagnose_gate.csv")
    args = ap.parse_args()
    from cpu_affinity import pin_to_performance
    pin_to_performance(verbose=False)

    rows = list(csv.DictReader(Path(args.rejected).open(encoding="utf-8")))
    # равномерно по площади: мелкие и крупные отказные ведут себя по-разному
    rows.sort(key=lambda r: float(r["km2"]))
    step = max(1, len(rows) // args.scenes)
    take = rows[::step][: args.scenes]
    print(f"отказных площадок: {len(rows)}, разбираем {len(take)}", flush=True)

    fields = ["scene", "km2", "режим", "узлов", "валидных"] + ["причины"]
    totals = defaultdict(lambda: [0, 0, Counter()])
    with Path(args.out).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(take, 1):
            src = Path(args.data_dir) / f"{r['scene']}.tif"
            if not src.exists():
                continue
            try:
                ortho = OrthoSource(src)
                base = BasemapSource(ortho)
            except NoImageryError as exc:
                print(f"  {r['scene'][:12]}: подложки нет — {exc}", flush=True)
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"  {r['scene'][:12]}: не открылся — {exc}", flush=True)
                continue
            try:
                step_m = SF.step_for_nodes(ortho, args.nodes * 2)
                nodes, _ = SF.build_nodes(ortho, step_m)
                nodes = nodes[: args.nodes]
                if not nodes:
                    print(f"  {r['scene'][:12]}: узлов не построить "
                          f"(рабочая зона мала)", flush=True)
                    totals["рабочая зона мала"][0] += 1
                    continue
                for mode, over in MODES.items():
                    ok, reasons = run_mode(ortho, base, nodes, over)
                    w.writerow({"scene": r["scene"], "km2": r["km2"], "режим": mode,
                                "узлов": len(nodes), "валидных": ok,
                                "причины": "; ".join(f"{k}×{v}" for k, v in
                                                     reasons.most_common())})
                    t = totals[mode]
                    t[0] += len(nodes)
                    t[1] += ok
                    t[2].update(reasons)
                fh.flush()
                print(f"  {i}/{len(take)} {r['scene'][:12]} ({float(r['km2']):.1f} км²): "
                      + ", ".join(f"{m} {totals[m][1]}" for m in MODES), flush=True)
            finally:
                ortho.close()

    print(f"\n{'режим':28} {'узлов':>7} {'валидных':>9}  главные причины отказа")
    for mode in MODES:
        n, ok, reasons = totals[mode]
        if not n:
            continue
        top = ", ".join(f"{k} — {v}" for k, v in reasons.most_common(3))
        print(f"{mode:28} {n:7} {ok:9} ({100*ok/n:3.0f}%)  {top}")
    print(f"\nпострочно: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
