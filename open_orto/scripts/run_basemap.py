"""Корпус пар «ортофотоплан ↔ спутниковая подложка» по многим площадкам.

Отличие от `run_parallel.py` (там `same_source`) — в том, что здесь у каждой
площадки своя привязка к подложке, и её надо измерить **до** генерации:

    привязка (этап Р) → гейт по числу валидных узлов → генерация пар

Гейт нужен потому, что подложка совпадает с ортопланом не везде: съёмка могла
быть зимой, а подложка летняя; территория могла застроиться; вне городов Esri
отдаёт грубый зум. Там, где привязку измерить не удалось, единственное честное
поведение — пропустить площадку: пара с непокрытой систематической ошибкой
привязки выглядит нормальной и учит модель неверному соответствию. Пропуски
пишутся в отказной список с причиной.

Параллелизм — пул воркеров по площадкам: каждая идёт целиком в своём процессе,
следующая берётся из очереди. Статичная нарезка здесь хуже, чем в
`run_parallel.py`: время площадки заранее неизвестно, оно зависит от того,
сколько узлов дадут валидный замер.

    python open_orto/scripts/run_basemap.py --jobs 5 --out open_orto/dataset_base
"""

from __future__ import annotations

import argparse
import csv
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_parallel import load_candidates, verify  # noqa: E402

#: Сколько узлов привязки заказывать на площадку. Шаг сетки считается из
#: площади, а не берётся постоянным: при шаге 250 м площадка в 45 км² даёт
#: ~700 узлов и полчаса только на привязку, тогда как пар с неё берётся
#: максимум `--max-per-raster`. Плотность узлов должна описывать поле
#: сдвигов, а не территорию.
TARGET_NODES = 60
STEP_RANGE = (200.0, 1200.0)


def step_for_area(km2: float, target: int = TARGET_NODES) -> float:
    """Шаг сетки узлов, м: столько, чтобы узлов вышло около `target`."""
    if km2 <= 0:
        return STEP_RANGE[0]
    return float(min(max((km2 * 1e6 / target) ** 0.5, STEP_RANGE[0]), STEP_RANGE[1]))


#: Гейт площадки: сколько узлов привязки должно дать валидный замер. Пять —
#: столько было у самой скудной из трёх пилотных площадок, давшей пригодный
#: корпус; меньше — привязка держится на одиночных замерах, а их ложные пики
#: нечем проверить.
MIN_VALID_NODES = 5


def field_nodes(path: Path) -> int:
    """Сколько узлов в готовом поле сдвигов (0, если поля нет)."""
    if not path.exists():
        return 0
    import numpy as np
    try:
        return int(len(np.load(path, allow_pickle=False)["x"]))
    except Exception:  # noqa: BLE001
        return 0


def run(cmd, log: Path, timeout: float | None = None) -> int:
    """Запуск шага площадки с журналом; таймаут — защита от зависшей сети."""
    with open(log, "a", encoding="utf-8") as fh:
        fh.write("$ " + " ".join(str(c) for c in cmd) + "\n")
        fh.flush()
        try:
            return subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                   env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                                   timeout=timeout)
        except subprocess.TimeoutExpired:
            fh.write("!! таймаут\n")
            return 124


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scan", default="open_orto/work/rasters_scan.csv")
    ap.add_argument("--data-dir", default="E:/open_ortophoto_data")
    ap.add_argument("--out", default="open_orto/dataset_base")
    ap.add_argument("--work", default="open_orto/work/basemap")
    ap.add_argument("--shift-dir", default="open_orto/work/shift")
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--min-km2", type=float, default=0.5)
    ap.add_argument("--max-res", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--step", type=float, default=0.0,
                    help="шаг сетки узлов привязки, м (0 — считать из площади "
                         "площадки под TARGET_NODES узлов)")
    ap.add_argument("--cell-m", type=float, default=400.0)
    ap.add_argument("--per-cell", default="2")
    ap.add_argument("--max-per-raster", type=int, default=12)
    ap.add_argument("--erosion-m", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timeout-min", type=float, default=25.0,
                    help="потолок времени на шаг: сеть иногда виснет, и одна "
                         "площадка не должна держать воркер часами")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out, work = Path(args.out), Path(args.work)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    Path(args.shift_dir).mkdir(parents=True, exist_ok=True)

    done = set()
    mf = out / "manifest.csv"
    if mf.exists():
        done = {r["scene"] for r in csv.DictReader(mf.open(encoding="utf-8"))}
    rejects = work / "rejected.csv"
    if rejects.exists():
        done |= {ln.split(",")[0] for ln in rejects.read_text(encoding="utf-8").splitlines()[1:]}

    cand = [(n, k) for n, k in load_candidates(Path(args.scan), args.min_km2, args.max_res)
            if n not in done]
    if args.limit:
        cand = cand[:args.limit]
    print(f"к обработке площадок: {len(cand)} "
          f"({sum(k for _, k in cand):.0f} км²), воркеров {args.jobs}")
    if args.dry_run or not cand:
        return 0

    q: queue.Queue = queue.Queue()
    for i, (name, km2) in enumerate(cand):
        q.put((i, name, km2))
    lock = threading.Lock()
    stats = {"ok": 0, "rejected": 0, "done": 0}
    if not rejects.exists():
        rejects.write_text("scene,km2,причина\n", encoding="utf-8")
    t0 = time.time()

    def worker(slot: int) -> None:
        log = work / f"log_{slot}.txt"
        mani = work / f"manifest_{slot}.csv"
        while True:
            try:
                _, name, km2 = q.get_nowait()
            except queue.Empty:
                return
            src = Path(args.data_dir) / f"{name}.tif"
            field = Path(args.shift_dir) / f"shift_field_{name}.npz"
            reason = None
            if not src.exists():
                reason = "файла нет"
            else:
                if not field.exists():
                    step = args.step or step_for_area(km2)
                    rc = run([sys.executable, "open_orto/scripts/shift_field.py",
                              "--raster", str(src), "--step", str(round(step)),
                              "--erosion-m", str(args.erosion_m)],
                             log, timeout=args.timeout_min * 60)
                    if rc != 0:
                        reason = f"привязка: код {rc}"
                nodes = field_nodes(field)
                if reason is None and nodes < MIN_VALID_NODES:
                    reason = f"узлов привязки {nodes} меньше {MIN_VALID_NODES}"
                if reason is None:
                    rc = run([sys.executable, "open_orto/scripts/generate.py",
                              "--raster", str(src), "--shift-field", str(field),
                              "--cell-m", str(args.cell_m), "--per-cell", args.per_cell,
                              "--max-per-raster", str(args.max_per_raster),
                              "--erosion-m", str(args.erosion_m), "--seed", str(args.seed),
                              "--out", str(out), "--manifest", str(mani)],
                             log, timeout=args.timeout_min * 60)
                    if rc != 0:
                        reason = f"генерация: код {rc}"
            with lock:
                stats["done"] += 1
                if reason:
                    stats["rejected"] += 1
                    with rejects.open("a", encoding="utf-8") as fh:
                        fh.write(f"{name},{km2:.2f},{reason}\n")
                else:
                    stats["ok"] += 1
                if stats["done"] % 10 == 0:
                    el = (time.time() - t0) / 60
                    left = (len(cand) - stats["done"]) * el / max(stats["done"], 1)
                    print(f"  {stats['done']}/{len(cand)} площадок "
                          f"(годных {stats['ok']}, отказов {stats['rejected']}), "
                          f"{el:.0f} мин, осталось ~{left:.0f} мин", flush=True)

    threads = [threading.Thread(target=worker, args=(s,), daemon=True)
               for s in range(args.jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    merge_manifests(out, work, args.jobs)
    print(f"готово за {(time.time() - t0) / 60:.0f} мин: площадок годных "
          f"{stats['ok']}, отказов {stats['rejected']} (причины — {rejects})")
    return 0


def merge_manifests(out: Path, work: Path, jobs: int) -> None:
    """Слияние манифестов воркеров с проверкой целостности записанных пар."""
    mf = out / "manifest.csv"
    head, rows = None, []
    for i in range(jobs):
        part = work / f"manifest_{i}.csv"
        if not part.exists():
            continue
        lines = part.read_text(encoding="utf-8").splitlines()
        if lines:
            head = head or lines[0]
            rows += [ln for ln in lines[1:] if ln.strip()]
        part.replace(part.with_suffix(".csv.merged"))
    if not rows:
        print("новых строк манифеста нет")
        return
    dropped = verify(out, [ln.split(",", 1)[0] for ln in rows])
    known = set()
    if mf.exists():
        known = {ln.split(",", 1)[0] for ln in mf.read_text(encoding="utf-8").splitlines()[1:]}
    rows = [ln for ln in rows
            if ln.split(",", 1)[0] not in dropped and ln.split(",", 1)[0] not in known]
    new = not mf.exists()
    with mf.open("a", encoding="utf-8") as dst:
        if new and head:
            dst.write(head + "\n")
        for ln in rows:
            dst.write(ln + "\n")
    print(f"манифест: +{len(rows)} строк → {mf}")


if __name__ == "__main__":
    raise SystemExit(main())
