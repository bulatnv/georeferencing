"""Параллельная генерация корпуса same_source по многим ортофотопланам.

Один процесс ``generate.py`` держит примерно одно ядро: чтение окон GeoTIFF,
ремап кадра и упаковка npz идут последовательно, и на 20-ядерной машине это
недогрузка в разы. Здесь список растров режется на несколько частей и каждая
отдаётся своему процессу — параллелизм по площадкам, а не внутри площадки:
растры независимы, общего состояния нет, поэтому это самое дешёвое место для
распараллеливания.

Единственное, что нельзя делить, — манифест: несколько процессов, дописывающих
один CSV, перемешают строки. Поэтому каждому выдаётся свой файл (``--manifest``
у генератора), а по окончании они сливаются в общий.

    python open_orto/scripts/run_parallel.py --jobs 5 --out open_orto/dataset_ss

Уже обработанные площадки исключаются по колонке ``scene`` существующего
манифеста, так что запуск можно повторять — он продолжает, а не начинает
заново.
"""

from __future__ import annotations

import argparse
import csv
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor
import subprocess
import sys
import time
from pathlib import Path

#: Отбор «годных» растров по результатам скана (``rasters_scan.csv``).
#: km2 = 0 означает географический CRS без метрического разрешения — такой
#: растр генератор не потянет; res > 0.1 м/пкс слишком груб для кадра борта.
MIN_KM2 = 0.5
MAX_RES = 0.1


def load_candidates(scan: Path, min_km2: float, max_res: float):
    """Годные площадки из скана: [(имя, км²), ...], крупные первыми."""
    rows = list(csv.DictReader(scan.open(encoding="utf-8")))
    out = []
    for r in rows:
        km2, res = float(r["km2"]), float(r["res"])
        if km2 >= min_km2 and 0.0 < res <= max_res:
            out.append((r["name"], km2))
    out.sort(key=lambda t: -t[1])
    return out


def done_scenes(out_dir: Path, extra: Path | None):
    """Площадки, которые уже в корпусе: по манифесту плюс явный чёрный список."""
    seen = set()
    mf = out_dir / "manifest.csv"
    if mf.exists():
        for r in csv.DictReader(mf.open(encoding="utf-8")):
            seen.add(r["scene"])
    if extra and extra.exists():
        seen.update(extra.read_text(encoding="utf-8").split())
    return seen


def split_balanced(items, jobs: int):
    """Раскладка по корзинам «самый большой — в самую пустую» (LPT).

    Время площадки растёт с её площадью, а размеры отличаются на порядок
    (0.5–45 км²). Нарезка подряд дала бы корзину, работающую вдвое дольше
    прочих, — то есть параллелизм только на первой половине прогона.
    """
    bins = [[] for _ in range(jobs)]
    load = [0.0] * jobs
    for name, km2 in items:
        i = min(range(jobs), key=lambda k: load[k])
        bins[i].append(name)
        load[i] += km2
    return bins, load


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scan", default="open_orto/work/rasters_scan.csv")
    ap.add_argument("--data-dir", default="E:/open_ortophoto_data")
    ap.add_argument("--out", default="open_orto/dataset_ss")
    ap.add_argument("--work", default="open_orto/work/parallel")
    ap.add_argument("--exclude", default="open_orto/work/rasters_clash.txt",
                    help="список площадок, которые брать нельзя")
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--min-km2", type=float, default=MIN_KM2)
    ap.add_argument("--max-res", type=float, default=MAX_RES)
    ap.add_argument("--limit", type=int, default=0, help="взять не больше N площадок")
    ap.add_argument("--cell-m", type=float, default=400.0)
    ap.add_argument("--per-cell", default="1")
    ap.add_argument("--max-per-raster", type=int, default=15)
    ap.add_argument("--erosion-m", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    cand = load_candidates(Path(args.scan), args.min_km2, args.max_res)
    seen = done_scenes(out, Path(args.exclude) if args.exclude else None)
    rest = [(n, k) for n, k in cand if n not in seen]
    if args.limit:
        rest = rest[:args.limit]

    total_km2 = sum(k for _, k in rest)
    print(f"годных площадок: {len(cand)}, уже в корпусе: {len(seen)}, "
          f"к обработке: {len(rest)} ({total_km2:.0f} км²)")
    if not rest:
        print("нечего обрабатывать")
        return 0

    jobs = min(args.jobs, len(rest))
    bins, load = split_balanced(rest, jobs)

    procs = []
    for i, names in enumerate(bins):
        part = work / f"part_{i}.txt"
        part.write_text("\n".join(names) + "\n", encoding="utf-8")
        mani = work / f"manifest_{i}.csv"
        log = work / f"log_{i}.txt"
        cmd = [sys.executable, "open_orto/scripts/generate.py",
               "--same-source-only",
               "--rasters", str(part),
               "--data-dir", args.data_dir,
               "--out", str(out),
               "--manifest", str(mani),
               "--cell-m", str(args.cell_m),
               "--per-cell", str(args.per_cell),
               "--max-per-raster", str(args.max_per_raster),
               "--erosion-m", str(args.erosion_m),
               "--seed", str(args.seed)]
        print(f"  часть {i}: {len(names)} площадок, {load[i]:.0f} км² → {log.name}")
        if args.dry_run:
            continue
        fh = log.open("w", encoding="utf-8")
        procs.append((i, subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                          env={**os.environ,
                                               "PYTHONIOENCODING": "utf-8"}), fh))
    if args.dry_run:
        return 0

    t0 = time.time()
    codes = []
    for i, p, fh in procs:
        codes.append((i, p.wait()))
        fh.close()
        print(f"  часть {i} завершена, код {codes[-1][1]}, "
              f"{(time.time() - t0) / 60:.0f} мин от старта", flush=True)

    merge(out, work, jobs)
    bad = [i for i, c in codes if c != 0]
    print(f"готово за {(time.time() - t0) / 60:.0f} мин"
          + (f"; с ошибкой: части {bad}" if bad else ""))
    return 1 if bad else 0


def _crc_ok(path: str):
    """None, если контейнер цел; иначе имя повреждённого члена."""
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip()
    except Exception as exc:  # noqa: BLE001
        return f"open: {exc}"


def verify(out: Path, names, workers: int = 8):
    """Проверка CRC записанных пар; битые удаляются, их имена возвращаются.

    На прогоне 2195 пар один контейнер оказался с битым CRC — запись
    оборвалась незаметно для генератора. Пара, которая не читается, хуже
    отсутствующей: она свалит обучение на середине эпохи, и виновника будут
    искать в загрузчике. Дешевле проверить сразу, пока известно, какие
    файлы новые.
    """
    paths = [str(out / n) for n in names]
    bad = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for name, err in zip(paths, ex.map(_crc_ok, paths, chunksize=8)):
            if err:
                bad.append(Path(name))
    for b in bad:
        b.unlink(missing_ok=True)
    print(f"проверка целостности: {len(paths)} пар, битых {len(bad)}"
          + (" (удалены)" if bad else ""))
    return {b.name for b in bad}


def merge(out: Path, work: Path, jobs: int) -> None:
    """Слияние частичных манифестов в общий (заголовок пишем один раз).

    Строки сначала собираются, затем проверяется целостность их файлов и
    только потом дописываются: в манифесте не должно быть того, что не
    читается.
    """
    mf = out / "manifest.csv"
    head = ("pair,scene,layout,pair_kind,season,height_m,tilt_deg,yaw_deg,"
            "scale_ratio,b_px,area_frac,covis_frac,bytes\n")
    rows = []
    for i in range(jobs):
        part = work / f"manifest_{i}.csv"
        if not part.exists():
            continue
        rows += [ln for ln in part.read_text(encoding="utf-8").splitlines()[1:] if ln.strip()]
        # os.replace, а не rename: следы прошлых прогонов не должны ронять
        # слияние уже после того, как часть строк переписана
        part.replace(part.with_suffix(".csv.merged"))

    dropped = verify(out, [ln.split(",", 1)[0] for ln in rows])
    rows = [ln for ln in rows if ln.split(",", 1)[0] not in dropped]

    new = not mf.exists()
    with mf.open("a", encoding="utf-8") as dst:
        if new:
            dst.write(head)
        for ln in rows:
            dst.write(ln + "\n")
    print(f"манифест: +{len(rows)} строк → {mf}")


if __name__ == "__main__":
    raise SystemExit(main())
