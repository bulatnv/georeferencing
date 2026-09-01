"""Перевод датасета OrthoLoC в компактный формат: 269.5 ГБ → 44.8 ГБ.

Формат и его точность описаны в :mod:`ortholoc_store`. Здесь — прогон по
сплитам: чтение исходного сэмпла, запись компактного, **сверка с оригиналом,
пока он ещё в памяти**, и (по флагу) замена исходника.

Профиль хранения выбирается по сплиту: тестовые кодируются без потерь (на них
снимаются числа бенчмарка), обучающие — с потерями. Переопределяется флагом
``--profile``.

Сверка не выборочная: оригинал уже прочитан, поэтому сравнить GT-карты
целиком стоит долей процента времени, а поймать порчу иначе будет негде —
после удаления исходников сравнивать станет не с чем.

Имена сэмплов и структура каталогов сохраняются, поэтому всё, что читает
датасет через :func:`ortholoc_store.open_sample`, продолжает работать.

    python scripts/shrink_ortholoc.py --splits test_inPlace --jobs 6
    python scripts/shrink_ortholoc.py --splits val,train --jobs 6 --replace
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[1] / "open_orto" / "scripts"))

import ortholoc_store as store  # noqa: E402

#: Профиль по имени сплита: на тестовых снимаются метрики, поэтому там
#: хранение без потерь; всё остальное идёт в обучение, где изображения и так
#: проходят через аугментации.
def profile_for(split: str) -> str:
    return "eval" if split.startswith("test") else "train"


def gt_tol(profile: str) -> float:
    """Допуск сверки GT: округление даёт не больше половины шага по каждой оси,
    значит по модулю — шаг/√2. Двухпроцентный запас на счёт с плавающей точкой:
    без него сэмпл, лёгший ровно в предел, читался бы как порча."""
    return store.PROFILES[profile]["gt_step"] / np.sqrt(2) * 1.02


def process(task) -> dict:
    """Один сэмпл: записать компактный, сверить с оригиналом, при удаче — заменить."""
    src, dst, profile, replace = task
    src, dst = Path(src), Path(dst)
    row = dict(sample=src.stem, split=src.parent.name, profile=profile,
               src_bytes=src.stat().st_size, dst_bytes=0, gt_err_px="",
               img_err="", status="ok", note="")
    try:
        if store.is_slim(src):
            row.update(status="skip", note="уже компактный")
            return row
        if dst != src and dst.exists() and store.is_slim(dst):
            row.update(status="skip", note="уже сконвертирован")
            return row
        tmp = dst.with_suffix(".part.npz")
        with store.open_sample(src) as raw:
            store.write(tmp, raw.raw, profile=profile)
            with store.open_sample(tmp) as slim:
                err = verify(raw, slim)
                row["img_err"] = round(image_error(raw, slim), 3)
        row["gt_err_px"] = round(float(err), 4)
        if err > gt_tol(profile):
            tmp.unlink(missing_ok=True)
            row.update(status="fail", note=f"GT разошлась на {err:.4f} px")
            return row
        os.replace(tmp, dst)
        row["dst_bytes"] = dst.stat().st_size
        if profile == "eval" and row["img_err"] != 0:
            dst.unlink(missing_ok=True)
            row.update(status="fail", note="кодек без потерь изменил пиксели")
            return row
        if replace and src != dst:
            src.unlink()
    except Exception as exc:  # noqa: BLE001
        row.update(status="fail", note=f"{type(exc).__name__}: {exc}")
    return row


def image_error(raw, slim) -> float:
    """Средняя разница яркости сторон после кодирования.

    В профиле ``eval`` она обязана быть нулём: изображение — это вход матчера,
    и потери в нём меняют не хранение, а измеряемую величину.
    """
    err = 0.0
    for key in ("image_query", "image_dop"):
        a, b = raw[key].astype(np.int16), slim[key].astype(np.int16)
        err = max(err, float(np.abs(a - b).mean()))
    return err


def verify(raw, slim) -> float:
    """Максимальное расхождение GT-карт; заодно проверяет формы и маску."""
    rx, ry = raw["gt"]
    sx, sy = slim["gt"]
    ok = np.isfinite(rx) & np.isfinite(ry)
    if not np.array_equal(ok, np.isfinite(sx) & np.isfinite(sy)):
        raise ValueError("маска валидности GT не совпала")
    for key in ("image_query", "image_dop"):
        if raw[key].shape != slim[key].shape:
            raise ValueError(f"форма {key} не совпала")
    if not ok.any():
        return 0.0
    return float(np.hypot(sx[ok] - rx[ok], sy[ok] - ry[ok]).max())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="data/OrthoLoC")
    ap.add_argument("--out", default=None,
                    help="куда писать; по умолчанию рядом с исходником")
    ap.add_argument("--splits", default="demo,test_inPlace,test_outPlace,val,train")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--profile", default="auto", choices=["auto", "eval", "train"],
                    help="auto: тестовые сплиты без потерь, остальные с потерями")
    ap.add_argument("--replace", action="store_true",
                    help="удалять исходный сэмпл после успешной сверки")
    ap.add_argument("--limit", type=int, default=0, help="взять первые N сэмплов")
    ap.add_argument("--report", default="eval_out/ortholoc_shrink.csv")
    args = ap.parse_args()
    try:
        from cpu_affinity import pin_to_performance
        pin_to_performance(verbose=False)
    except Exception:  # noqa: BLE001
        pass

    root = Path(args.root)
    tasks = []
    for split in args.splits.split(","):
        src_dir = root / split
        if not src_dir.is_dir():
            print(f"нет каталога {src_dir} — пропуск")
            continue
        dst_dir = Path(args.out) / split if args.out else src_dir
        dst_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(src_dir.glob("*.npz"))
        if args.limit:
            files = files[:args.limit]
        prof = profile_for(split) if args.profile == "auto" else args.profile
        tasks += [(str(f), str(dst_dir / f.name), prof, args.replace)
                  for f in files]

    if not tasks:
        print("нечего делать")
        return 1
    print(f"сэмплов: {len(tasks)}, процессов: {args.jobs}, "
          f"замена исходников: {'да' if args.replace else 'нет'}")

    rows, t0 = [], time.time()
    src_total = dst_total = 0
    with Pool(args.jobs) as pool:
        for i, row in enumerate(pool.imap_unordered(process, tasks, chunksize=4), 1):
            rows.append(row)
            src_total += row["src_bytes"]
            dst_total += row["dst_bytes"]
            if i % 200 == 0 or i == len(tasks):
                el = time.time() - t0
                left = el / i * (len(tasks) - i)
                print(f"  {i}/{len(tasks)}  {src_total/2**30:.1f} → "
                      f"{dst_total/2**30:.1f} ГБ  {el/60:.1f} мин, "
                      f"осталось ~{left/60:.0f} мин", flush=True)

    bad = [r for r in rows if r["status"] == "fail"]
    done = [r for r in rows if r["status"] == "ok"]
    rep = Path(args.report)
    rep.parent.mkdir(parents=True, exist_ok=True)
    with rep.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["split"], r["sample"])))

    print(f"\nготово за {(time.time()-t0)/60:.1f} мин")
    print(f"  успешно {len(done)}, пропущено {len(rows)-len(done)-len(bad)}, "
          f"ошибок {len(bad)}")
    if done:
        s = sum(r["src_bytes"] for r in done)
        d = sum(r["dst_bytes"] for r in done)
        print(f"  объём {s/2**30:.1f} → {d/2**30:.1f} ГБ  (×{s/max(d,1):.1f}, "
              f"освобождено {(s-d)/2**30:.1f} ГБ)")
        for prof in sorted({r["profile"] for r in done}):
            part = [r for r in done if r["profile"] == prof]
            errs = [r["gt_err_px"] for r in part if r["gt_err_px"] != ""]
            imgs = [r["img_err"] for r in part if r["img_err"] != ""]
            ps = sum(r["src_bytes"] for r in part)
            pd = sum(r["dst_bytes"] for r in part)
            print(f"  профиль {prof}: {len(part)} сэмплов, "
                  f"{ps/2**30:.1f} → {pd/2**30:.1f} ГБ (×{ps/max(pd,1):.1f}), "
                  f"GT макс {max(errs):.4f} px при допуске {gt_tol(prof):.4f}, "
                  f"изображения макс {max(imgs):.3f} уровня")
    for r in bad[:10]:
        print(f"  ОШИБКА {r['split']}/{r['sample']}: {r['note']}")
    print(f"отчёт: {rep}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
