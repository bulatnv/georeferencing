"""E1: сигналы качества на ОРАКУЛЬНОМ выравнивании — что различает, а что нет.

Первый шаг трека B ([ROADMAP.md](../docs/ROADMAP.md), фаза 1). Обзор
`RESEARCH_B_VERIFICATION.md` сделал точное наблюдение: NCC выполняет **две разные
работы** — меряет качество выравнивания и служит дискриминатором «то место или не
то». Сезон ломает первую наверняка. Ломает ли вторую — вопрос эмпирический, и
ответ на него не требует ни одного нового матча.

Идея: у всех 16 кейсов есть истина, поэтому верную позу можно построить **без
матчера** и посчитать на ней все меры-кандидаты. Это даёт то, чего пайплайн дать
не может, — **кросс-сезонные позитивы**: `Ufa2`/`Ufa3` при заведомо верной позе.

Как строится оракульное выравнивание
------------------------------------
Кадр приводится к разрешению подложки (``frame_at_mpp``), поворачивается на
``−yaw`` (та же операция, что делает предповорот в пайплайне) и обрезается
вписанным квадратом — так он становится «север вверх» в масштабе подложки.
Окно подложки берётся того же размера и центрируется в той же точке. Значит,
пиксель ``(i, j)`` обеих картинок — одна и та же точка земли.

Откуда берутся центр и курс:

``exif``   центр — GPS кадра, курс — из EXIF. Ни одного пикселя подложки в
           построении не участвует, матчер не вызывается. **Кросс-сезонные
           кейсы попадают именно сюда** — то есть главная группа чиста.
``manual`` курс из карты не восстановить, поэтому берётся поза, найденная
           пайплайном и **подтверждённая владельцем** визуально (её ошибка
           проверяется по манифесту). Это летние кадры, они и так локализуются;
           их роль — референс «как выглядят меры, когда всё хорошо».

Негативы — те же окна, сдвинутые на заданные расстояния по четырём азимутам.
Сдвиг делается по подложке при неизменном кадре: так «ложная пара» отличается от
верной **только местом**, а не сезоном, масштабом и поворотом.

    python scripts/e1_signals.py
    python scripts/e1_signals.py --cases Ufa2,Ufa3,00049 --no-dino

Результат — CSV со всеми парами и таблица «мера × режим»: AUC внутри режима
(различает ли вообще) и сдвиг медианы верных между режимами (нужен ли режимный
порог). Три исхода и три разных вывода, см. `docs/JOURNAL.md`.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from aero_geoloc.basemap import ESRI_WORLD_IMAGERY, TileBasemap, TileCache  # noqa: E402
from aero_geoloc.dataset import EvalCase, load_dataset  # noqa: E402
from aero_geoloc.geo import ground_mpp  # noqa: E402
from aero_geoloc.oracle import alignment_for, north_up_crop, offset_lonlat, to_gray  # noqa: E402
from aero_geoloc.similarity import SIGNALS, DenseDinoSimilarity  # noqa: E402

FIELDS = ["case", "regime", "truth_source", "align_source", "pair", "offset_m", "bearing_deg",
          "resample", "size_px", "work_px", *SIGNALS, "dino"]


def _at_work_scale(a, b, work_px: int):
    """Обе картинки — к общему рабочему размеру ОДНИМ коэффициентом."""
    if work_px <= 0 or a.shape[0] <= work_px:
        return a, b
    size = (work_px, work_px)
    return (cv2.resize(a, size, interpolation=cv2.INTER_AREA),
            cv2.resize(b, size, interpolation=cv2.INTER_AREA))


def _load_poses(path: Path) -> dict[str, tuple[float, float, float]]:
    """Позы manual-кейсов: имя → (lat, lon, heading). Снимаются прогоном оценки."""
    if not path.exists():
        return {}
    poses = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                poses[row["case"]] = (float(row["found_lat"]), float(row["found_lon"]),
                                      float(row["heading_deg"]))
            except (KeyError, ValueError):
                continue
    return poses


def _pairs_for_case(case: EvalCase, args, basemap, max_zoom, poses) -> list[dict]:
    align = alignment_for(case, poses, tolerance_m=args.pose_tolerance_m)
    if align is None:
        print("  пропуск: оракульную позу построить не из чего "
              f"(нужен прогон eval_dataset.py → {args.poses})")
        return []
    lat, lon, yaw = align.lat, align.lon, align.yaw_deg
    source = align.source + (f"(+{align.drift_m:.0f}м)" if align.drift_m else "")

    z_fine = case.basemap_zoom(max_zoom=max_zoom)
    mpp = ground_mpp(case.prior.lat, z_fine)
    frame, _ = case.frame_at_mpp(mpp)
    query = north_up_crop(frame, yaw)
    side = query.shape[0]
    # Рабочее разрешение. Меры на градиентах меряют структуру, а на полном
    # разрешении (у Саратова это ~1100 px при GSD 6.5 см) структуру забивает
    # текстура: NGF на верной паре давал 0.534 при поле 0.5. Обе картинки
    # приводятся ОДНИМ коэффициентом — они уже одного размера, так что ловушка
    # раздельного ресайза (см. историю LoFTR в docs/JOURNAL.md) здесь исключена
    # по построению, но об этом стоит помнить при любой правке.
    work = min(args.work_px, side) if args.work_px > 0 else side
    rows = []

    def measure(pair: str, ref_lat: float, ref_lon: float, offset_m: float, bearing: float):
        ref, _ = basemap(ref_lon, ref_lat, z_fine, side, side)
        row = {
            "case": case.name, "regime": case.regime, "truth_source": case.truth_source,
            "align_source": source, "pair": pair, "offset_m": round(offset_m),
            "bearing_deg": round(bearing), "resample": round(case.gsd_m / mpp, 3),
            "size_px": side, "work_px": work,
        }
        a, b = _at_work_scale(query, to_gray(ref), work)
        for name, fn in SIGNALS.items():
            row[name] = round(float(fn(a, b)), 4)
        row["dino"] = round(float(args.dino(a, b)), 4) if args.dino else ""
        return row

    rows.append(measure("positive", lat, lon, 0.0, 0.0))
    for distance in args.offsets_m:
        for bearing in args.bearings:
            dlat, dlon = offset_lonlat(lat, lon, distance, bearing)
            rows.append(measure("negative", dlat, dlon, distance, bearing))
    return rows


# --- анализ -----------------------------------------------------------------

def _auc(positives: list[float], negatives: list[float]) -> float:
    """AUC = доля пар (позитив, негатив), где позитив выше. Ничьи считаются за ½."""
    if not positives or not negatives:
        return float("nan")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0
               for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def _youden(positives: list[float], negatives: list[float]) -> tuple[float, float]:
    """Порог с максимумом ``TPR − FPR`` и сам этот максимум."""
    if not positives or not negatives:
        return float("nan"), float("nan")
    best, best_j = float("nan"), -1.0
    for threshold in sorted(set(positives) | set(negatives)):
        tpr = sum(p >= threshold for p in positives) / len(positives)
        fpr = sum(n >= threshold for n in negatives) / len(negatives)
        if tpr - fpr > best_j:
            best, best_j = threshold, tpr - fpr
    return best, best_j


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def _within_case(rows: list[dict], name: str) -> tuple[int, int, float]:
    """Сколько кейсов, где верная пара выше ВСЕХ своих ложных, и медианный запас.

    Это ближе к тому, что делает пайплайн, чем глобальный порог: Этаж 2 выбирает
    победителя **среди кандидатов одного кадра**, а не сравнивает число с
    константой. Мера, у которой абсолютная шкала едет от кадра к кадру, может
    прекрасно решать эту задачу и при этом проваливать задачу с общим порогом.
    """
    wins, total, margins = 0, 0, []
    for case in sorted({r["case"] for r in rows}):
        subset = [r for r in rows if r["case"] == case and r[name] != ""]
        pos = [float(r[name]) for r in subset if r["pair"] == "positive"]
        neg = [float(r[name]) for r in subset if r["pair"] == "negative"]
        if not pos or not neg:
            continue
        total += 1
        margin = pos[0] - max(neg)
        margins.append(margin)
        if margin > 0:
            wins += 1
    return wins, total, _median(margins)


def report(rows: list[dict], signals: list[str]) -> None:
    regimes = sorted({r["regime"] for r in rows})
    width = 104
    print(f"\n{'='*width}\nE1: МЕРА × РЕЖИМ на оракульном выравнивании\n{'='*width}")
    for regime in regimes:
        subset = [r for r in rows if r["regime"] == regime]
        cases = sorted({r["case"] for r in subset})
        pos_n = sum(r["pair"] == "positive" for r in subset)
        neg_n = sum(r["pair"] == "negative" for r in subset)
        print(f"\n--- {regime}: {len(cases)} кейсов ({', '.join(cases)}), "
              f"{pos_n} верных пар, {neg_n} ложных ---")
        print(f"{'мера':<12}{'верные':>10}{'ложные':>10}{'AUC':>8}{'Юден':>8}{'порог':>10}   вывод")
        for name in signals:
            pos = [float(r[name]) for r in subset
                   if r["pair"] == "positive" and r[name] != ""]
            neg = [float(r[name]) for r in subset
                   if r["pair"] == "negative" and r[name] != ""]
            if not pos:
                continue
            auc = _auc(pos, neg)
            threshold, j = _youden(pos, neg)
            verdict = ("разделяет идеально" if auc >= 0.999 else
                       "разделяет" if auc >= 0.9 else
                       "слабо" if auc >= 0.7 else "НЕ РАЗДЕЛЯЕТ")
            print(f"{name:<12}{_median(pos):>10.3f}{_median(neg):>10.3f}"
                  f"{auc:>8.3f}{j:>8.2f}{threshold:>10.3f}   {verdict}")

    if len(regimes) > 1:
        print(f"\n{'='*width}\nСДВИГ ШКАЛЫ МЕЖДУ РЕЖИМАМИ (медиана ВЕРНЫХ пар)\n{'='*width}")
        print(f"{'мера':<12}" + "".join(f"{r:>16}" for r in regimes)
              + f"{'отношение':>12}   вывод")
        for name in signals:
            medians = []
            for regime in regimes:
                vals = [float(r[name]) for r in rows
                        if r["regime"] == regime and r["pair"] == "positive" and r[name] != ""]
                medians.append(_median(vals))
            if any(math.isnan(m) for m in medians):
                continue
            base = medians[regimes.index("in_season")] if "in_season" in regimes else medians[0]
            other = medians[regimes.index("cross_season")] if "cross_season" in regimes else medians[-1]
            ratio = other / base if base else float("nan")
            verdict = ("шкала СОПОСТАВИМА" if 0.75 <= ratio <= 1.33 else
                       "шкала поехала — нужен режимный порог" if ratio > 0.3 else
                       "шкала РУШИТСЯ")
            print(f"{name:<12}" + "".join(f"{m:>16.3f}" for m in medians)
                  + f"{ratio:>12.2f}   {verdict}")

        print(f"\n{'='*width}\nВНУТРИ КЕЙСА: верная пара выше ВСЕХ своих ложных?\n{'='*width}")
        print("Пайплайн выбирает победителя среди кандидатов ОДНОГО кадра, поэтому это")
        print("ближе к его задаче, чем общий порог. Мера может выигрывать здесь и")
        print("проваливаться там — тогда лечится не мера, а способ её применения.\n")
        print(f"{'мера':<12}" + "".join(f"{r:>18}" for r in regimes)
              + f"{'всего':>12}{'запас':>10}")
        for name in signals:
            cells = []
            for regime in regimes:
                wins, total, _ = _within_case([r for r in rows if r["regime"] == regime], name)
                cells.append(f"{wins}/{total}" if total else "—")
            wins, total, margin = _within_case(rows, name)
            print(f"{name:<12}" + "".join(f"{c:>18}" for c in cells)
                  + f"{f'{wins}/{total}':>12}{margin:>10.3f}")

        print(f"\n{'='*width}\nЕДИНЫЙ ПОРОГ НА ОБА РЕЖИМА (главный вопрос трека B)\n{'='*width}")
        print(f"{'мера':<12}{'AUC общий':>12}{'порог':>10}{'Юден':>8}   вывод")
        for name in signals:
            pos = [float(r[name]) for r in rows if r["pair"] == "positive" and r[name] != ""]
            neg = [float(r[name]) for r in rows if r["pair"] == "negative" and r[name] != ""]
            if not pos:
                continue
            auc = _auc(pos, neg)
            threshold, j = _youden(pos, neg)
            verdict = ("годится ОДИН порог" if j >= 0.95 else
                       "один порог с потерями" if j >= 0.8 else
                       "единого порога нет")
            print(f"{name:<12}{auc:>12.3f}{threshold:>10.3f}{j:>8.2f}   {verdict}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default="datasets/test_images.yaml")
    parser.add_argument("--cases", default="")
    parser.add_argument("--poses", default="eval_out/regress.csv",
                        help="таблица прогона оценки: откуда брать позу manual-кейсов")
    parser.add_argument("--pose-tolerance-m", type=float, default=150.0,
                        help="насколько поза может расходиться с ручной истиной и всё "
                             "ещё считаться оракулом (владелец метит объект, а не центр)")
    parser.add_argument("--offsets", default="150,400,1000",
                        help="сдвиги ложных пар, м")
    parser.add_argument("--bearings", default="0,90,180,270", help="азимуты сдвига, °")
    parser.add_argument("--work-px", type=int, default=512,
                        help="рабочее разрешение мер, px (0 = полное). Меры на "
                             "градиентах на полном разрешении тонут в текстуре")
    parser.add_argument("--no-dino", action="store_true", help="без плотных фич DINOv2")
    parser.add_argument("--dino-model", default="dinov2_vitb14")
    parser.add_argument("--cache", default="tiles")
    parser.add_argument("--out", default="eval_out/e1_signals.csv")
    parser.add_argument("--from-csv", default="",
                        help="только пересчитать отчёт по готовой таблице")
    args = parser.parse_args()

    if args.from_csv:
        with open(args.from_csv, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        signals = [n for n in list(SIGNALS) + ["dino"] if any(r.get(n) for r in rows)]
        report(rows, signals)
        return 0

    args.offsets_m = [float(v) for v in args.offsets.split(",") if v.strip()]
    args.bearings = [float(v) for v in args.bearings.split(",") if v.strip()]
    args.dino = None if args.no_dino else DenseDinoSimilarity(args.dino_model)

    dataset = load_dataset(args.manifest)
    cases = [c for c in dataset.cases if c.has_truth]
    if args.cases:
        wanted = {c.strip() for c in args.cases.split(",") if c.strip()}
        cases = [c for c in cases if c.name in wanted]

    basemap = TileBasemap(cache=TileCache(args.cache))
    max_zoom = ESRI_WORLD_IMAGERY.max_zoom
    poses = _load_poses(Path(args.poses))
    print(f"E1: {len(cases)} кейсов, сдвиги ложных пар {args.offsets_m} м × "
          f"{len(args.bearings)} азимутов, DINO {'нет' if args.no_dino else args.dino_model}")

    rows: list[dict] = []
    for case in cases:
        print(f"\n[{case.name}] {case.regime}, истина {case.truth_source}", flush=True)
        t0 = time.perf_counter()
        try:
            case_rows = _pairs_for_case(case, args, basemap, max_zoom, poses)
        except Exception as exc:  # noqa: BLE001 — один кейс не должен рушить прогон
            print(f"  ОШИБКА: {type(exc).__name__}: {exc}")
            continue
        rows.extend(case_rows)
        if case_rows:
            pos = case_rows[0]
            print(f"  {len(case_rows)} пар за {time.perf_counter()-t0:.0f} с; "
                  f"верная: " + "  ".join(f"{n}={pos[n]}" for n in SIGNALS)
                  + (f"  dino={pos['dino']}" if pos.get("dino") != "" else ""), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    signals = list(SIGNALS) + ([] if args.no_dino else ["dino"])
    report(rows, signals)
    print(f"\nсырьё → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
