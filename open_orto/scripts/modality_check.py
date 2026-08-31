"""Проверка модальности площадок: фотоснимок или визуализация рельефа.

В наборе попадаются не только ортофото: часть растров — карты высот (DSM/DTM),
сохранённые как обычная RGB-картинка. Отличить их по паспорту нельзя (те же
uint8, три канала), а последствия разные:

- если такой растр попал в корпус как фото, его пары учат матчер несуществующей
  фактуре — рельеф гладкий, «объектов» на нём нет;
- если он лежит **рядом** с настоящим ортофото той же территории, это не дубль
  и не другая дата, а **другая модальность**: ценный, но совсем иной материал.

Различаются по содержанию. У фотоснимка мелкая текстура и разнородный цвет; у
визуализации рельефа поле гладкое (энергия высоких частот мала), а цвет либо
почти серый, либо идёт непрерывной палитрой без разрывов.

    python open_orto/scripts/modality_check.py --sheet
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import rasterio

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

from rasters import to_rgb  # noqa: E402

#: Решающий признак — **доля строго серых пикселей** (R = G = B). У
#: визуализации рельефа и у панхрома каналы совпадают точно, у блёклого
#: фотоснимка — почти никогда: горная зелень даёт низкую насыщенность, но не
#: нулевую. Проверено: малонасыщенный горный снимок имел sat 0.057 и долю
#: серых 0.03, карты высот — sat 0.0 и долю 1.0.
GRAY_FRAC_MIN = 0.90
#: Вторичный признак: гладкость. Разделяет слабо (у рельефа 0.45–0.64, у фото
#: от 0.635), поэтому идёт справкой, а не критерием.
SMOOTH_LOW = 0.70


def features(path: Path, side: int = 384) -> dict | None:
    """Признаки содержания: гладкость, насыщенность, разнообразие цвета."""
    try:
        with rasterio.open(path) as ds:
            k = max(ds.width, ds.height) / side
            arr = ds.read(out_shape=(ds.count, max(1, int(ds.height / k)),
                                     max(1, int(ds.width / k))))
        rgb = to_rgb(arr).astype(np.uint8)
    except Exception:  # noqa: BLE001
        return None
    # рабочая область: у ортопланов края — чёрные треугольники поворота
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    live = gray > 8
    if live.mean() < 0.1:
        return None
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = float(hsv[..., 1][live].mean() / 255.0)
    # гладкость: энергия лапласиана, нормированная на разброс яркости —
    # у рельефа мелкой фактуры нет, у фото она и есть содержание
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    smooth = float(np.abs(lap[live]).mean() / max(gray[live].std(), 1e-6))
    # строго серые пиксели: каналы совпадают до единицы яркости
    r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
    gray_frac = float(((np.abs(r - g) <= 1) & (np.abs(g - b) <= 1))[live].mean())
    return dict(sat=round(sat, 4), smooth=round(smooth, 5),
                gray_frac=round(gray_frac, 4), live=round(float(live.mean()), 3))


def verdict(f: dict) -> str:
    """Вид содержимого. «Серый» — это ещё не рельеф: панхром выглядит так же,
    поэтому окончательное различение делает человек по контрольному листу."""
    if f is None:
        return "не прочитан"
    if f["gray_frac"] >= GRAY_FRAC_MIN:
        return ("серый: похоже на рельеф" if f["smooth"] < SMOOTH_LOW
                else "серый: панхром или рельеф")
    return "фото"


def thumb(path: Path, side: int = 300) -> str:
    try:
        with rasterio.open(path) as ds:
            k = max(ds.width, ds.height) / side
            arr = ds.read(out_shape=(ds.count, max(1, int(ds.height / k)),
                                     max(1, int(ds.width / k))))
        rgb = to_rgb(arr).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 82])
        return base64.b64encode(buf).decode()
    except Exception:  # noqa: BLE001
        return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default="openaerialmap_dataset/manifest.csv")
    ap.add_argument("--data-dir", default="E:/open_ortophoto_data")
    ap.add_argument("--out", default="open_orto/work/modality.csv")
    ap.add_argument("--sheet", action="store_true", help="HTML с подозрительными")
    ap.add_argument("--sheet-out", default="openaerialmap_dataset/MODALITY.html")
    args = ap.parse_args()
    from cpu_affinity import pin_to_performance
    pin_to_performance(verbose=False)

    man = list(csv.DictReader(Path(args.manifest).open(encoding="utf-8")))
    pairs_by_scene = Counter(r["scene"] for r in man)
    scenes = sorted(pairs_by_scene)
    data = Path(args.data_dir)

    rows = []
    for i, s in enumerate(scenes, 1):
        f = features(data / f"{s}.tif")
        rows.append(dict(scene=s, pairs=pairs_by_scene[s], вердикт=verdict(f),
                         **(f or dict(sat="", smooth="", gray_frac="", live=""))))
        if i % 100 == 0:
            print(f"  {i}/{len(scenes)}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    tally = Counter(r["вердикт"] for r in rows)
    print(f"\nплощадок: {len(rows)}")
    for v, n in tally.most_common():
        print(f"  {v:28} {n:4}")
    good = [r for r in rows if r["вердикт"] == "фото" and r["smooth"] != ""]
    sm = np.array([r["smooth"] for r in good], float)
    sa = np.array([r["sat"] for r in good], float)
    print(f"\nу фото: гладкость {np.percentile(sm, 1):.4f}–{sm.max():.4f} "
          f"(медиана {np.median(sm):.4f}), насыщенность "
          f"{np.percentile(sa, 1):.3f}–{sa.max():.3f} (медиана {np.median(sa):.3f})")
    susp = [r for r in rows if r["вердикт"] not in ("фото",)]
    if susp:
        print(f"\nподозрительные ({len(susp)}), пар в корпусе "
              f"{sum(r['pairs'] for r in susp)}:")
        for r in sorted(susp, key=lambda r: r["smooth"] if r["smooth"] != "" else 9)[:15]:
            print(f"  {r['scene'][:16]:16} серых {r['gray_frac']} "
                  f"гладкость {r['smooth']} — {r['вердикт']} ({r['pairs']} пар)")

    if args.sheet and susp:
        parts = [f"""<h1>Проверка модальности площадок</h1>
<p class=lead>Проверено {len(rows)} площадок корпуса. Подозрительных —
<b>{len(susp)}</b> ({sum(r['pairs'] for r in susp)} пар). Признаки: гладкость
(энергия высоких частот к разбросу яркости) и насыщенность цвета. У
визуализации рельефа мелкой фактуры нет, у фотоснимка она и есть содержание.</p>
<p class=note>Решающий признак — доля строго серых пикселей (R = G = B ± 1):
у визуализации рельефа и у панхрома каналы совпадают точно, у блёклого снимка
почти никогда. Порог {GRAY_FRAC_MIN}. Гладкость идёт справкой: она разделяет
слабо. Числа — повод посмотреть, а не приговор.</p>
<div class=row>"""]
        for r in sorted(susp, key=lambda r: r["smooth"] if r["smooth"] != "" else 9):
            img = thumb(data / f"{r['scene']}.tif")
            parts.append(
                f'<figure><img src="data:image/jpeg;base64,{img}"/>'
                f'<figcaption><code>{html.escape(r["scene"][:20])}</code><br>'
                f'{r["вердикт"]}<br>серых {r["gray_frac"]} · гладкость {r["smooth"]}<br>'
                f'{r["pairs"]} пар в корпусе</figcaption></figure>')
        parts.append("</div>")
        Path(args.sheet_out).write_text(f"""<!doctype html><html lang=ru><meta charset=utf-8>
<title>Модальность площадок</title>
<style>body{{font:15.5px/1.6 Georgia,serif;max-width:1200px;margin:0 auto;padding:24px 18px 70px;color:#1a1a1a;background:#fcfcfb}}
h1{{font-size:26px;margin:0 0 8px}} .lead{{font-size:17px;color:#333}}
.note{{color:#444;font-size:14.5px;background:#f8f8f6;border-left:3px solid #c9c9c0;padding:10px 14px}}
.row{{display:flex;flex-wrap:wrap;gap:16px;margin-top:18px}}
figure{{margin:0;width:300px}} img{{max-width:100%;border:1px solid #ccc;display:block}}
figcaption{{color:#666;font-size:12.5px;margin-top:5px;line-height:1.45}}
code{{background:#f0f0ec;padding:1px 4px;font-family:ui-monospace,monospace;font-size:12px}}</style>
{"".join(parts)}
</html>""", encoding="utf-8")
        print(f"\nлист: {args.sheet_out}")
    print(f"таблица: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
