"""Поиск площадок-дублей внутри набора и разметка групп в манифесте.

Прежняя проверка (`scan_rasters.py`) сравнивала **новый набор с уже
обработанными** — внутринаборные дубли через неё не проходили. Здесь набор
сверяется сам с собой: один и тот же ортоплан приходил под разными
идентификаторами и стал в манифесте двумя разными `scene`.

Дубли **не удаляются**. Это разные рендеры одной территории, они остаются
полезными; задача — не дать им разъехаться по сплитам (иначе train и held-out
увидят одни и те же дома) и не переоценить эту территорию в смеси.

Геометрическое совпадение — не доказательство тождества содержимого, поэтому
скрипт собирает контрольный лист: обзорные миниатюры площадок каждой группы
рядом, решение принимает человек.

    python open_orto/scripts/scenes_dedup.py --check-sheet
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

#: Критерий дубля — все три условия сразу. Пороги узкие намеренно: соседние
#: вылеты одной территории законно перекрываются и совпадать по всем трём
#: признакам не обязаны.
NEAR_M = 50.0
AREA_TOL = 0.02
RES_TOL = 0.02


def haversine_m(lon1, lat1, lon2, lat2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def find_pairs(scenes):
    """Пары площадок, совпавшие по центру, площади и разрешению."""
    out = []
    items = list(scenes.items())
    for i, (a, ra) in enumerate(items):
        for b, rb in items[i + 1:]:
            d = haversine_m(ra["lon"], ra["lat"], rb["lon"], rb["lat"])
            if d > NEAR_M:
                continue
            if abs(ra["km2"] - rb["km2"]) > AREA_TOL * max(ra["km2"], rb["km2"]):
                continue
            if abs(ra["res"] - rb["res"]) > RES_TOL * max(ra["res"], rb["res"]):
                continue
            out.append((a, b, d))
    return out


def content_ncc(a: Path, b: Path, side: int = 256) -> float:
    """Насколько похоже содержимое двух растров: нормированная корреляция обзоров.

    Геометрия у кандидатов совпадает по построению критерия, а вот содержимое
    может отличаться: один и тот же участок часто снимали дважды в разные даты.
    Копия даёт ~1.0, повторный вылет — заметно меньше. Различать обязательно:
    для сплита оба случая одинаково опасны, но копия в обучающей смеси — это
    удвоенный вес территории, а повторный вылет — редкий и ценный
    кросс-датный материал.
    """
    import cv2
    import rasterio
    from rasters import to_rgb
    def thumb(path):
        with rasterio.open(path) as ds:
            k = max(ds.width, ds.height) / side
            arr = ds.read(out_shape=(ds.count, max(1, int(ds.height / k)),
                                     max(1, int(ds.width / k))))
        g = cv2.cvtColor(to_rgb(arr).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return cv2.resize(g, (side, side)).astype(np.float32)
    try:
        return float(cv2.matchTemplate(thumb(a), thumb(b), cv2.TM_CCOEFF_NORMED)[0, 0])
    except Exception:  # noqa: BLE001
        return float("nan")


#: Выше этого порога содержимое считается тождественным (копия файла).
COPY_NCC = 0.97


def modality_index(path="openaerialmap_dataset/scene_modality.csv",
                   fallback="open_orto/work/modality.csv"):
    """Вид содержимого площадки: photo / pan / relief.

    Берётся **проверенная глазами** разметка: автоматика различает цветное
    фото и серую картинку, но отличить панхром от карты рельефа с теневой
    отмывкой она не может — у отмывки текстура не хуже фотографической.
    """
    f = Path(path)
    if f.exists():
        return {r["scene"]: r["modality"] for r in csv.DictReader(f.open(encoding="utf-8"))}
    g = Path(fallback)
    if not g.exists():
        return {}
    out = {}
    for r in csv.DictReader(g.open(encoding="utf-8")):
        v = r["вердикт"]
        out[r["scene"]] = "photo" if v == "фото" else "gray?"
    return out


def relation(a: str, b: str, ncc: float, mod: dict) -> str:
    """Чем связаны две площадки одной группы.

    Три разных случая, и путать их нельзя. Копия — просто удвоенный вес
    территории. Повторный вылет — редкий кросс-датный материал. Разная
    модальность (фото против карты высот, цвет против панхрома) — вообще
    другой материал: пары из него учат не тому же самому.
    """
    ma, mb = mod.get(a, ""), mod.get(b, "")
    if "relief" in (ma, mb) and ma != mb:
        return "modality: фото ↔ рельеф"
    if ma == "relief" and mb == "relief":
        return "copy" if ncc >= COPY_NCC else "revisit: две карты рельефа"
    if ma != mb and {ma, mb} <= {"photo", "pan"}:
        return "modality: цвет ↔ панхром"
    return "copy" if ncc >= COPY_NCC else "revisit: другая дата"


def group(pairs, all_scenes):
    """Транзитивное замыкание: площадки-дубли собираются в одну группу."""
    parent = {s: s for s in all_scenes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _ in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups = defaultdict(list)
    for s in all_scenes:
        groups[find(s)].append(s)
    return groups


def overview(path: Path, side: int = 420):
    """Обзорная миниатюра растра — по ней человек сверяет тождество."""
    import cv2
    import rasterio
    from rasters import to_rgb
    try:
        with rasterio.open(path) as ds:
            k = max(ds.width, ds.height) / side
            w, h = max(1, int(ds.width / k)), max(1, int(ds.height / k))
            arr = ds.read(out_shape=(ds.count, h, w))
        rgb = to_rgb(arr)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buf).decode()
    except Exception as exc:  # noqa: BLE001
        print(f"  миниатюра {path.name}: {exc}")
        return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scan", default="open_orto/work/rasters_scan_all.csv")
    ap.add_argument("--manifest", default="openaerialmap_dataset/manifest.csv")
    ap.add_argument("--data-dir", default="E:/open_ortophoto_data")
    ap.add_argument("--check-sheet", action="store_true",
                    help="собрать контрольный лист миниатюр для сверки глазами")
    ap.add_argument("--sheet-out", default="openaerialmap_dataset/DUPLICATES.html")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    from cpu_affinity import pin_to_performance
    pin_to_performance(verbose=False)

    man = list(csv.DictReader(Path(args.manifest).open(encoding="utf-8")))
    used = {r["scene"] for r in man}
    pairs_by_scene = defaultdict(int)
    for r in man:
        pairs_by_scene[r["scene"]] += 1

    scan = {}
    for r in csv.DictReader(Path(args.scan).open(encoding="utf-8")):
        if r["name"] in used and r["lon"]:
            scan[r["name"]] = dict(lon=float(r["lon"]), lat=float(r["lat"]),
                                   km2=float(r["km2"]), res=float(r["res"]),
                                   w=int(r["w"]), h=int(r["h"]))
    print(f"использованных площадок: {len(used)}, с координатами: {len(scan)}")
    missing = used - set(scan)
    if missing:
        print(f"без координат: {len(missing)} — {sorted(missing)[:5]}")

    dup_pairs = find_pairs(scan)
    print("сверяю содержимое кандидатов...", flush=True)
    data = Path(args.data_dir)
    mod = modality_index()
    ncc = {(a, b): content_ncc(data / f"{a}.tif", data / f"{b}.tif")
           for a, b, _ in dup_pairs}
    rel = {k: relation(k[0], k[1], v, mod) for k, v in ncc.items()}
    groups = group(dup_pairs, sorted(used))
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"пар-кандидатов: {len(dup_pairs)} | групп с дублями: {len(multi)}")
    affected_scenes = sum(len(v) for v in multi.values())
    affected_pairs = sum(pairs_by_scene[s] for v in multi.values() for s in v)
    print(f"затронуто площадок: {affected_scenes}, пар: {affected_pairs}")
    n_copy = sum(1 for v in ncc.values() if v >= COPY_NCC)
    print(f"из них копий: {n_copy}, повторных вылетов (другая дата): "
          f"{len(dup_pairs) - n_copy}")
    print(f"{'площадка A':14} {'площадка B':14} {'сходство':>9}  что это")
    for (a, b), v in sorted(ncc.items(), key=lambda kv: -kv[1]):
        print(f"  {a[:12]:12} {b[:12]:12} {v:9.3f}  {rel[(a, b)]}"
              f"   [{mod.get(a, '?')} / {mod.get(b, '?')}]")
    print()
    for k, n in Counter(rel.values()).most_common():
        print(f"  {n:3}× {k}")

    if args.dry_run:
        return 0

    # ——— разметка манифеста
    gid = {}
    for i, (root, members) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1]))):
        # вид группы: копия хотя бы по одной связи — «copy», иначе «revisit»
        inner_rel = [r for (a, b), r in rel.items() if a in members and b in members]
        # приоритет: разная модальность важнее прочего — это другой материал
        kind = ("" if len(members) == 1
                else "modality" if any(r.startswith("modality") for r in inner_rel)
                else "copy" if any(r == "copy" for r in inner_rel) else "revisit")
        for s in members:
            gid[s] = (f"g{i:04d}", len(members), kind)
    head = list(man[0].keys())
    for col in ("dup_group", "dup_size", "dup_kind"):
        if col not in head:
            head.append(col)
    for r in man:
        g, n, k = gid.get(r["scene"], ("", 1, ""))
        r["dup_group"], r["dup_size"], r["dup_kind"] = g, n, k
    with Path(args.manifest).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=head)
        w.writeheader()
        w.writerows(man)
    print(f"манифест размечен: {len(man)} строк, колонки dup_group/dup_size")

    # ——— контрольный лист
    if args.check_sheet and multi:
        print("собираю контрольный лист...", flush=True)
        parts = [f"""<h1>Кандидаты в дубли площадок</h1>
<p class=lead>Найдено <b>{len(multi)}</b> групп ({affected_scenes} площадок,
{affected_pairs} пар). Критерий геометрический: центры ближе {NEAR_M:.0f} м,
площадь и разрешение совпадают в пределах {AREA_TOL:.0%}. Это <b>не</b>
доказательство тождества — сверьте миниатюры.</p>
<p class=note>Дубли не удаляются: разные рендеры одной территории остаются в
корпусе. Разметка нужна, чтобы они не разъехались по сплитам и не переоценили
эту территорию в обучающей смеси.</p>"""]
        for i, (root, members) in enumerate(
                sorted(multi.items(), key=lambda kv: -len(kv[1])), 1):
            g = gid[members[0]][0]
            kind = gid[members[0]][2]
            inner = [v for (a, b), v in ncc.items() if a in members and b in members]
            label = {"copy": "копия файла",
                     "modality": "другая модальность: фото против рельефа либо панхрома",
                     "revisit": "повторный вылет: та же территория, другая дата",
                     }.get(kind, kind)
            sim = f"сходство обзоров {min(inner):.2f}–{max(inner):.2f}" if inner else ""
            parts.append(f"<h2>Группа {g} — {len(members)} площадки · {label}</h2>"
                         f"<p class=dim>{sim}</p><div class=row>")
            for s in sorted(members):
                r = scan.get(s, {})
                img = overview(Path(args.data_dir) / f"{s}.tif")
                parts.append(
                    f'<figure><img src="data:image/jpeg;base64,{img}"/>'
                    f'<figcaption><code>{html.escape(s[:20])}</code><br>'
                    f'{mod.get(s, "?")} · {r.get("km2", 0):.2f} км² · {r.get("res", 0):.3f} м/пкс · '
                    f'{r.get("w", 0)}×{r.get("h", 0)} px<br>'
                    f'{pairs_by_scene[s]} пар в корпусе</figcaption></figure>')
            parts.append("</div>")
            if i % 5 == 0:
                print(f"  {i}/{len(multi)}", flush=True)
        out = Path(args.sheet_out)
        out.write_text(f"""<!doctype html><html lang=ru><meta charset=utf-8>
<title>Кандидаты в дубли площадок</title>
<style>body{{font:15.5px/1.6 Georgia,serif;max-width:1200px;margin:0 auto;padding:24px 18px 70px;color:#1a1a1a;background:#fcfcfb}}
h1{{font-size:27px;margin:0 0 8px}} h2{{margin-top:32px;border-top:1px solid #ddd;padding-top:14px;font-size:19px}}
.lead{{font-size:17px;color:#333}}
.note{{color:#444;font-size:14.5px;background:#f8f8f6;border-left:3px solid #c9c9c0;padding:10px 14px}}
.dim{{color:#777;font-size:13.5px;margin:2px 0 10px}}
.row{{display:flex;flex-wrap:wrap;gap:16px}}
figure{{margin:0}} img{{max-width:100%;border:1px solid #ccc;display:block}}
figcaption{{color:#666;font-size:13px;margin-top:5px;line-height:1.45}}
code{{background:#f0f0ec;padding:1px 4px;font-family:ui-monospace,monospace;font-size:12.5px}}</style>
{"".join(parts)}
</html>""", encoding="utf-8")
        print(f"контрольный лист: {out} ({out.stat().st_size/2**20:.1f} МБ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
