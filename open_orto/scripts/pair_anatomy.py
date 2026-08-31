"""Разбор одной обучающей пары: что лежит внутри `.npz` и как это читать.

Формат пары описан в README одной таблицей, но «плотное соответствие A→B с
маской» — то, что проще один раз увидеть, чем прочитать. Здесь берётся
конкретная пара и показывается каждый её массив: изображения, обе компоненты
поля соответствий, маска, метаданные — и главное, как они работают вместе.

Проверки, которые документ делает заодно: соответствие переносит точку A ровно
туда, где тот же объект лежит в B (сетка и трассировка точек), а маска
совпадает с областью, где соответствие определено.

    python open_orto/scripts/pair_anatomy.py --pair <имя>.npz
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

#: Пояснения к полям метаданных: без них JSON читается как набор чисел.
META_HELP = {
    "source": "как получена пара: orto_basemap — ортоплан против подложки",
    "scene": "идентификатор исходного ортофотоплана (площадки)",
    "pair_kind": "вид пары: orto_basemap или same_source (контрольная ось)",
    "pair_layout": "компоновка: inside — след кадра целиком внутри кропа B; partial — частичное перекрытие",
    "rectified": "выпрямлялся ли кадр (нет: борт снимает под наклоном, и это часть задачи)",
    "tilt_deg": "наклон оптической оси от надира, градусы",
    "tilt_az_deg": "азимут наклона: в какую сторону завалена камера",
    "yaw_deg": "курс кадра (сторона A), градусы",
    "yaw_b_deg": "курс кропа подложки (сторона B)",
    "delta_yaw_deg": "разница курсов сторон — сколько взаимного поворота видит матчер",
    "height_m": "высота съёмки виртуального борта, м",
    "fov_deg": "поле зрения камеры по горизонтали",
    "gsd_a": "размер пикселя стороны A на земле, м",
    "gsd_b": "размер пикселя стороны B на земле, м",
    "scale_ratio": "gsd_a / gsd_b: во сколько раз масштабы сторон различаются",
    "scale_planned": "какой масштаб планировался при выборе кропа",
    "b_px": "сторона квадратного кропа B в пикселях",
    "footprint_a_m": "ширина следа кадра A на земле, м",
    "footprint_b_m": "ширина кропа B на земле, м",
    "area_frac": "какая доля следа кадра попала внутрь кропа B",
    "covis_frac": "ко-видимость: доля пикселей A, у которых есть соответствие в B",
    "valid_a_frac": "доля кадра A, покрытая реальными данными ортоплана",
    "valid_b_frac": "доля кропа B, покрытая данными подложки",
    "anchor_xy": "координаты центра кропа B в проекции ортоплана, м",
    "basemap_provider": "источник подложки",
    "basemap_zoom": "зум тайлов, с которого читалась подложка",
    "compensation_m": "поправка привязки, внесённая в положение кропа B, м",
    "compensation_src": "откуда взята поправка: field — интерполяция поля сдвигов, cell_refine — прицельный замер ячейки, global — константа площадки",
    "shift_field": "файл поля сдвигов площадки",
    "season_a": "сезонная перекраска стороны A (pseudo_*) либо None",
    "season_b": "сезонная перекраска стороны B",
    "date_a": "дата съёмки стороны A, если известна",
    "date_b": "дата съёмки стороны B, если известна",
    "gen_commit": "коммит генератора: чем именно построена пара",
    "gen_date": "когда построена",
    "seed": "зерно генератора случайных чисел прогона",
    "index": "порядковый номер пары внутри прогона площадки",
    "cell_m": "сторона ячейки разбиения территории, м",
    "cell_xy": "центр ячейки, из которой взята пара",
}


def b64(img, quality=86, max_width=1400) -> str:
    if img.shape[1] > max_width:
        k = max_width / img.shape[1]
        img = cv2.resize(img, (max_width, int(img.shape[0] * k)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()


def fig(img, caption: str, **kw) -> str:
    return (f'<figure><img src="data:image/jpeg;base64,{b64(img, **kw)}"/>'
            f'<figcaption>{caption}</figcaption></figure>')


def colorize(chan: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Компонента поля соответствий в псевдоцвете; вне маски — серое."""
    out = np.zeros(chan.shape + (3,), np.uint8)
    if valid.any():
        lo, hi = np.nanpercentile(chan[valid], [1, 99])
        norm = np.clip((chan - lo) / max(hi - lo, 1e-6), 0, 1)
        # NaN вне маски при касте в uint8 дают мусор, а не нули: зануляем явно
        norm = np.nan_to_num(norm, nan=0.0)
        out = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    out[~valid] = (60, 60, 60)
    return out


def draw_grid(a, b, warp, mask, step=64):
    """Регулярная сетка в A и её образ в B: показывает саму геометрию пары."""
    ga, gb = a.copy(), b.copy()
    h, w = mask.shape
    ys = range(step // 2, h, step)
    xs = range(step // 2, w, step)
    for y in ys:
        for x in xs:
            if not mask[y, x]:
                continue
            u, v = warp[y, x]
            if not np.isfinite(u) or not np.isfinite(v):
                continue
            # цвет узла кодирует его положение в кадре: так видно, что сетка
            # не просто «легла куда-то», а сохранила взаимный порядок точек
            col = (int(255 * x / w), int(255 * y / h), 200)
            cv2.circle(ga, (x, y), 5, col, -1, cv2.LINE_AA)
            cv2.circle(ga, (x, y), 5, (0, 0, 0), 1, cv2.LINE_AA)
            if 0 <= u < b.shape[1] and 0 <= v < b.shape[0]:
                cv2.circle(gb, (int(u), int(v)), 5, col, -1, cv2.LINE_AA)
                cv2.circle(gb, (int(u), int(v)), 5, (0, 0, 0), 1, cv2.LINE_AA)
    return ga, gb


def side_by_side(left, right, gap=14):
    """Две стороны рядом в натуральном размере: ресайз ужимал бы метки на B."""
    h = max(left.shape[0], right.shape[0])
    w = left.shape[1] + right.shape[1] + gap
    out = np.full((h, w, 3), 250, np.uint8)
    # обе стороны по центру высоты: кадр ниже кропа, и прижатый к верху он
    # оставлял бы под собой пустую четверть картинки
    yl = (h - left.shape[0]) // 2
    yr = (h - right.shape[0]) // 2
    out[yl:yl + left.shape[0], :left.shape[1]] = left
    out[yr:yr + right.shape[0], left.shape[1] + gap:] = right
    return out


def trace_points(a, b, warp, mask, n=6, seed=3):
    """Несколько точек с подписями: A → B, чтобы проверить соответствие глазами."""
    rng = np.random.default_rng(seed)
    if not mask.any():
        return a.copy(), b.copy(), []
    # точки берутся по ячейкам сетки, а не случайно по всему кадру: случайная
    # выборка кучкуется, и половина меток налезает друг на друга
    h, w = mask.shape
    cols, rows_n = 3, 2
    picked = []
    for gy in range(rows_n):
        for gx in range(cols):
            y0, y1 = gy * h // rows_n, (gy + 1) * h // rows_n
            x0, x1 = gx * w // cols, (gx + 1) * w // cols
            cell = mask[y0:y1, x0:x1]
            if not cell.any():
                continue
            cy, cx = np.nonzero(cell)
            k = int(rng.integers(len(cx)))
            picked.append((int(cx[k]) + x0, int(cy[k]) + y0))
    ta, tb, rows = a.copy(), b.copy(), []
    for i, (x, y) in enumerate(picked[:n], 1):
        u, v = warp[y, x]
        if not np.isfinite(u) or not np.isfinite(v):
            continue
        col = (255, 210, 40)
        for img, (px, py) in ((ta, (x, y)), (tb, (int(u), int(v)))):
            cv2.drawMarker(img, (px, py), (0, 0, 0), cv2.MARKER_CROSS, 34, 5, cv2.LINE_AA)
            cv2.drawMarker(img, (px, py), col, cv2.MARKER_CROSS, 30, 2, cv2.LINE_AA)
            cv2.putText(img, str(i), (px + 12, py - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(img, str(i), (px + 12, py - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, col, 2, cv2.LINE_AA)
        rows.append((i, x, y, float(u), float(v)))
    return ta, tb, rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="openaerialmap_dataset")
    ap.add_argument("--pair", default="base_pair_0f2e9280b238_00002_partial.npz")
    ap.add_argument("--out", default="openaerialmap_dataset/PAIR_ANATOMY.html")
    args = ap.parse_args()
    from cpu_affinity import pin_to_performance
    pin_to_performance(verbose=False)

    path = Path(args.root) / args.pair
    d = np.load(path, allow_pickle=False)
    a = cv2.cvtColor(cv2.imdecode(d["image_a_jpeg"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    b = cv2.cvtColor(cv2.imdecode(d["image_b_jpeg"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    warp = d["warp_ab"].astype(np.float32)
    mask = d["mask_ab"].astype(bool)
    meta = json.loads(str(d["meta"]))
    finite = np.isfinite(warp).all(axis=-1)

    # ——— как массивы лежат в файле
    sizes = []
    with zipfile.ZipFile(path) as z:
        for i in z.infolist():
            sizes.append((i.filename.replace(".npy", ""), i.file_size, i.compress_size))
    total = path.stat().st_size

    parts = [f"""<h1>Анатомия обучающей пары</h1>
<p class=lead>Разбор файла <code>{html.escape(args.pair)}</code> — что лежит
внутри, что каждый массив значит и как они работают вместе. Формат общий для
всего датасета, так что по этому разбору читается любая пара.</p>

<h2>1. Файл целиком</h2>
<p>Пара — это <code>.npz</code>, то есть zip-архив с несколькими массивами
NumPy. Размер файла {total/2**20:.1f} МБ распределён так:</p>
<table><tr><th>массив<th>форма<th>тип<th>в файле<th>распакован<th>что это</tr>"""]

    what = {
        "image_a_jpeg": "сторона A — кадр виртуального борта, хранится готовым JPEG",
        "image_b_jpeg": "сторона B — кроп подложки, тоже JPEG",
        "warp_ab": "плотное соответствие: для каждого пикселя A его координата в B",
        "mask_ab": "маска ко-видимости: где соответствие определено",
        "meta": "параметры съёмки и происхождение, JSON-строка",
        "pinhole": "зарезервировано: у стороны B нет пинхольной камеры",
    }
    for name, raw, comp in sizes:
        arr = d[name]
        shape = "скаляр" if arr.shape == () else "×".join(str(x) for x in arr.shape)
        parts.append(f"<tr><td><code>{name}</code><td>{shape}<td>{arr.dtype}"
                     f"<td class=num>{comp/1024:.0f} КБ<td class=num>{raw/1024:.0f} КБ"
                     f"<td>{what.get(name, '')}</tr>")
    parts.append(f"""</table>
<p class=note>Изображения лежат <b>уже сжатыми</b> (JPEG в байтах), а не
массивами пикселей: так пара весит мегабайты вместо десятков. Поле
соответствий — <code>float16</code>: полтора пикселя точности ему не нужны,
шум самой разметки на парах с подложкой около 4 px, зато вдвое меньше объём.
Маска ужимается zip-ом в сотню раз ({[s for s in sizes if s[0]=='mask_ab'][0][1]/1024:.0f} КБ →
{[s for s in sizes if s[0]=='mask_ab'][0][2]/1024:.0f} КБ): она почти всюду
постоянна.</p>

<h2>2. Две стороны пары</h2>
<p>Сторона A — то, что «видит борт»: кадр {a.shape[1]}×{a.shape[0]}, снятый с
высоты {meta['height_m']} м под наклоном {meta['tilt_deg']}° и курсом
{meta['yaw_deg']}°. Сторона B — квадратный кроп подложки {b.shape[1]}×{b.shape[0]},
повёрнутый на свой курс {meta.get('yaw_b_deg', '—')}°; разница курсов —
{meta.get('delta_yaw_deg', '—')}°, и именно её матчер должен преодолеть.</p>""")
    parts.append(fig(a, f"<b>image_a_jpeg</b> — сторона A, {a.shape[1]}×{a.shape[0]} px, "
                     f"GSD {meta['gsd_a']} м/пкс, след на земле {meta['footprint_a_m']} м"))
    parts.append(fig(b, f"<b>image_b_jpeg</b> — сторона B, {b.shape[1]}×{b.shape[0]} px, "
                     f"GSD {meta['gsd_b']} м/пкс, охват {meta['footprint_b_m']} м. "
                     f"Кроп крупнее кадра: искать нужно <i>внутри</i> него"))

    # ——— warp
    parts.append(f"""<h2>3. Разметка: <code>warp_ab</code></h2>
<p>Главный массив пары. Форма {warp.shape[0]}×{warp.shape[1]}×2: для каждого
пикселя стороны A записаны <b>две координаты</b> — где этот же кусок земли
находится на стороне B. Не смещение, а именно абсолютная позиция в B.</p>
<pre><code>warp_ab[y, x] = [u, v]   # пиксель (x, y) кадра A ↔ пиксель (u, v) кропа B
</code></pre>
<p>Вне зоны ко-видимости стоит <code>NaN</code>: соответствия там нет, и это
явно записано, а не закодировано нулями. Доля <code>NaN</code> здесь —
{100*(1-finite.mean()):.1f} %.</p>""")
    parts.append(fig(colorize(warp[..., 0], finite),
                     "<b>warp_ab[..., 0]</b> — координата <i>u</i> (столбец в B). "
                     "Плавный градиент слева направо: соседние пиксели кадра ложатся "
                     "в соседние пиксели кропа. Серое — вне маски"))
    parts.append(fig(colorize(warp[..., 1], finite),
                     "<b>warp_ab[..., 1]</b> — координата <i>v</i> (строка в B). "
                     "Градиент идёт сверху вниз и наклонён: наклон — это и есть "
                     "взаимный поворот сторон"))

    # ——— маска
    mask_vis = np.zeros(mask.shape + (3,), np.uint8)
    mask_vis[mask] = (40, 175, 120)
    mask_vis[~mask] = (55, 55, 58)
    blend = cv2.addWeighted(a, 0.55, mask_vis, 0.45, 0)
    parts.append(f"""<h2>4. Маска ко-видимости: <code>mask_ab</code></h2>
<p>Маска {mask.shape[1]}×{mask.shape[0]}, хранится как <code>uint8</code>
(0 или 1) и читается приведением к <code>bool</code> — единица там, где у
пикселя A есть соответствие в B. Здесь покрыто {100*mask.mean():.1f} % кадра —
это и есть <code>covis_frac</code> = {meta['covis_frac']} из метаданных.
Обучение обязано считать функцию потерь только по маске: остальное — не ошибка
модели, а отсутствие данных.</p>""")
    parts.append(fig(blend, "<b>mask_ab</b> поверх кадра A: зелёное — соответствие "
                     "есть, тёмное — нет. У компоновки <code>partial</code> часть "
                     "следа кадра просто вышла за кроп подложки"))

    # ——— сетка и точки
    ga, gb = draw_grid(a, b, warp, mask)
    ta, tb, rows = trace_points(a, b, warp, mask)
    parts.append("""<h2>5. Как это работает вместе</h2>
<p>Регулярная сетка точек на кадре A и её образ в B по <code>warp_ab</code>.
Цвет узла кодирует его положение в кадре, поэтому видно не только «куда легло»,
но и что взаимный порядок точек сохранился — сетка повернулась и сжалась
целиком, а не рассыпалась.</p>""")
    parts.append(fig(side_by_side(ga, gb), "слева — узлы на стороне A, справа — те же "
                     "узлы, перенесённые в сторону B. Стороны в натуральном размере: "
                     "кроп подложки крупнее кадра", max_width=1900))
    parts.append("""<p>Та же проверка точечно: несколько случайных пикселей A и их
адреса в B. Достаточно посмотреть, что под каждой парой меток лежит один и тот
же объект.</p>""")
    parts.append(fig(side_by_side(ta, tb), "пронумерованные соответствия: метка N "
                     "слева и метка N справа — один и тот же кусок земли",
                     max_width=1900))
    parts.append("<table><tr><th>№<th>пиксель A (x, y)<th>→ пиксель B (u, v)</tr>")
    for i, x, y, u, v in rows:
        parts.append(f"<tr><td>{i}<td class=num>({x}, {y})<td class=num>({u:.1f}, {v:.1f})</tr>")
    parts.append("</table>")

    # ——— мета
    parts.append(f"""<h2>6. Метаданные: <code>meta</code></h2>
<p>JSON-строка с {len(meta)} полями — параметры съёмки, привязки и
происхождение пары. Происхождение здесь не формальность: по нему видно, каким
коммитом генератора построена пара и откуда взята поправка привязки.</p>
<table><tr><th>поле<th>значение<th>что означает</tr>""")
    for k, v in meta.items():
        parts.append(f"<tr><td><code>{html.escape(k)}</code>"
                     f"<td class=num>{html.escape(str(v))}"
                     f"<td>{html.escape(META_HELP.get(k, ''))}</tr>")
    parts.append("</table>")

    parts.append(f"""<h2>7. Как прочитать пару в коде</h2>
<pre><code>import cv2, json, numpy as np

d = np.load("{html.escape(args.pair)}", allow_pickle=False)

a = cv2.imdecode(d["image_a_jpeg"], cv2.IMREAD_COLOR)   # кадр борта
b = cv2.imdecode(d["image_b_jpeg"], cv2.IMREAD_COLOR)   # кроп подложки
warp = d["warp_ab"].astype(np.float32)                  # (H, W, 2), NaN вне маски
mask = d["mask_ab"].astype(bool)                        # (H, W)
meta = json.loads(str(d["meta"]))

# соответствия как список пар точек — то, что обычно нужно для обучения
ys, xs = np.nonzero(mask)
uv = warp[ys, xs]                       # координаты тех же точек в B
ok = np.isfinite(uv).all(axis=1)        # NaN отсеиваем даже внутри маски
pts_a = np.stack([xs[ok], ys[ok]], 1)
pts_b = uv[ok]
</code></pre>
<p class=note>Две вещи, о которых легко забыть. <b>Первая:</b> проверять
<code>isfinite</code> даже внутри маски — маска и конечность значений хранятся
отдельно, и надёжнее опираться на оба признака. <b>Вторая:</b> шум самой
разметки на парах с подложкой около <b>4 px</b> (геопривязка ортоплана, мозаика
подложки, параллакс зданий); порог инлайера 3 px на них лежит внутри этого шума,
и метрики надо читать на 5 и 10 px. На парах <code>same_source</code> разметка
точна до сотых пикселя.</p>

<h2>8. То же самое у контрольных пар</h2>
<p>У пар <code>same_source</code> (сторона B — второй вид того же ортоплана)
структура файла ровно та же: те же шесть массивов, те же правила чтения.
Отличается только происхождение стороны B и точность разметки — там она
аналитическая, до сотых пикселя, поэтому такие пары служат контрольной осью:
по ним видно, что геометрия рендера верна, независимо от того, как повела себя
подложка.</p>

<p style="color:#777;font-size:13px;margin-top:36px">Состав датасета —
<code>README.md</code>, как он получен — <code>METHODOLOGY.md</code>,
распределения и примеры — <code>SUMMARY.html</code>.</p>""")

    out = Path(args.out)
    out.write_text(f"""<!doctype html><html lang=ru><meta charset=utf-8>
<title>Анатомия обучающей пары</title>
<style>body{{font:15.5px/1.6 Georgia,serif;max-width:1080px;margin:0 auto;padding:24px 18px 70px;color:#1a1a1a;background:#fcfcfb}}
h1{{font-size:28px;margin:0 0 8px}} h2{{margin-top:36px;border-top:1px solid #ddd;padding-top:16px;font-size:22px}}
.lead{{font-size:17px;color:#333}}
table{{border-collapse:collapse;margin:14px 0;font-size:14.5px;width:100%}}
td,th{{border:1px solid #ddd;padding:6px 10px;text-align:left;vertical-align:top}}
th{{background:#f4f4f4}} td.num{{font-family:ui-monospace,monospace;font-size:13.5px;white-space:nowrap}}
.note{{color:#444;font-size:14.5px;background:#f8f8f6;border-left:3px solid #c9c9c0;padding:10px 14px;margin:14px 0}}
figure{{margin:20px 0}} img{{max-width:100%;border:1px solid #ccc;display:block}}
figcaption{{color:#666;font-size:13.5px;margin-top:6px}}
pre{{background:#f5f5f2;padding:12px 14px;border-radius:6px;overflow-x:auto;font-size:13.5px;line-height:1.5}}
code{{font-family:ui-monospace,SFMono-Regular,monospace}}
p code,td code,li code{{background:#f0f0ec;padding:1px 4px;border-radius:3px;font-size:13.5px}}</style>
{"".join(parts)}
</html>""", encoding="utf-8")
    print(f"разбор: {out} ({out.stat().st_size/2**20:.1f} МБ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
