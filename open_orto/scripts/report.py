"""Отчёт по корпусу пар «орто ↔ подложка» (§7 задания).

Считает счётчики прогона, распределения осей и **метрику контроля
разметки**: остаток между кадром A и стороной B, натянутой по ``warp_ab``.
Две вещи, без которых отчёт врёт (§7):

1. гейт замера по высоте пика — на сплошном лесу и ровном поле корреляции
   не за что зацепиться, и она выдаёт «остаток 56 px» при нулевом пике;
   такой случай — отказ замера, а не ошибка разметки;
2. медиана всегда с числом замеров, при ``n < 20`` — помечена непоказательной.

    python open_orto/scripts/report.py --dataset open_orto/dataset \\
        --out open_orto/work/report.html
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rasters import gradient_map, phase_shift  # noqa: E402

PEAK_GATE = 0.004      # ниже — замер не состоялся (калибровка §7.1, см. отчёт)
GALLERY = 24


def load_pair(path: Path):
    d = np.load(path, allow_pickle=False)
    return {
        "image_a": cv2.cvtColor(cv2.imdecode(d["image_a_jpeg"], cv2.IMREAD_COLOR),
                                cv2.COLOR_BGR2RGB),
        "image_b": cv2.cvtColor(cv2.imdecode(d["image_b_jpeg"], cv2.IMREAD_COLOR),
                                cv2.COLOR_BGR2RGB),
        "warp": d["warp_ab"].astype(np.float32),
        "mask": d["mask_ab"].astype(bool),
        "meta": json.loads(str(d["meta"])),
    }


def warp_b_to_a(pair):
    """Сторона B, натянутая на геометрию кадра A по warp (для замера остатка)."""
    warp, mask = pair["warp"], pair["mask"]
    mx = np.nan_to_num(warp[..., 0], nan=-1).astype(np.float32)
    my = np.nan_to_num(warp[..., 1], nan=-1).astype(np.float32)
    out = cv2.remap(pair["image_b"], mx, my, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
    out[~mask] = 114
    return out


def homography_a_to_b(pair):
    """Гомография кадр A → кроп B, снятая с самой разметки.

    Сцена плоская (z = 0), поэтому связь сторон **строго проективная**, и
    четырёх точек хватило бы; берём сетку валидных точек и решаем МНК — так
    устойчивее к субпиксельному шуму warp.
    """
    warp, mask = pair["warp"], pair["mask"]
    h, w = mask.shape
    ys, xs = np.mgrid[0:h:16, 0:w:16]
    sel = mask[ys, xs]
    if sel.sum() < 12:
        return None
    src = np.stack([xs[sel], ys[sel]], axis=-1).astype(np.float32)
    dst = warp[ys[sel], xs[sel]].astype(np.float32)
    good = np.isfinite(dst).all(axis=-1)
    if good.sum() < 12:
        return None
    H, _ = cv2.findHomography(src[good], dst[good], 0)
    return H


def warp_a_to_b(pair):
    """Кадр A, натянутый на геометрию кропа B: (rgb, маска попадания).

    Накладываем **меньшее на большее** — так нагляднее: кроп подложки шире
    кадра, и видно, куда именно кадр лёг.
    """
    H = homography_a_to_b(pair)
    hb, wb = pair["image_b"].shape[:2]
    if H is None:
        return np.full((hb, wb, 3), 114, np.uint8), np.zeros((hb, wb), bool)
    a = pair["image_a"]
    on_b = cv2.warpPerspective(a, H, (wb, hb), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
    cover = cv2.warpPerspective(pair["mask"].astype(np.uint8) * 255, H, (wb, hb),
                                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0) > 127
    return on_b, cover


def residual_px(pair):
    """Остаток разметки: (сдвиг в px, высота пика). Гейт применяет вызывающий."""
    a = pair["image_a"]
    b_on_a = warp_b_to_a(pair)
    m = pair["mask"]
    if m.mean() < 0.2:
        return None, 0.0
    ga = gradient_map(a, m)
    gb = gradient_map(b_on_a, m)
    dx, dy, peak = phase_shift(ga, gb, max_shift_px=40)
    return math.hypot(dx, dy), peak


def panel(pair, note: str, *, max_width: int = 2000, tile: int = 96):
    """Панель: кадр A | кроп B | шахматка в геометрии B.

    Стороны кладутся в **натуральном размере** (без подгонки под общую
    высоту) — так виден реальный масштаб: кроп подложки крупнее кадра.
    В третьей колонке кадр наложен на подложку, а не наоборот.
    """
    a, b = pair["image_a"], pair["image_b"]
    a_on_b, cover = warp_a_to_b(pair)
    hb, wb = b.shape[:2]
    yy, xx = np.mgrid[0:hb, 0:wb]
    checker = (((yy // tile) + (xx // tile)) % 2).astype(bool) & cover
    mix = b.copy()
    mix[checker] = a_on_b[checker]
    # контур следа кадра, чтобы было видно, куда он лёг
    edges = cv2.morphologyEx(cover.astype(np.uint8), cv2.MORPH_GRADIENT,
                             np.ones((5, 5), np.uint8)).astype(bool)
    mix[edges] = (255, 220, 60)

    gap = 12
    ha, wa = a.shape[:2]
    H = max(ha, hb)
    W = wa + wb * 2 + gap * 2
    canvas = np.full((H, W, 3), 32, np.uint8)
    canvas[:ha, :wa] = a
    canvas[:hb, wa + gap: wa + gap + wb] = b
    canvas[:hb, wa + gap * 2 + wb:] = mix
    if W > max_width:
        k = max_width / W
        canvas = cv2.resize(canvas, (max_width, int(H * k)), interpolation=cv2.INTER_AREA)
    cv2.putText(canvas, note, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, note, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 82])
    import base64
    return base64.b64encode(buf).decode()


def auto_bins(values, n=6):
    """Бины по фактическому диапазону данных, с «красивым» шагом.

    Раньше границы были прописаны руками под конкретный диапазон параметров,
    и после его смены (высоты 250–400 → 175–300) гистограмма показывала
    почти пустоту: значения просто не попадали в бины. Теперь шкала следует
    за данными, а :func:`hist_svg` вдобавок печатает, сколько значений
    осталось вне бинов — молча потеряться они больше не могут.
    """
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return [(0.0, 1.0)]
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-9:
        pad = max(abs(lo) * 0.05, 0.5)
        lo, hi = lo - pad, hi + pad
    raw = (hi - lo) / n
    mag = 10.0 ** np.floor(np.log10(raw)) if raw > 0 else 1.0
    for mult in (1, 2, 2.5, 5, 10):
        step = mult * mag
        if raw <= step:
            break
    start = np.floor(lo / step) * step
    edges = [start + i * step for i in range(int(np.ceil((hi - start) / step)) + 1)]
    if len(edges) < 2:
        edges = [start, start + step]
    edges[-1] += step * 1e-6           # правый край включаем
    return list(zip(edges[:-1], edges[1:]))


def hist_svg(values, bins, label, fmt=lambda v: f"{v:g}"):
    if not len(values):
        return ""
    if bins is None:
        bins = auto_bins(values)
    counts = [int(((values >= lo) & (values < hi)).sum()) for lo, hi in bins]
    outside = int(len(values) - sum(counts))
    n = max(1, len(values))
    W, Hh, PL, PB, PT = 620, 200, 40, 34, 10
    pw, ph = W - PL - 10, Hh - PB - PT
    gw = pw / len(bins)
    top = max(1.0, max(counts) / n * 100 * 1.15)
    out = [f'<svg viewBox="0 0 {W} {Hh}">']
    for i, ((lo, hi), c) in enumerate(zip(bins, counts)):
        pct = c / n * 100
        bh = ph * pct / top
        x = PL + i * gw + gw * 0.15
        out.append(f'<rect x="{x:.1f}" y="{PT + ph - bh:.1f}" width="{gw*0.7:.1f}" '
                   f'height="{max(bh,1):.1f}" rx="3" fill="#2a78d6">'
                   f'<title>{fmt(lo)}–{fmt(hi)}: {c} ({pct:.1f}%)</title></rect>')
        out.append(f'<text x="{PL+(i+0.5)*gw:.0f}" y="{Hh-16}" text-anchor="middle" '
                   f'font-size="10.5" fill="#555">{fmt(lo)}</text>')
    out.append(f'<line x1="{PL}" y1="{PT+ph}" x2="{W-10}" y2="{PT+ph}" stroke="#999"/>')
    tail = f"{label} · n={len(values)}"
    if outside:
        tail += f" · ВНЕ ШКАЛЫ: {outside}"      # видимый сигнал о дефекте бинов
    out.append(f'<text x="{W/2:.0f}" y="{Hh-2}" text-anchor="middle" font-size="11" '
               f'fill="{"#c0392b" if outside else "#333"}">{tail}</text></svg>')
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="open_orto/dataset")
    ap.add_argument("--out", default="open_orto/work/report.html")
    ap.add_argument("--max-residual-checks", type=int, default=120)
    args = ap.parse_args()

    root = Path(args.dataset)
    files = sorted(root.glob("*.npz"))
    if not files:
        print("нет пар в", root)
        return 1
    print(f"пар: {len(files)}")

    metas, resid = [], []
    for i, f in enumerate(files):
        pair = load_pair(f)
        metas.append(pair["meta"])
        if i < args.max_residual_checks:
            r, peak = residual_px(pair)
            resid.append({"file": f.name, "resid": r, "peak": peak,
                          "kind": pair["meta"]["pair_kind"],
                          "layout": pair["meta"]["pair_layout"]})
        if (i + 1) % 50 == 0:
            print(f"  обработано {i+1}", flush=True)

    arr = lambda key: np.array([m[key] for m in metas], dtype=float)  # noqa: E731
    layouts = Counter(m["pair_layout"] for m in metas)
    kinds = Counter(m["pair_kind"] for m in metas)
    comp = Counter(m["compensation_src"] for m in metas)

    ok_meas = [r for r in resid if r["resid"] is not None and r["peak"] >= PEAK_GATE]
    ss = [r for r in ok_meas if r["kind"] == "same_source"]
    bm = [r for r in ok_meas if r["kind"] != "same_source"]
    failed = len(resid) - len(ok_meas)

    def med_line(rows, name):
        if not rows:
            return f"<li>{name}: <b>замеров нет</b></li>"
        v = np.array([r["resid"] for r in rows])
        warn = " <b>(n &lt; 20 — непоказательно)</b>" if len(v) < 20 else ""
        return (f"<li>{name}: медиана <b>{np.median(v):.2f} px</b>, p90 "
                f"{np.percentile(v, 90):.2f} px, n = {len(v)}{warn}</li>")

    # галерея: контрольные same-source, лучшие структурные, худшие по остатку
    gal = []
    by_res = sorted(ok_meas, key=lambda r: r["resid"])
    picks = ([("контроль same-source", r) for r in ss[:4]]
             + [("типичный", r) for r in by_res[len(by_res)//2: len(by_res)//2 + 8]]
             + [("худший по остатку", r) for r in by_res[-6:]])
    seen = set()
    for tag, r in picks:
        if r["file"] in seen or len(gal) >= GALLERY:
            continue
        seen.add(r["file"])
        pair = load_pair(root / r["file"])
        m = pair["meta"]
        note = (f"{tag} | {m['pair_layout']} | H={m['height_m']}м tilt={m['tilt_deg']}° "
                f"yaw A={m['yaw_deg']}° B={m.get('yaw_b_deg', 0)}° "
                f"(dyaw={m.get('delta_yaw_deg', 0)}°) scale={m['scale_ratio']} "
                f"covis={m['covis_frac']} | остаток {r['resid']:.2f} px "
                f"(пик {r['peak']:.4f})")
        gal.append(f'<figure><img src="data:image/jpeg;base64,{panel(pair, note)}"/>'
                   f'<figcaption>{note}</figcaption></figure>')

    sizes = np.array([ (root / m_file).stat().st_size for m_file in [f.name for f in files] ])
    html = f"""<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8">
<title>Корпус пар «орто ↔ подложка»</title>
<style>body{{font:15.5px/1.55 Georgia,serif;max-width:1000px;margin:0 auto;padding:24px 18px 70px}}
h1{{font-size:30px}} h2{{margin-top:1.8em;border-top:1px solid #e5e5e0;padding-top:.7em}}
table{{border-collapse:collapse;font-size:14px}} td,th{{border-bottom:1px solid #e5e5e0;padding:5px 10px}}
.num{{text-align:right;font-family:ui-monospace,monospace}}
figure{{margin:16px 0}} figure img{{width:100%;border:1px solid #ddd;border-radius:5px}}
figcaption{{font-size:12.5px;color:#555;margin-top:4px}}
code{{background:#f4f3ee;padding:1px 5px;border-radius:4px;font-size:.92em}}</style></head><body>
<h1>Корпус пар «ортофотоплан ↔ спутниковая подложка»</h1>
<p>Площадка <code>{metas[0]['scene']}</code> · подложка
{metas[0].get('basemap_provider') or '—'} · поле сдвигов
<code>{metas[0].get('shift_field')}</code> · сборка {metas[0].get('gen_date')},
коммит <code>{metas[0].get('gen_commit')}</code></p>

<h2>Счётчики прогона</h2>
<table><tbody>
<tr><td>пар в корпусе</td><td class="num">{len(files)}</td></tr>
<tr><td>компоновки</td><td class="num">{dict(layouts)}</td></tr>
<tr><td>тип пары</td><td class="num">{dict(kinds)}</td></tr>
<tr><td>источник компенсации</td><td class="num">{dict(comp)}</td></tr>
<tr><td>объём</td><td class="num">{sizes.sum()/1e6:.0f} МБ (медиана {np.median(sizes)/1e6:.2f} МБ/пара)</td></tr>
</tbody></table>

<h2>Контроль разметки</h2>
<p>Остаток — сдвиг между кадром A и стороной B, натянутой по <code>warp_ab</code>,
замеренный фазовой корреляцией по градиентным картам. Замеры с высотой пика
ниже {PEAK_GATE} считаются <b>несостоявшимися</b> (однородная фактура), а не
плохой разметкой: таких {failed} из {len(resid)}.</p>
<ul>
{med_line(ss, "контрольная ось same-source (разметка точна по построению)")}
{med_line(bm, "пары с подложкой")}
</ul>
<p>На контрольной оси остаток обязан быть около нуля — это единственная
проверка, отделяющая ошибку кода от расхождения самих данных. У пар с
подложкой к нему добавляется остаточная ошибка привязки и параллакс зданий:
подложка ортотрансформирована по своему DEM, и высокие объекты завалены в
другую сторону.</p>

<h2>Распределения осей</h2>
{hist_svg(arr('height_m'), None, 'высота съёмки, м')}
{hist_svg(arr('tilt_deg'), None, 'наклон камеры, °')}
{hist_svg(arr('yaw_deg'), None, 'курс кадра, °')}
{hist_svg(arr('delta_yaw_deg'), None, 'разница курсов A и B, °')}
{hist_svg(arr('scale_ratio'), None, 'масштаб GSD_A / GSD_B')}
{hist_svg(arr('covis_frac'), None, 'ко-видимость')}
{hist_svg(arr('footprint_b_m'), None, 'след кропа B, м')}
{hist_svg(arr('footprint_a_m'), None, 'след кадра A, м')}
{hist_svg(arr('gsd_a'), None, 'GSD кадра, м/пкс')}

<h2>Галерея</h2>
<p>Панель: <b>кадр A | кроп B | шахматка в геометрии B</b>. Стороны показаны в
натуральном размере, поэтому видно реальное соотношение: кроп подложки крупнее
кадра. В третьей колонке <b>кадр наложен на подложку</b> (меньшее на большее),
жёлтым обведён след кадра. Структуры обязаны продолжаться через границы клеток.</p>
{''.join(gal)}
</body></html>"""
    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(html, encoding="utf-8")
    print(f"отчёт: {dst} ({dst.stat().st_size/1e6:.1f} МБ)")
    if ss:
        v = np.array([r["resid"] for r in ss])
        print(f"контроль same-source: медиана {np.median(v):.3f} px, n={len(v)}")
    if bm:
        v = np.array([r["resid"] for r in bm])
        print(f"пары с подложкой: медиана {np.median(v):.2f} px, n={len(v)}")
    print(f"замеров не состоялось (пик < {PEAK_GATE}): {failed}/{len(resid)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
