"""Подробный HTML-отчёт по прогону матчеров на корпусе «орто ↔ подложка».

Собирает всё в один самодостаточный файл: методику, состав корпуса, таблицы
по разрезам, SVG-диаграммы и галерею — как именно ядро видит пару
(соответствия, разложенные на инлайеры и промахи по плотному GT).

    python open_orto/scripts/bench_report_html.py \\
        --csv open_orto/work/bench_pairs.csv --dataset open_orto/dataset \\
        --out open_orto/BENCH_PAIRS_REPORT.html
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[2]))

from bench_pairs import CONFIGS, load_pair  # noqa: E402

MATCHERS = ["loftr", "minima_loftr", "roma", "minima_roma", "romav2"]
#: Категориальные слоты палитры (светлая / тёмная тема) — по одному на ядро.
COLORS = {
    "loftr": ("#eda100", "#c98500"),
    "minima_loftr": ("#eb6834", "#d95926"),
    "roma": ("#2a78d6", "#3987e5"),
    "minima_roma": ("#1baf7a", "#199e70"),
    "romav2": ("#4a3aa7", "#9085e9"),
}
INLIER_PX = 5.0
GT_NOISE_PX = 3.2      # измеренный шум разметки на парах с подложкой


def num(r, k):
    try:
        return float(r[k])
    except (TypeError, ValueError):
        return np.nan


def med(rows, key):
    return float(np.nanmedian([num(r, key) for r in rows])) if rows else np.nan


def success(rows):
    return float(np.nanmean([1.0 if (num(r, "inl5_frac") or 0) >= 0.5 else 0.0
                             for r in rows])) if rows else np.nan


# --- диаграммы -----------------------------------------------------------------

def bar_chart(labels, series, *, title, ylabel, fmt="{:.2f}", ymax=None, height=250):
    """Группированные столбики: серия на ядро, 2px зазор, hover-подсказки."""
    W, PL, PB, PT = 720, 52, 46, 12
    pw, ph = W - PL - 14, height - PB - PT
    gw = pw / max(1, len(labels))
    n = len(series)
    bw = min(30.0, (gw - 12) / n)
    top = ymax or max(1e-9, max(max(v for v in s["values"] if np.isfinite(v))
                                for s in series) * 1.18)
    out = [f'<svg viewBox="0 0 {W} {height}" role="img" aria-label="{title}">']
    ticks = 4
    for t in range(ticks + 1):
        yv = top * t / ticks
        y = PT + ph * (1 - t / ticks)
        out.append(f'<line x1="{PL}" y1="{y:.1f}" x2="{W-14}" y2="{y:.1f}" '
                   f'class="grid"/>')
        out.append(f'<text x="{PL-8}" y="{y:.1f}" dy="4" text-anchor="end" '
                   f'class="tick">{fmt.format(yv)}</text>')
    for gi, lab in enumerate(labels):
        x0 = PL + gi * gw + (gw - bw * n - 2 * (n - 1)) / 2
        for si, s in enumerate(series):
            v = s["values"][gi]
            if not np.isfinite(v):
                continue
            h = ph * min(v / top, 1.0)
            x = x0 + si * (bw + 2)
            out.append(
                f'<rect x="{x:.1f}" y="{PT + ph - h:.1f}" width="{bw:.1f}" '
                f'height="{max(h, 1.2):.1f}" rx="4" fill="var(--c-{s["key"]})">'
                f'<title>{s["name"]} · {lab}: {fmt.format(v)}</title></rect>')
        out.append(f'<text x="{PL + (gi + 0.5) * gw:.0f}" y="{height - 26}" '
                   f'text-anchor="middle" class="axis">{lab}</text>')
    out.append(f'<line x1="{PL}" y1="{PT+ph:.1f}" x2="{W-14}" y2="{PT+ph:.1f}" class="axisline"/>')
    out.append(f'<text x="14" y="{PT + ph/2:.0f}" class="axis" transform="rotate(-90 14 {PT + ph/2:.0f})" '
               f'text-anchor="middle">{ylabel}</text>')
    out.append("</svg>")
    return "".join(out)


def legend(keys):
    items = "".join(
        f'<span class="lg"><span class="sw" style="background:var(--c-{k})"></span>{k}</span>'
        for k in keys)
    return f'<div class="legend">{items}</div>'


def scatter_speed_quality(stats):
    """Скорость против качества: точка на ядро, подписи прямые."""
    W, H, PL, PB, PT = 720, 300, 54, 46, 16
    pw, ph = W - PL - 120, H - PB - PT
    xs = [stats[m]["sec"] for m in MATCHERS]
    ys = [stats[m]["success"] for m in MATCHERS]
    xmax = max(xs) * 1.25
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="скорость против качества">']
    for t in range(5):
        y = PT + ph * (1 - t / 4)
        out.append(f'<line x1="{PL}" y1="{y:.1f}" x2="{PL+pw}" y2="{y:.1f}" class="grid"/>')
        out.append(f'<text x="{PL-8}" y="{y:.1f}" dy="4" text-anchor="end" class="tick">{t/4:.2f}</text>')
    for t in range(5):
        x = PL + pw * t / 4
        out.append(f'<text x="{x:.0f}" y="{H-24}" text-anchor="middle" class="tick">'
                   f'{xmax*t/4:.2f}</text>')
    for m in MATCHERS:
        x = PL + pw * stats[m]["sec"] / xmax
        y = PT + ph * (1 - stats[m]["success"])
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="var(--c-{m})" '
                   f'stroke="var(--surface)" stroke-width="2"><title>{m}: '
                   f'успех {stats[m]["success"]:.2f}, {stats[m]["sec"]:.2f} с/пара</title></circle>')
        out.append(f'<text x="{x+12:.1f}" y="{y+4:.1f}" class="pt">{m}</text>')
    out.append(f'<line x1="{PL}" y1="{PT+ph:.1f}" x2="{PL+pw}" y2="{PT+ph:.1f}" class="axisline"/>')
    out.append(f'<text x="{PL+pw/2:.0f}" y="{H-6}" text-anchor="middle" class="axis">'
               f'секунд на пару →</text>')
    out.append(f'<text x="14" y="{PT+ph/2:.0f}" class="axis" '
               f'transform="rotate(-90 14 {PT+ph/2:.0f})" text-anchor="middle">'
               f'доля пар, где ядро справилось →</text>')
    out.append("</svg>")
    return "".join(out)


# --- галерея: как ядро видит пару ----------------------------------------------

def draw_matches(pair, corr, *, max_lines=140, seed=0):
    """Панель «кадр | кроп» с линиями соответствий: зелёные — инлайеры (< 5 px
    от GT), красные — промахи. Линии прорежены, чтобы картинка читалась."""
    a = pair["image_a"].copy()
    b = pair["image_b"].copy()
    warp, mask = pair["warp"], pair["mask"]
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    H = max(ha, hb)
    canvas = np.full((H, wa + 24 + wb, 3), 28, np.uint8)
    canvas[:ha, :wa] = a
    canvas[:hb, wa + 24:] = b
    if len(corr) == 0:
        return canvas, 0, 0
    xi = np.clip(np.round(corr.pts_q[:, 0]).astype(int), 0, wa - 1)
    yi = np.clip(np.round(corr.pts_q[:, 1]).astype(int), 0, ha - 1)
    ok = mask[yi, xi]
    gt = warp[yi, xi]
    good = ok & np.isfinite(gt).all(axis=1)
    if good.sum() == 0:
        return canvas, 0, 0
    epe = np.hypot(corr.pts_r[good, 0] - gt[good, 0], corr.pts_r[good, 1] - gt[good, 1])
    idx = np.nonzero(good)[0]
    rng = np.random.default_rng(seed)
    pick = rng.permutation(len(idx))[:max_lines]
    n_in = int((epe < INLIER_PX).sum())
    for k in pick:
        i = idx[k]
        p1 = (int(round(corr.pts_q[i, 0])), int(round(corr.pts_q[i, 1])))
        p2 = (int(round(corr.pts_r[i, 0])) + wa + 24, int(round(corr.pts_r[i, 1])))
        inlier = epe[k] < INLIER_PX
        col = (60, 190, 110) if inlier else (70, 70, 235)   # BGR: зелёный / красный
        cv2.line(canvas, p1, p2, col, 1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 2, col, -1, cv2.LINE_AA)
        cv2.circle(canvas, p2, 2, col, -1, cv2.LINE_AA)
    return canvas, n_in, int(good.sum())


def jpeg_b64(img, quality=80, max_width=1900):
    if img.shape[1] > max_width:
        k = max_width / img.shape[1]
        img = cv2.resize(img, (max_width, int(img.shape[0] * k)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()


def build_gallery(dataset: Path, rows, matchers, n_pairs=3):
    """Для нескольких пар прогоняем ядра заново и рисуем их соответствия."""
    from aero_geoloc.matcher import create_matcher

    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["pair"], {})[r["matcher"]] = r
    # берём пары с подложкой: лучшую, типичную и трудную по roma
    cands = [(p, d) for p, d in by_pair.items()
             if d.get("roma") and d["roma"]["pair_kind"] == "orto_basemap"]
    cands.sort(key=lambda kv: num(kv[1]["roma"], "inl5_frac"))
    picks = []
    if cands:
        picks = [("трудная пара", cands[0][0]),
                 ("типичная пара", cands[len(cands) // 2][0]),
                 ("лёгкая пара", cands[-1][0])]
    ss = [(p, d) for p, d in by_pair.items()
          if d.get("roma") and d["roma"]["pair_kind"] == "same_source"]
    if ss:
        picks.append(("контроль same_source", ss[len(ss) // 2][0]))
    picks = picks[:n_pairs + 1]

    cards = []
    for name in matchers:
        matcher = create_matcher(name, **CONFIGS[name])
        for tag, pair_id in picks:
            path = dataset / f"{pair_id}.npz"
            if not path.exists():
                continue
            pair = load_pair(path)
            a = cv2.cvtColor(pair["image_a"], cv2.COLOR_BGR2GRAY)
            b = cv2.cvtColor(pair["image_b"], cv2.COLOR_BGR2GRAY)
            corr = matcher.match(a, b)
            img, n_in, n_tot = draw_matches(pair, corr)
            m = pair["meta"]
            frac = n_in / n_tot if n_tot else 0.0
            cards.append({
                "matcher": name, "tag": tag, "pair": pair_id,
                "b64": jpeg_b64(img),
                "note": (f"{name} · {tag} · {m['pair_layout']} · "
                         f"курс A={m['yaw_deg']}° B={m.get('yaw_b_deg','—')}° "
                         f"(Δ={m.get('delta_yaw_deg','—')}°) · наклон {m['tilt_deg']}° · "
                         f"H={m['height_m']} м · масштаб {m['scale_ratio']} · "
                         f"<b>{n_in} инлайеров из {n_tot}</b> ({frac:.0%})"),
            })
        del matcher
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        print(f"  галерея: {name} готов", flush=True)
    return cards, [p for _, p in picks]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", default="open_orto/work/bench_pairs.csv")
    ap.add_argument("--dataset", default="open_orto/dataset")
    ap.add_argument("--gallery-matchers", default="roma,minima_roma,romav2,minima_loftr")
    ap.add_argument("--out", default="open_orto/BENCH_PAIRS_REPORT.html")
    args = ap.parse_args()

    rows = list(csv.DictReader(Path(args.csv).open(encoding="utf-8")))
    bm = [r for r in rows if r["pair_kind"] == "orto_basemap"]
    ss = [r for r in rows if r["pair_kind"] == "same_source"]

    stats, stats_ss = {}, {}
    for m in MATCHERS:
        rs = [r for r in bm if r["matcher"] == m]
        rss = [r for r in ss if r["matcher"] == m]
        stats[m] = dict(epe=med(rs, "epe_med_px"), inl3=med(rs, "inl3_frac"),
                        inl5=med(rs, "inl5_frac"), inl10=med(rs, "inl10_frac"),
                        success=success(rs), npairs=med(rs, "n_pairs"),
                        sec=med(rs, "sec"), n=len(rs))
        stats_ss[m] = dict(epe=med(rss, "epe_med_px"), inl3=med(rss, "inl3_frac"),
                           inl5=med(rss, "inl5_frac"), success=success(rss),
                           npairs=med(rss, "n_pairs"), n=len(rss))

    # разрезы
    def cut(pred):
        return {m: dict(inl5=med([r for r in bm if r["matcher"] == m and pred(r)], "inl5_frac"),
                        epe=med([r for r in bm if r["matcher"] == m and pred(r)], "epe_med_px"),
                        success=success([r for r in bm if r["matcher"] == m and pred(r)]),
                        n=len([r for r in bm if r["matcher"] == m and pred(r)]))
                for m in MATCHERS}

    yaw_bins = [("0–8°", lambda r: abs(num(r, "delta_yaw_deg")) <= 8),
                ("8–16°", lambda r: 8 < abs(num(r, "delta_yaw_deg")) <= 16),
                ("16–25°", lambda r: abs(num(r, "delta_yaw_deg")) > 16)]
    yaw_cuts = [(lab, cut(pred)) for lab, pred in yaw_bins]
    tilt_bins = [("< 3°", lambda r: num(r, "tilt_deg") < 3),
                 ("3–7°", lambda r: 3 <= num(r, "tilt_deg") <= 7),
                 ("> 7°", lambda r: num(r, "tilt_deg") > 7)]
    tilt_cuts = [(lab, cut(pred)) for lab, pred in tilt_bins]
    layout_cuts = [("inside", cut(lambda r: r["layout"] == "inside")),
                   ("partial", cut(lambda r: r["layout"] == "partial"))]
    covis_cuts = [("≥ 0.95", cut(lambda r: num(r, "covis_frac") >= 0.95)),
                  ("0.8–0.95", cut(lambda r: 0.8 <= num(r, "covis_frac") < 0.95)),
                  ("< 0.8", cut(lambda r: num(r, "covis_frac") < 0.8))]

    print("строю галерею (прогон ядер на примерах)...", flush=True)
    gal_matchers = [m.strip() for m in args.gallery_matchers.split(",") if m.strip()]
    cards, picked = build_gallery(Path(args.dataset), rows, gal_matchers)

    # --- сборка HTML ---
    def table(stat_map, keys, headers, fmts):
        head = "".join(f"<th class='num'>{h}</th>" for h in headers)
        body = []
        best = {}
        lower_is_better = {"epe", "sec"}     # ошибка и время — чем меньше, тем лучше
        for k, f in zip(keys, fmts):
            vals = [stat_map[m][k] for m in MATCHERS if np.isfinite(stat_map[m][k])]
            if vals:
                best[k] = min(vals) if k in lower_is_better else max(vals)
        for m in MATCHERS:
            cells = []
            for k, f in zip(keys, fmts):
                v = stat_map[m][k]
                cls = "num best" if np.isfinite(v) and v == best.get(k) else "num"
                cells.append(f"<td class='{cls}'>{f.format(v)}</td>" if np.isfinite(v)
                             else "<td class='num'>—</td>")
            body.append(f"<tr><td><span class='sw' style='background:var(--c-{m})'></span>"
                        f"{m}</td>{''.join(cells)}</tr>")
        return (f"<table><thead><tr><th>ядро</th>{head}</tr></thead>"
                f"<tbody>{''.join(body)}</tbody></table>")

    def cut_table(cuts, title_col):
        head = "".join(f"<th class='num'>{lab}<br><span class='sub'>n={c[MATCHERS[0]]['n']}</span></th>"
                       for lab, c in cuts)
        body = []
        for m in MATCHERS:
            cells = "".join(f"<td class='num'>{c[m]['inl5']:.2f}<br>"
                            f"<span class='sub'>усп {c[m]['success']:.2f}</span></td>"
                            if np.isfinite(c[m]["inl5"]) else "<td class='num'>—</td>"
                            for _, c in cuts)
            body.append(f"<tr><td><span class='sw' style='background:var(--c-{m})'></span>"
                        f"{m}</td>{cells}</tr>")
        return (f"<table><thead><tr><th>{title_col}</th>{head}</tr></thead>"
                f"<tbody>{''.join(body)}</tbody></table>")

    series_main = [dict(key=m, name=m,
                        values=[stats[m]["inl5"], stats_ss[m]["inl5"]]) for m in MATCHERS]
    chart_main = bar_chart(["пары с подложкой", "контроль same_source"], series_main,
                           title="доля инлайеров", ylabel="доля инлайеров @5 px", ymax=1.05)
    series_yaw = [dict(key=m, name=m, values=[c[m]["inl5"] for _, c in yaw_cuts])
                  for m in MATCHERS]
    chart_yaw = bar_chart([lab for lab, _ in yaw_cuts], series_yaw,
                          title="инлайеры по разнице курсов",
                          ylabel="доля инлайеров @5 px", ymax=0.75)
    series_epe = [dict(key=m, name=m, values=[c[m]["epe"] for _, c in yaw_cuts])
                  for m in MATCHERS]
    chart_epe = bar_chart([lab for lab, _ in yaw_cuts], series_epe,
                          title="EPE по разнице курсов", ylabel="EPE, px", fmt="{:.0f}")

    gallery_html = "".join(
        f'<figure class="gal"><img src="data:image/jpeg;base64,{c["b64"]}" '
        f'alt="{c["matcher"]} на паре {c["pair"]}"/>'
        f'<figcaption>{c["note"]}</figcaption></figure>' for c in cards)

    css_vars = "\n".join(f"  --c-{m}: {COLORS[m][0]};" for m in MATCHERS)
    css_vars_dark = "\n".join(f"  --c-{m}: {COLORS[m][1]};" for m in MATCHERS)

    html = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Матчеры на парах «орто ↔ подложка»</title>
<style>
:root {{
  --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --ink-3: #86857c;
  --line: #e1e0d9; --note: #f5f4ef; --good: #1baf7a; --bad: #e34948;
{css_vars}
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --surface: #1a1a19; --ink: #fff; --ink-2: #c3c2b7; --ink-3: #8a897f;
    --line: #2c2c2a; --note: #232320;
{css_vars_dark}
  }}
}}
:root[data-theme="dark"] {{
  --surface: #1a1a19; --ink: #fff; --ink-2: #c3c2b7; --ink-3: #8a897f;
  --line: #2c2c2a; --note: #232320;
{css_vars_dark}
}}
* {{ box-sizing: border-box; }}
body {{ margin:0; background: var(--surface); color: var(--ink);
  font: 16px/1.6 Georgia, 'Times New Roman', serif; }}
.wrap {{ max-width: 1080px; margin: 0 auto; padding: 0 20px 90px; }}
h1 {{ font-size: 34px; margin: .3em 0 .2em; line-height: 1.15; }}
h2 {{ font-size: 25px; margin-top: 2.1em; border-top: 1px solid var(--line); padding-top: .8em; }}
h3 {{ font-size: 19px; margin-top: 1.6em; }}
h2 .num {{ color: var(--ink-3); font: 600 14px ui-monospace, monospace; margin-right: 10px; }}
.eyebrow {{ color: var(--ink-3); font-size: 13px; letter-spacing: .07em;
  text-transform: uppercase; margin-top: 30px; }}
.stand {{ font-size: 19px; color: var(--ink-2); max-width: 66ch; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14.5px; margin: 12px 0; }}
th, td {{ padding: 7px 10px; border-bottom: 1px solid var(--line); text-align: left; }}
th {{ color: var(--ink-3); font-weight: 600; font-size: 12.5px; vertical-align: bottom; }}
td.num, th.num {{ text-align: right; font-family: ui-monospace, monospace; font-size: 13.5px; }}
td.best {{ font-weight: 700; }}
.sub {{ color: var(--ink-3); font-size: 11px; font-family: ui-monospace, monospace; }}
.sw {{ display:inline-block; width:11px; height:11px; border-radius:3px; margin-right:7px;
  vertical-align: -1px; }}
.legend {{ display:flex; flex-wrap:wrap; gap:16px; font-size:13px; color: var(--ink-2);
  margin: 4px 0 2px; }}
.lg {{ display:inline-flex; align-items:center; }}
figure {{ margin: 18px 0; }}
figcaption {{ font-size: 13px; color: var(--ink-2); margin-top: 6px; }}
.gal img {{ width: 100%; border: 1px solid var(--line); border-radius: 6px; display:block; }}
.note {{ background: var(--note); border-radius: 9px; padding: 14px 18px; margin: 18px 0; }}
.note .tag {{ font-size: 11.5px; letter-spacing: .06em; text-transform: uppercase;
  color: var(--ink-3); display:block; margin-bottom: 4px; }}
code {{ font-family: ui-monospace, monospace; font-size: .9em; background: var(--note);
  padding: 1px 5px; border-radius: 4px; }}
svg {{ max-width: 100%; height: auto; display:block; }}
.grid {{ stroke: var(--line); stroke-width: 1; }}
.axisline {{ stroke: var(--ink-3); stroke-width: 1; }}
.tick {{ fill: var(--ink-3); font: 11px ui-monospace, monospace; }}
.axis {{ fill: var(--ink-2); font: 12px Georgia, serif; }}
.pt {{ fill: var(--ink); font: 600 12.5px Georgia, serif; }}
.tw {{ overflow-x: auto; }}
ul, ol {{ max-width: 70ch; }}
li {{ margin: .35em 0; }}
</style></head><body><div class="wrap">

<p class="eyebrow">aero-geoloc · open_orto · {date.today().isoformat()}</p>
<h1>Пять матчеров на боевом типе пары: вид с борта против спутниковой подложки</h1>
<p class="stand">160 пар с одной площадки (Санкт-Петербург, Обводный канал),
каждая прогнана пятью ядрами: {len(rows)} замеров. Корпус собран так, что у
каждой пары есть <b>плотная попиксельная разметка</b>, поэтому качество ядра
меряется напрямую — расстоянием между предсказанным соответствием и истиной,
без промежуточной оценки позы.</p>

<h2><span class="num">01</span>Коротко: что показал прогон</h2>
<ol>
<li><b>На парах с подложкой работает только плотная линия RoMa.</b> Доля пар,
где ядро в целом справилось: {stats['roma']['success']:.2f} у ванильной RoMa v1
и {stats['minima_roma']['success']:.2f} у MINIMA-RoMa против
{stats['minima_loftr']['success']:.2f} у MINIMA-LoFTR. Разрыв качественный:
полуплотное ядро почти не находит верных соответствий.</li>
<li><b>Ванильная RoMa v1 не хуже дообученной MINIMA</b> (EPE
{stats['roma']['epe']:.2f} px против {stats['minima_roma']['epe']:.2f}) — это
четвёртая независимая точка того же наблюдения после трека F и внешнего
бенчмарка OrthoLoC.</li>
<li><b>LoFTR-линия разваливается при взаимном повороте сторон.</b> При разнице
курсов больше 16° её EPE растёт с {yaw_cuts[0][1]['minima_loftr']['epe']:.1f} до
{yaw_cuts[2][1]['minima_loftr']['epe']:.1f} px, а RoMa-линия к повороту
безразлична. Практическое следствие: веер предповоротов в тракте обязателен
именно для полуплотного ядра.</li>
<li><b>Контрольная ось подтверждает разметку:</b> когда обе стороны берутся из
одного ортофотоплана, все пять ядер решают задачу почти идеально
(EPE {stats_ss['roma']['epe']:.2f}–{stats_ss['minima_loftr']['epe']:.2f} px).
Значит вся разница на подложке — это доменный разрыв, а не дефект данных.</li>
</ol>

<h2><span class="num">02</span>Методика: что именно меряется</h2>
<p>Матчер получает пару изображений и возвращает список соответствий «точка
кадра ↔ точка подложки» — обычно от нескольких сотен до двух тысяч. Для каждой
предсказанной точки кадра известно, куда она обязана попасть на подложке: это
записано в плотной разметке пары (<code>warp_ab</code>, координата для каждого
пикселя кадра). Все метрики считаются от расстояния между предсказанием и
истиной:</p>
<div class="tw"><table>
<thead><tr><th>метрика</th><th>что это</th><th>как читать</th></tr></thead>
<tbody>
<tr><td><b>EPE</b></td><td>медианная ошибка соответствия в пикселях кропа
подложки (1 px ≈ 0.45 м на земле при типичном масштабе корпуса)</td>
<td>меньше — лучше; 5 px ≈ 2 м</td></tr>
<tr><td><b>inl3 / inl5 / inl10</b></td><td>доля соответствий, легших ближе
3 / 5 / 10 px от истины; медиана по парам</td>
<td>больше — лучше; 0.50 = половина соответствий верна</td></tr>
<tr><td><b>успех</b></td><td>доля пар, где инлайеров @5 px не меньше
половины</td><td>ответ на вопрос «на скольких кадрах ядро справилось»</td></tr>
<tr><td><b>пар/кадр</b></td><td>сколько соответствий ядро вернуло</td>
<td>плотные ядра сэмплируют фиксированные ~2000</td></tr>
</tbody></table></div>

<div class="note"><span class="tag">Пол измеримости — важно для честного чтения</span>
<p>У самой разметки есть измеренный шум. На парах с подложкой остаточное
расхождение между кадром и натянутой на него подложкой — <b>≈ {GT_NOISE_PX} px</b>:
это остаточная ошибка привязки ортофотоплана плюс параллакс зданий (подложка
ортотрансформирована по своему рельефу, и высокие объекты завалены в другую
сторону). Поэтому EPE ниже ~3 px на таких парах недостижим физически, а порог
inl3 работает на грани шума — читать его надо вместе с inl10. На контрольной
оси <code>same_source</code> разметка точна (остаток 0.026 px), и там числа
отражают чистые возможности ядра.</p></div>

<h2><span class="num">03</span>Корпус</h2>
<p>Пары собраны генератором <code>open_orto/scripts/generate.py</code> из одного
ортофотоплана (GSD 9.4 см, UTM 36N) и тайлов Esri World Imagery. Сторона A —
кадр виртуального борта, отрендеренный лучами камеры на плоскость земли;
сторона B — кроп подложки, повёрнутый вслед за курсом кадра.</p>
<div class="tw"><table>
<thead><tr><th>параметр</th><th>значение</th></tr></thead><tbody>
<tr><td>камера</td><td>1024×576, f = 735 px, поле зрения 69.7° — как у тестового набора</td></tr>
<tr><td>высота съёмки</td><td>250–400 м → GSD кадра 0.34–0.54 м, след 348–558 м</td></tr>
<tr><td>курс кадра</td><td>произвольный (0–360°)</td></tr>
<tr><td>курс подложки</td><td>в пределах ±25° от курса кадра</td></tr>
<tr><td>наклон камеры</td><td>0–10° по случайному азимуту</td></tr>
<tr><td>масштаб сторон</td><td>0.85–1.20 (кадр к подложке)</td></tr>
<tr><td>компоновки</td><td>inside — след целиком в кропе; partial — перекрытие 0.60–0.90</td></tr>
<tr><td>контрольная ось</td><td>{len([r for r in ss if r['matcher']=='roma'])} пар same_source: сторона B из того же ортоплана</td></tr>
<tr><td>компенсация привязки</td><td>поле сдвигов, 44 узла, медиана сдвига 1.97 м</td></tr>
</tbody></table></div>

<h2><span class="num">04</span>Главный результат</h2>
{legend(MATCHERS)}
<figure>{chart_main}<figcaption>Доля инлайеров @5 px (медиана по парам). Слева —
{stats['roma']['n']} пар с подложкой Esri, справа — {stats_ss['roma']['n']} контрольных
пар same_source. На контроле все ядра почти идеальны; вся разница возникает
именно на подложке.</figcaption></figure>

<h3>Пары с подложкой Esri ({stats['roma']['n']} пар)</h3>
<div class="tw">{table(stats, ['epe','inl3','inl5','inl10','success','npairs','sec'],
                       ['EPE, px','inl3','inl5','inl10','успех','пар/кадр','с'],
                       ['{:.2f}','{:.2f}','{:.2f}','{:.2f}','{:.2f}','{:.0f}','{:.2f}'])}</div>
<p>Плотная линия RoMa отрывается от полуплотной не на проценты, а в разы:
успех {stats['roma']['success']:.2f} против {stats['loftr']['success']:.2f}.
При этом ядра LoFTR возвращают вчетверо меньше соответствий
({stats['minima_loftr']['npairs']:.0f} против {stats['roma']['npairs']:.0f}), и
большая их часть — промахи.</p>

<h3>Контроль same_source ({stats_ss['roma']['n']} пар)</h3>
<div class="tw">{table(stats_ss, ['epe','inl3','inl5','success','npairs'],
                       ['EPE, px','inl3','inl5','успех','пар/кадр'],
                       ['{:.2f}','{:.2f}','{:.2f}','{:.2f}','{:.0f}'])}</div>
<p>Здесь стороны различаются только геометрией — перспективой, поворотом и
масштабом, — и все ядра решают задачу практически идеально. Это двойная
проверка: подтверждает верность разметки корпуса и показывает, что дальше
измеряется именно доменный разрыв, а не сложность геометрии.</p>

<h2><span class="num">05</span>Разрезы</h2>

<h3>Разница курсов сторон — самый резкий разрез</h3>
{legend(MATCHERS)}
<figure>{chart_yaw}<figcaption>Доля инлайеров @5 px по разнице курсов кадра и
подложки. У LoFTR-линии она падает почти до нуля, у RoMa-линии не меняется.</figcaption></figure>
<figure>{chart_epe}<figcaption>Та же ось в единицах ошибки: EPE ядер LoFTR
растёт втрое, у плотных ядер остаётся на месте.</figcaption></figure>
<div class="tw">{cut_table(yaw_cuts, "разница курсов")}</div>
<p class="sub">В ячейках: доля инлайеров @5 px и доля пар, где ядро справилось.</p>
<div class="note"><span class="tag">Что из этого следует для тракта</span>
<p>Боевой тракт разворачивает карту района веером предповоротов — это дорого и
регулярно обсуждается как кандидат на экономию. Прогон показывает, чем именно
платят за отказ: <b>полуплотное ядро без предповорота не работает</b>, а
плотное прощает остаточную невязку курса до 25° без потерь. Значит экономить
на предповороте можно только вместе со сменой ядра на плотное — и это уже
вопрос бюджета времени, а не точности.</p></div>

<h3>Компоновка пары</h3>
<div class="tw">{cut_table(layout_cuts, "компоновка")}</div>
<p>Частичное перекрытие стоит плотной линии около 0.1 доли инлайеров —
умеренно и предсказуемо.</p>

<h3>Наклон камеры</h3>
<div class="tw">{cut_table(tilt_cuts, "наклон")}</div>
<p>Наклон до 10° влияет заметно слабее, чем доменный разрыв: рендер приводит
кадр к плоскости земли, и перспектива для ядра оказывается лёгкой частью
задачи.</p>

<h3>Ко-видимость</h3>
<div class="tw">{cut_table(covis_cuts, "ко-видимость")}</div>

<h3>Скорость против качества</h3>
<figure>{scatter_speed_quality(stats)}<figcaption>Медианное время на пару и доля
пар, где ядро справилось (пары с подложкой). LoFTR-линия быстрее плотной на
порядок — и на этом типе пары бесполезна.</figcaption></figure>

<h2><span class="num">06</span>Галерея: как ядро видит пару</h2>
<p>Слева кадр, справа кроп подложки, линии — соответствия, найденные ядром.
<b style="color:var(--good)">Зелёные</b> легли ближе {INLIER_PX:.0f} px от истины,
<b style="color:var(--bad)">красные</b> — промахи. Показана случайная выборка
до 140 линий, чтобы картинка читалась. Одни и те же пары прогнаны разными
ядрами — видно, что именно отличается.</p>
{gallery_html}

<h2><span class="num">07</span>Оговорки</h2>
<ul>
<li><b>Одна площадка, один сезон.</b> Числа характеризуют плотную городскую
застройку Санкт-Петербурга, а не домен вообще; разрезы по территории строить
не из чего.</li>
<li><b>Шум разметки ограничивает измеримость снизу</b> (≈ {GT_NOISE_PX} px):
ядра, различающиеся меньше чем на ~1 px EPE, этим прогоном не разделяются.</li>
<li><b>Пороги ядер — боевые калибровки</b> предыдущих треков, они не
подбирались под этот корпус. Перекалибровка может сдвинуть абсолютные числа,
но вряд ли порядок расслоения.</li>
<li><b>Контрольная ось мала</b> ({stats_ss['roma']['n']} пар): она нужна для
проверки разметки, а не для сравнения ядер между собой.</li>
<li><b>Параллакс зданий не снят.</b> Метрического DSM для площадки нет (в
исходных данных лежала цветовая визуализация, а не высоты), поэтому высокие
объекты дают систематический вклад в остаток.</li>
</ul>

<h2><span class="num">08</span>Воспроизведение</h2>
<p>Прогон докачивающий: повторный запуск досчитывает недостающие строки.</p>
<pre style="background:var(--note);padding:14px 16px;border-radius:8px;overflow-x:auto"><code>python open_orto/scripts/bench_pairs.py --dataset open_orto/dataset \\
    --matchers loftr,minima_loftr,roma,minima_roma,romav2 \\
    --out open_orto/work/bench_pairs.csv

python open_orto/scripts/bench_report_html.py \\
    --csv open_orto/work/bench_pairs.csv --dataset open_orto/dataset \\
    --out open_orto/BENCH_PAIRS_REPORT.html</code></pre>
<p style="color:var(--ink-3);font-size:13px;margin-top:36px">Сырьё:
<code>open_orto/work/bench_pairs.csv</code> ({len(rows)} строк). Сводные таблицы
в машинном виде — <code>open_orto/BENCH_PAIRS_METRICS.md</code>. Корпус и его
приёмка — <code>open_orto/RESULTS_BASEMAP_PILOT.md</code>.</p>
</div></body></html>"""

    dst = Path(args.out)
    dst.write_text(html, encoding="utf-8")
    print(f"отчёт: {dst} ({dst.stat().st_size/1e6:.1f} МБ), карточек в галерее: {len(cards)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
