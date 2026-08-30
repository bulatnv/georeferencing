"""Этап Р (§4 задания): поле сдвигов «ортофотоплан ↔ подложка».

Один раз на площадку меряем, как привязка растра смещена относительно
подложки, и сохраняем поле, из которого генератор берёт компенсацию.
Регистрация делается на native-данных (обе стороны north-up, один масштаб,
без перспективы) — надёжнее и дешевле, чем замер на каждом кропе.

**Семантика знака** (закреплена тестом и повторена здесь, потому что ошибка
знака даёт молча неверную разметку): точка земли с координатой ``g`` в сетке
ортоплана находится в подложке в точке ``g + (dx, dy)``.

    python open_orto/scripts/shift_field.py --raster <файл>.tif \\
        --step 300 --out open_orto/work/shift
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate_check import overview_mask  # noqa: E402
from rasters import BasemapSource, Grid, OrthoSource, gradient_map, phase_shift  # noqa: E402

#: Замер двухступенчатый — так решена главная беда городской фактуры:
#: на кропе ~150 м повторяющиеся структуры (пути, ряды домов) дают ложные
#: пики с разбросом ±17 м, а на ~300 м корреляция цепляется за уникальный
#: рисунок и даёт согласованные единицы метров (замерено на 10 узлах).
#: Грубая ступень ловит сдвиг на большом контексте, тонкая уточняет его до
#: субпикселя в узком радиусе — и расхождение ступеней служит гейтом качества.
#: Разрешения ступеней задаются **относительно фактического разрешения
#: подложки**, а не числом: 0.30/0.15 были выведены под Esri z19 (0.15 м/пкс)
#: в Санкт-Петербурге, и на площадках, где доступен только z18 (0.38 м/пкс),
#: тонкая ступень шла по вдвое апсемпленной подложке — субпиксельного пика
#: там нет, и контроль остатка не проходил ни в одном узле (замерено на
#: площадке 69faa4b8: 0 валидных из 14). Тот же класс ошибки, что «порог
#: чужой калибровки»: параметр, снятый на одних данных, перенесён на другие.
NODE_COARSE_FACTOR = 2.0   # грубая ступень: вдвое грубее подложки
NODE_FINE_FACTOR = 1.0     # тонкая: родное разрешение подложки
#: Размер узла задан **в метрах земли**, а не в пикселях. При фиксированных
#: 1024 px и подложке 0.38 м/пкс грубая ступень требовала кроп 776 м, который
#: на площадке 1.2 × 1.5 км почти всегда упирался в край съёмки: 21 узел из 25
#: браковался по «мало данных ортоплана». Теперь охват постоянен, а число
#: пикселей подстраивается под разрешение подложки.
NODE_COARSE_M = 400.0
NODE_FINE_M = 200.0
NODE_PX_RANGE = (384, 1536)
FINE_RADIUS_M = 3.0        # радиус уточнения вокруг грубого решения
MAX_STAGE_DIFF_M = 3.0     # расхождение ступеней — гейт качества замера
EROSION_M = 200.0      # отступ от края съёмки (§4.1)
MAX_SHIFT_M = 20.0     # потолок для пар с подложкой (§4.5)
MAX_RESID_M = 1.0      # контроль остатка (§4.4)
#: Согласие с соседями. Порог 3 м из задания выведен на парах «орто ↔ орто»,
#: где обе стороны в одной сетке; у пар с подложкой абсолютные сдвиги больше и
#: поле неоднороднее (замерено: медиана 2.4 м, p90 4.3 м), и порог 3 м выбивал
#: половину валидных узлов. Перекалибровано под этот тип пары — по регламенту
#: проекта «смена условий = смена калибровки порога», а не подтянуто по вкусу.
MAX_NEIGHBOUR_DIFF_M = 5.0
MIN_VALID_A = 0.90     # покрытие узла данными ортоплана


def step_for_nodes(ortho: OrthoSource, target: int, erosion_m: float = EROSION_M,
                   step_range: tuple[float, float] = (100.0, 1200.0)) -> float:
    """Шаг сетки, дающий около `target` узлов **в рабочей зоне**.

    Считать шаг по паспортной площади растра нельзя: узлы ставятся только
    там, где есть данные, а после эрозии края рабочая зона бывает втрое —
    вдесятеро меньше габарита (замерено: медиана 15 узлов там, где по
    площади ожидалось 60). Из-за этого площадки с нормальной долей валидных
    замеров (0.22) не набирали пяти узлов и уходили в отказ — не потому, что
    привязка не строится, а потому что её негде было мерить.
    """
    mask, _ = overview_mask(ortho, width=1500)
    b = ortho.bounds
    sx = (b.right - b.left) / mask.shape[1]
    er_px = max(1, int(erosion_m / sx))
    core = cv2.erode(mask.astype(np.uint8), np.ones((er_px, er_px), np.uint8)).astype(bool)
    area_m2 = float(core.mean()) * (b.right - b.left) * (b.top - b.bottom)
    if area_m2 <= 0 or target <= 0:
        return step_range[0]
    return float(np.clip((area_m2 / target) ** 0.5, *step_range))


def build_nodes(ortho: OrthoSource, step_m: float, erosion_m: float = EROSION_M):
    """Узлы сетки внутри рабочей зоны (валидные данные с эрозией)."""
    mask, _ = overview_mask(ortho, width=1500)
    b = ortho.bounds
    sx = (b.right - b.left) / mask.shape[1]
    er_px = max(1, int(erosion_m / sx))
    core = cv2.erode(mask.astype(np.uint8), np.ones((er_px, er_px), np.uint8)).astype(bool)
    xs = np.arange(b.left + erosion_m, b.right - erosion_m, step_m)
    ys = np.arange(b.bottom + erosion_m, b.top - erosion_m, step_m)
    nodes = []
    for gy in ys:
        for gx in xs:
            j = int((gx - b.left) / sx)
            i = int((b.top - gy) / ((b.top - b.bottom) / mask.shape[0]))
            if 0 <= i < core.shape[0] and 0 <= j < core.shape[1] and core[i, j]:
                nodes.append((float(gx), float(gy)))
    return nodes, float(core.mean())


def node_gsds(base):
    """Разрешения ступеней для этой подложки: (грубая, тонкая)."""
    mpp = base.min_mpp
    return NODE_COARSE_FACTOR * mpp, NODE_FINE_FACTOR * mpp


def node_px(size_m: float, gsd: float) -> int:
    """Сторона кропа узла в пикселях под заданный охват в метрах."""
    return int(np.clip(round(size_m / gsd), *NODE_PX_RANGE))


def _stage(ortho, base, gx, gy, gsd, *, radius_m, size_m, shift_m=(0.0, 0.0), zoom=None):
    """Одна ступень замера: (dx_m, dy_m, peak, valid_a, valid_b, zoom) либо None."""
    grid = Grid(x=gx, y=gy, size_px=node_px(size_m, gsd), gsd=gsd)
    a_rgb, a_val = ortho.read_grid(grid)
    va = float(a_val.mean())
    if va < MIN_VALID_A:
        return None, va, 0.0, zoom
    b_rgb, b_val, info = base.read_grid(grid, zoom=zoom, shift_m=shift_m)
    vb = float(b_val.mean())
    if vb < 0.90:
        return None, va, vb, info["zoom"]
    dx_px, dy_px, peak = phase_shift(gradient_map(a_rgb, a_val),
                                     gradient_map(b_rgb, b_val),
                                     max_shift_px=radius_m / gsd)
    # пиксели сетки → метры: +x пикселей = +X метров, +y пикселей = −Y метров
    return (dx_px * gsd, -dy_px * gsd, peak), va, vb, info["zoom"]


def measure_node(ortho, base, gx, gy):
    """Двухступенчатый замер в узле: грубая ступень + уточнение + контроль остатка."""
    rec = {"x": gx, "y": gy, "ok": False}

    coarse_gsd, fine_gsd = node_gsds(base)
    coarse, va, vb, zoom = _stage(ortho, base, gx, gy, coarse_gsd,
                                  radius_m=MAX_SHIFT_M, size_m=NODE_COARSE_M)
    rec.update(valid_a=va, valid_b=vb, zoom=zoom)
    if coarse is None:
        rec["reason"] = "мало данных ортоплана" if va < MIN_VALID_A else "дырявая подложка"
        return rec
    cdx, cdy, cpeak = coarse
    if math.hypot(cdx, cdy) > MAX_SHIFT_M:
        rec["reason"] = f"сдвиг {math.hypot(cdx, cdy):.1f} м вне потолка"
        return rec

    fine, va_f, vb_f, _ = _stage(ortho, base, gx, gy, fine_gsd, size_m=NODE_FINE_M,
                                 radius_m=FINE_RADIUS_M, shift_m=(cdx, cdy), zoom=zoom)
    if fine is None:
        rec["reason"] = "тонкая ступень без данных"
        return rec
    fdx, fdy, fpeak = fine
    stage_diff = math.hypot(fdx, fdy)
    dx_m, dy_m = cdx + fdx, cdy + fdy
    rec.update(dx=dx_m, dy=dy_m, shift_m=math.hypot(dx_m, dy_m),
               peak=cpeak, peak_fine=fpeak, stage_diff_m=stage_diff)
    if stage_diff > MAX_STAGE_DIFF_M:
        rec["reason"] = f"ступени разошлись на {stage_diff:.1f} м"
        return rec
    if rec["shift_m"] > MAX_SHIFT_M:
        rec["reason"] = f"сдвиг {rec['shift_m']:.1f} м вне потолка"
        return rec

    resid, _, _, _ = _stage(ortho, base, gx, gy, fine_gsd, size_m=NODE_FINE_M,
                            radius_m=FINE_RADIUS_M, shift_m=(dx_m, dy_m), zoom=zoom)
    if resid is None:
        rec["reason"] = "контроль остатка без данных"
        return rec
    rx, ry, peak2 = resid
    rec["resid_m"] = math.hypot(rx, ry)
    rec["peak2"] = peak2
    if rec["resid_m"] > MAX_RESID_M:
        rec["reason"] = f"остаток {rec['resid_m']:.2f} м — замер не сошёлся"
        return rec
    rec["ok"] = True
    rec["reason"] = "ok"
    return rec


def filter_by_neighbours(recs, radius_m=600.0, max_diff_m=None):
    """Согласие с медианой соседних валидных узлов ≤ MAX_NEIGHBOUR_DIFF_M (§4.5)."""
    good = [r for r in recs if r["ok"]]
    if len(good) < 3:
        return recs
    pts = np.array([[r["x"], r["y"]] for r in good])
    vecs = np.array([[r["dx"], r["dy"]] for r in good])
    for r in good:
        d = np.hypot(pts[:, 0] - r["x"], pts[:, 1] - r["y"])
        sel = (d <= radius_m) & (d > 1e-6)
        if sel.sum() < 2:
            continue
        med = np.median(vecs[sel], axis=0)
        if math.hypot(r["dx"] - med[0], r["dy"] - med[1]) > (max_diff_m or MAX_NEIGHBOUR_DIFF_M):
            r["ok"] = False
            r["reason"] = "расходится с соседями"
    return recs


def report_html(recs, meta, dst: Path):
    good = [r for r in recs if r["ok"]]
    reasons = {}
    for r in recs:
        if not r["ok"]:
            reasons[r.get("reason", "?")] = reasons.get(r.get("reason", "?"), 0) + 1
    svg = []
    if good:
        xs = np.array([r["x"] for r in good]); ys = np.array([r["y"] for r in good])
        x0, x1 = xs.min(), xs.max(); y0, y1 = ys.min(), ys.max()
        W, H = 820, max(260, int(820 * (y1 - y0 + 1) / (x1 - x0 + 1)))
        k = 18.0  # масштаб стрелок: метры сдвига → пиксели рисунка
        svg.append(f'<svg viewBox="0 0 {W} {H}" style="max-width:100%">')
        for r in good:
            px = (r["x"] - x0) / max(x1 - x0, 1) * (W - 40) + 20
            py = H - ((r["y"] - y0) / max(y1 - y0, 1) * (H - 40) + 20)
            svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.4" fill="#2a78d6"/>')
            svg.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{px + r["dx"]*k:.1f}" '
                       f'y2="{py - r["dy"]*k:.1f}" stroke="#eb6834" stroke-width="1.6"/>'
                       f'<title>{r["shift_m"]:.2f} м, пик {r["peak"]:.3f}</title>')
        svg.append("</svg>")
    rows = "".join(f"<tr><td>{k}</td><td class='num'>{v}</td></tr>" for k, v in
                   sorted(reasons.items(), key=lambda kv: -kv[1]))
    stat = ""
    if good:
        sh = np.array([r["shift_m"] for r in good])
        rs = np.array([r["resid_m"] for r in good])
        pk = np.array([r["peak"] for r in good])
        stat = (f"<p>Валидных узлов: <b>{len(good)}</b> из {len(recs)} "
                f"({100*len(good)//max(len(recs),1)}%). Сдвиг: медиана "
                f"<b>{np.median(sh):.2f} м</b> (p10 {np.percentile(sh,10):.2f}, "
                f"p90 {np.percentile(sh,90):.2f}). Остаток: медиана "
                f"{np.median(rs):.3f} м. Высота пика: медиана {np.median(pk):.4f}, "
                f"минимум {pk.min():.4f}.</p>"
                f"<p>Глобальная константа-фолбэк: dx={meta['global_dx']:.2f} м, "
                f"dy={meta['global_dy']:.2f} м.</p>")
    html = f"""<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8">
<title>Поле сдвигов: {meta['raster']}</title>
<style>body{{font:15px/1.5 Georgia,serif;max-width:900px;margin:24px auto;padding:0 16px}}
table{{border-collapse:collapse}}td,th{{border-bottom:1px solid #ddd;padding:4px 10px}}
.num{{text-align:right;font-family:ui-monospace,monospace}}</style></head><body>
<h1>Поле сдвигов «орто ↔ подложка»</h1>
<p>Растр <code>{meta['raster']}</code> · шаг сетки {meta['step_m']} м ·
узел {NODE_COARSE_M:.0f}/{NODE_FINE_M:.0f} м, ступени {meta.get("coarse_gsd", "?")}/{meta.get("fine_gsd", "?")} м/пкс ·
подложка {meta['provider']} zoom {meta.get('zoom','—')}</p>
{stat}
<h2>Стрелочная карта поля</h2>
<p>Точка — узел, стрелка — направление и величина сдвига (×{18} к масштабу карты).</p>
{''.join(svg)}
<h2>Причины отбраковки</h2>
<table><thead><tr><th>причина</th><th class="num">узлов</th></tr></thead><tbody>{rows}</tbody></table>
<p style="color:#777">Семантика знака: точка земли с координатой g в сетке ортоплана
находится в подложке в точке g + (dx, dy).</p>
</body></html>"""
    dst.write_text(html, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raster", required=True)
    ap.add_argument("--step", type=float, default=0.0,
                    help="шаг сетки узлов, м (0 — подобрать под --target-nodes "
                         "по площади рабочей зоны)")
    ap.add_argument("--target-nodes", type=int, default=60,
                    help="сколько узлов заказывать, когда шаг не задан явно")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--erosion-m", type=float, default=EROSION_M,
                    help="отступ от края съёмки, м")
    ap.add_argument("--neighbour-diff", type=float, default=MAX_NEIGHBOUR_DIFF_M,
                    help="порог согласия с соседями, м")
    ap.add_argument("--out", default="open_orto/work/shift")
    args = ap.parse_args()
    from cpu_affinity import pin_to_performance
    pin_to_performance(verbose=False)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ortho = OrthoSource(args.raster)
    base = BasemapSource(ortho)
    step = args.step or step_for_nodes(ortho, args.target_nodes, args.erosion_m)
    nodes, core_frac = build_nodes(ortho, step, erosion_m=args.erosion_m)
    if args.limit:
        nodes = nodes[: args.limit]
    print(f"рабочая зона (обзорно): {core_frac:.3f} площади | "
          f"шаг {step:.0f} м | узлов: {len(nodes)}")

    recs = []
    for i, (gx, gy) in enumerate(nodes, 1):
        recs.append(measure_node(ortho, base, gx, gy))
        if i % 10 == 0 or i == len(nodes):
            ok = sum(1 for r in recs if r["ok"])
            print(f"  {i}/{len(nodes)} узлов, валидных {ok}", flush=True)
    recs = filter_by_neighbours(recs, max_diff_m=args.neighbour_diff)

    good = [r for r in recs if r["ok"]]
    if not good:
        print("НИ ОДНОГО валидного узла — компенсацию строить не из чего")
        return 1
    gdx = float(np.median([r["dx"] for r in good]))
    gdy = float(np.median([r["dy"] for r in good]))
    stem = Path(args.raster).stem
    np.savez_compressed(
        out / f"shift_field_{stem}.npz",
        x=np.array([r["x"] for r in good]), y=np.array([r["y"] for r in good]),
        dx=np.array([r["dx"] for r in good]), dy=np.array([r["dy"] for r in good]),
        peak=np.array([r["peak"] for r in good]),
        resid=np.array([r["resid_m"] for r in good]),
        # все узлы с результатом замера — чтобы перефильтровать порогом
        # соседей без повторного прогона (замер стоит минуты на узел)
        all_x=np.array([r["x"] for r in recs if "dx" in r]),
        all_y=np.array([r["y"] for r in recs if "dx" in r]),
        all_dx=np.array([r["dx"] for r in recs if "dx" in r]),
        all_dy=np.array([r["dy"] for r in recs if "dx" in r]),
        all_ok=np.array([bool(r["ok"]) for r in recs if "dx" in r]),
        global_dx=gdx, global_dy=gdy,
        meta=np.str_(json.dumps({"raster": stem, "step_m": step,
                                 "node_coarse_m": NODE_COARSE_M,
                                 "node_fine_m": NODE_FINE_M,
                                 "coarse_gsd": round(node_gsds(base)[0], 3),
                                 "fine_gsd": round(node_gsds(base)[1], 3),
                                 "provider": "esri_world_imagery"}, ensure_ascii=False)))
    cg, fg = node_gsds(base)
    meta = {"raster": stem, "step_m": step, "provider": "Esri World Imagery",
            "coarse_gsd": round(cg, 3), "fine_gsd": round(fg, 3),
            "global_dx": gdx, "global_dy": gdy,
            "zoom": good[0].get("zoom")}
    report_html(recs, meta, out / f"shift_field_{stem}.html")
    sh = np.array([r["shift_m"] for r in good])
    print(f"ИТОГ: валидных {len(good)}/{len(recs)} | сдвиг медиана {np.median(sh):.2f} м | "
          f"константа ({gdx:.2f}, {gdy:.2f}) м")
    print(f"поле: {out / f'shift_field_{stem}.npz'} | отчёт: {out / f'shift_field_{stem}.html'}")
    ortho.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
