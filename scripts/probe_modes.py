"""Проба мод: верна ли матрица сходства S до усреднения (RESEARCH_F §3, И2).

Решающий тест эксперимента Э2: в оракульной паре истинное преобразование —
тождество по построению, значит патч ``m`` изображения A должен соответствовать
патчу ``m`` изображения B. Скрипт снимает с плотного ядра ДВЕ доли:

- ``argmax_hit_frac`` — знает ли правильный ответ сама матрица сходства
  (аргмакс по строке S попадает в свой патч, допуск ±1 патч);
- ``warp_hit_frac`` — доносит ли этот ответ финальный warp (после усреднения
  и регрессии).

Расхождение этих двух чисел и разделяет гипотезы Г-арх/Г-дан из
``docs/RESEARCH_F_BASE_CHOICE.md`` §3.6. Обе доли считаются ОДНИМ И ТЕМ ЖЕ
кодом для всех ядер (§4 спеки: разница не должна оказаться разницей в способе
счёта); ядра отличаются только тем, откуда берётся S:

- ``romav2`` — тензор ``attn_AB_logits`` из выхода модуля ``Matcher``
  (пакет кладёт его в preds своего forward, но верхний ``match()`` пересобирает
  словарь и отбрасывает — снимаем hook-ом);
- ``roma`` / ``minima_roma`` — якорная классификация ``gm_warp_or_cls``
  (B, 4096, H, W) из ``embedding_decoder`` на грубом уровне: v1 берёт от неё
  argmax (``cls_to_flow_refine``), то есть это и есть её «матрица сходства».

Имя и форма снятого тензора пишутся в CSV (``s_tensor_name``/``s_tensor_shape``)
— предполагать нельзя (§3.4 И1).

Скрипт **opt-in** и боевой тракт не трогает: геометрия пары повторяет
``probe_matcher.py``, сэмплирование пар повторяет соответствующий матчер.

    python scripts/probe_modes.py --cases DRZ_00755,DRZ_01018,DRZ_06498,Volgograd3 \\
        --matcher romav2 --out eval_out/modes_romav2.csv

Э2.2 (сужение окна, обе стороны согласованно): ``--side-px 1000``.
Э2.3 (рассогласование масштаба): ``--ref-zoom-offset -1`` (подложка на ступень
грубее) или ``+1`` (тоньше); истина тогда — чистое масштабирование вокруг
центра, и допуски пересчитываются тем же кодом.
``--dump-field каталог`` (И3) сохраняет поля warp/уверенности и карту argmax
в ``.npz`` на разбор глазами; в evidence массивы не кладутся.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aero_geoloc.basemap import ESRI_WORLD_IMAGERY, TileBasemap, TileCache  # noqa: E402
from aero_geoloc.dataset import load_dataset  # noqa: E402
from aero_geoloc.geo import ground_mpp, haversine_m  # noqa: E402
from aero_geoloc.oracle import alignment_for, north_up_crop, to_gray  # noqa: E402
from aero_geoloc.pose import estimate_similarity  # noqa: E402
from poses_provenance import PROVENANCE_FIELDS, PosesError, load_poses_with_provenance  # noqa: E402

FIELDS = [
    "case", "matcher", "pair_kind", "side_px", "side_ref_px", "ref_zoom_offset",
    "grid_n", "s_tensor_name", "s_tensor_shape",
    "argmax_hit_frac", "argmax_hit_frac_top10", "warp_hit_frac",
    "identity_frac_6px", "n_sampled",
    "best_wrong_consensus", "rot_deg", "scale", "err_m", "n_inliers",
    "sec", *PROVENANCE_FIELDS,
]

#: Порог «поза неверна» для ``best_wrong_consensus`` — тот же рубеж 50 м, что
#: и ``--correct-m`` стенда.
WRONG_POSE_M = 50.0


# --- Чистая геометрия: считается одним кодом для всех ядер (numpy, без torch) --

def expected_target_norm(grid_h: int, grid_w: int, scale_q2r: float,
                         side_q: int, side_r: int) -> np.ndarray:
    """Куда обязан попасть каждый патч A в нормированных координатах B.

    Патчи равномерно кроют вход ядра (обе картинки ресайзятся целиком), поэтому
    центр патча ``(i, j)`` лежит в исходных пикселях A на
    ``(j + 0.5) / grid_w * side_q``. Истина оракульной пары — масштабирование
    вокруг центров (тождество при ``scale_q2r == 1``): пиксельные центры по
    конвенции проекта ``(side - 1) / 2``.

    Возвращает (grid_h, grid_w, 2) — (x, y) в [-1, 1] координатах B с
    пиксель-центровой нормировкой ``x_norm = (2 * x_px + 1) / side - 1``.
    """
    jj, ii = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
    x_q = (jj + 0.5) / grid_w * side_q - 0.5     # исходные пиксели A (центр пикселя)
    y_q = (ii + 0.5) / grid_h * side_q - 0.5
    c_q = (side_q - 1) / 2.0
    c_r = (side_r - 1) / 2.0
    x_r = (x_q - c_q) * scale_q2r + c_r
    y_r = (y_q - c_q) * scale_q2r + c_r
    x_n = (2.0 * x_r + 1.0) / side_r - 1.0
    y_n = (2.0 * y_r + 1.0) / side_r - 1.0
    return np.stack([x_n, y_n], axis=-1)


def hit_frac(pred_norm: np.ndarray, expected_norm: np.ndarray,
             tol_norm: float, weights: np.ndarray | None = None) -> float:
    """Доля патчей, чей предсказанный таргет в пределах допуска от ожидаемого.

    Допуск — чебышёвский (по каждой оси), в нормированных единицах B;
    «±1 патч» спеки = ``2 / grid_n``. ``weights`` — маска патчей (для top-10%).
    """
    d = np.abs(pred_norm - expected_norm).max(axis=-1)
    ok = (d <= tol_norm).astype(np.float64)
    if weights is None:
        return float(ok.mean())
    w = weights.astype(np.float64)
    if w.sum() <= 0:
        return float("nan")
    return float((ok * w).sum() / w.sum())


def anchor_grid(n_anchors: int) -> np.ndarray:
    """Координаты якорей v1 в нормированных [-1, 1] B — зеркало формулы
    ``cls_to_flow_refine`` из ``romatch`` (G = linspace(-1+1/res, 1-1/res, res),
    стек в порядке (x, y)). Возвращает (n_anchors, 2)."""
    res = round(math.sqrt(n_anchors))
    if res * res != n_anchors:
        raise ValueError(f"число якорей {n_anchors} не квадрат")
    lin = np.linspace(-1 + 1 / res, 1 - 1 / res, res)
    gy, gx = np.meshgrid(lin, lin, indexing="ij")
    return np.stack([gx, gy], axis=-1).reshape(n_anchors, 2)


def top_frac_mask(score: np.ndarray, frac: float = 0.10) -> np.ndarray:
    """Маска ``frac`` доли патчей с наибольшим score (минимум один патч)."""
    k = max(1, int(round(score.size * frac)))
    thr = np.partition(score.reshape(-1), score.size - k)[score.size - k]
    return score >= thr


def consensus_row(corr_q: np.ndarray, corr_r: np.ndarray, conf: np.ndarray,
                  georef, side_q: int, truth_lat: float, truth_lon: float):
    """Свободный RANSAC (§9 отчёта v2: масштаб 0.3–3, поворот без ограничений)
    поверх сэмплированных пар; возвращает поля best_wrong_consensus/rot/scale/
    err_m/n_inliers."""
    from aero_geoloc.matcher import Correspondences

    out = {"best_wrong_consensus": "", "rot_deg": "", "scale": "",
           "err_m": "", "n_inliers": ""}
    if len(corr_q) < 3:
        return out
    corr = Correspondences(corr_q.astype(np.float32), corr_r.astype(np.float32),
                           conf.astype(np.float32))
    pose = estimate_similarity(
        corr, ransac_threshold_px=6.0, min_inliers=6,
        scale_bounds=(0.3, 3.0), expected_rotation_deg=0.0,
        rotation_tolerance_deg=180.0,
    )
    if pose is None:
        out["best_wrong_consensus"] = 0.0
        return out
    centre = ((side_q - 1) / 2.0, (side_q - 1) / 2.0)
    cx, cy = pose.transform.apply([centre])[0]
    lon, lat = georef.pixel_to_lonlat(float(cx), float(cy))
    err_m = haversine_m(truth_lat, truth_lon, lat, lon)
    wrong = err_m > WRONG_POSE_M
    out.update(
        best_wrong_consensus=round(pose.n_inliers / len(corr), 4) if wrong else 0.0,
        rot_deg=round(pose.transform.rotation_deg, 1),
        scale=round(pose.transform.scale, 3),
        err_m=round(err_m, 1), n_inliers=pose.n_inliers,
    )
    return out


# --- Снятие S и warp с конкретных ядер (единственное место, где они различны) --

def _capture_forward(module, store: dict, key: str):
    """forward hook: сохранить выход модуля под ``store[key]`` (последний вызов)."""
    def hook(_mod, _inp, out):
        store[key] = out
    return module.register_forward_hook(hook)


def run_romav2(matcher, query_gray, ref_gray):
    """RoMa v2: S = ``attn_AB_logits`` из модуля Matcher (см. шапку). Возвращает
    (S как (Ha,Wa,Hb,Wb) numpy, warp_norm (H,W,2), conf_field (H,W),
    pts_q, pts_r, conf_pairs, имя, форма)."""
    import cv2
    import torch
    from PIL import Image

    matcher._ensure()
    model = matcher._model
    inner = None
    for name, mod in model.named_modules():
        if type(mod).__name__ == "Matcher":
            inner = (name, mod)
    if inner is None:
        raise RuntimeError("в romav2 не найден модуль Matcher — сверить версию пакета")
    store: dict = {}
    handle = _capture_forward(inner[1], store, "out")
    try:
        with torch.inference_mode():
            im_q = Image.fromarray(cv2.cvtColor(query_gray, cv2.COLOR_GRAY2RGB))
            im_r = Image.fromarray(cv2.cvtColor(ref_gray, cv2.COLOR_GRAY2RGB))
            preds = model.match(im_q, im_r)
            matches, overlaps, _pab, _pba = model.sample(preds, matcher.max_samples)
            hq, wq = query_gray.shape[:2]
            hr, wr = ref_gray.shape[:2]
            pts_q, pts_r = model.to_pixel_coordinates(matches, hq, wq, hr, wr)
    finally:
        handle.remove()
    out = store.get("out")
    if not isinstance(out, dict) or "attn_AB_logits" not in out:
        raise RuntimeError(
            f"модуль {inner[0]} не отдал attn_AB_logits (ключи: "
            f"{sorted(out) if isinstance(out, dict) else type(out)}) — сверить пакет")
    s = out["attn_AB_logits"].detach().float()          # (B, Ha, Wa, Hb, Wb)
    s_name = f"{inner[0]}.attn_AB_logits"
    s_shape = tuple(s.shape)
    s = s[0].cpu().numpy()
    warp = preds["warp_AB"].detach().float()            # (..., H, W, 2) в [-1,1] B
    warp = (warp[0] if warp.ndim == 4 else warp).cpu().numpy()
    # Поле ко-видимости: канал 0 confidence_AB, сигмоида — как в боевом матчере.
    cf = preds["confidence_AB"].detach().float()
    cf = cf[0] if cf.ndim == 4 else cf
    conf_field = torch.sigmoid(cf[..., 0]).cpu().numpy()
    return (s, warp, conf_field,
            pts_q.detach().cpu().numpy(), pts_r.detach().cpu().numpy(),
            overlaps.detach().float().cpu().numpy().reshape(-1), s_name, s_shape)


def run_roma_v1(matcher, query_gray, ref_gray):
    """RoMa v1 / MINIMA: S = якорная классификация ``gm_warp_or_cls``
    (B, 4096, H, W) из ``embedding_decoder``. Приводим её к той же форме
    (Ha, Wa, n_anchor), что у v2 — дальше метрики считает общий код."""
    import cv2
    import torch
    from PIL import Image

    matcher._ensure()
    model = matcher._model
    target = None
    for name, mod in model.named_modules():
        if name.endswith("embedding_decoder"):
            target = (name, mod)
    if target is None:
        raise RuntimeError("в romatch не найден embedding_decoder — сверить версию")
    store: dict = {}
    handle = _capture_forward(target[1], store, "out")
    try:
        with torch.inference_mode():
            im_q = Image.fromarray(cv2.cvtColor(query_gray, cv2.COLOR_GRAY2RGB))
            im_r = Image.fromarray(cv2.cvtColor(ref_gray, cv2.COLOR_GRAY2RGB))
            warp, certainty = model.match(im_q, im_r, device=str(matcher._device))
            matches, conf = model.sample(warp, certainty, num=matcher.max_samples)
            hq, wq = query_gray.shape[:2]
            hr, wr = ref_gray.shape[:2]
            pts_q, pts_r = model.to_pixel_coordinates(matches, hq, wq, hr, wr)
    finally:
        handle.remove()
    out = store.get("out")
    cls = out[0] if isinstance(out, (tuple, list)) else out
    if not (hasattr(cls, "ndim") and cls.ndim == 4):
        raise RuntimeError(f"embedding_decoder отдал {type(out)} — сверить версию romatch")
    s_name = f"{target[0]}.gm_warp_or_cls"
    s_shape = tuple(cls.shape)
    n_anchor = cls.shape[1]
    if round(math.sqrt(n_anchor)) ** 2 != n_anchor:
        raise RuntimeError(
            f"каналы {s_shape} — не квадрат числа якорей: это не cls-режим, "
            "вывод argmax_hit_frac для этого ядра неопределён")
    cls_np = cls.detach().float()[0].cpu().numpy()       # (K, H, W)
    s = np.moveaxis(cls_np, 0, -1)                       # (Ha, Wa, K)
    # Плотный warp v1 в symmetric-режиме — (hs, 2*ws, 4): по ширине склеены
    # A→B и B→A (см. хвост romatch.match). Нам нужна левая половина, последние
    # 2 канала — координаты B в [-1, 1]. Certainty склеена так же.
    warp_np = warp.detach().float().cpu().numpy()
    cert_np = certainty.detach().float().cpu().numpy()
    if warp_np.ndim == 4:                                # PIL-вход romatch батчит: (1, hs, 2*ws, 4)
        warp_np = warp_np[0]
    cert_np = cert_np[0] if cert_np.ndim == 3 else cert_np
    ws = warp_np.shape[1] // 2
    if warp_np.shape[1] == 2 * warp_np.shape[0]:         # symmetric: ширина удвоена
        warp_ab = warp_np[:, :ws, 2:]
        cert_np = cert_np[:, :ws]
    else:
        warp_ab = warp_np[..., 2:]
    return (s, warp_ab, cert_np,
            pts_q.detach().cpu().numpy(), pts_r.detach().cpu().numpy(),
            conf.detach().float().cpu().numpy().reshape(-1), s_name, s_shape)


def argmax_targets(s: np.ndarray, side_r_grid_hw=None) -> np.ndarray:
    """Нормированные координаты argmax-таргета для каждого патча A.

    ``s`` — (Ha, Wa, Hb, Wb) (v2, патч↔патч) или (Ha, Wa, K) (v1, якоря).
    Возвращает (Ha, Wa, 2) в [-1, 1] координатах B. Формула позиций патчей B —
    центр патча, та же пиксель-центровая нормировка, что в
    :func:`expected_target_norm`; якоря v1 — :func:`anchor_grid` (зеркало
    ``cls_to_flow_refine``)."""
    if s.ndim == 4:
        ha, wa, hb, wb = s.shape
        flat = s.reshape(ha, wa, hb * wb)
        idx = flat.argmax(axis=-1)
        by, bx = np.divmod(idx, wb)
        x_n = (2.0 * (bx + 0.5) / wb) - 1.0
        y_n = (2.0 * (by + 0.5) / hb) - 1.0
        return np.stack([x_n, y_n], axis=-1)
    if s.ndim == 3:
        anchors = anchor_grid(s.shape[-1])
        idx = s.argmax(axis=-1)
        return anchors[idx]
    raise ValueError(f"неожиданная форма S: {s.shape}")


def pool_to_grid(field: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    """Среднее поле по патчам: (H, W) → (grid_h, grid_w). Общий код ранжирования
    top-10% для всех ядер."""
    h, w = field.shape
    ys = (np.arange(h) * grid_h // h).clip(0, grid_h - 1)
    xs = (np.arange(w) * grid_w // w).clip(0, grid_w - 1)
    out = np.zeros((grid_h, grid_w))
    cnt = np.zeros((grid_h, grid_w))
    np.add.at(out, (ys[:, None].repeat(w, 1), xs[None, :].repeat(h, 0)), field)
    np.add.at(cnt, (ys[:, None].repeat(w, 1), xs[None, :].repeat(h, 0)), 1.0)
    return out / np.maximum(cnt, 1.0)


def sample_field_at_patches(warp: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    """Значение плотного warp в центрах патчей: (H, W, 2) → (grid_h, grid_w, 2)."""
    h, w = warp.shape[:2]
    ys = ((np.arange(grid_h) + 0.5) / grid_h * h - 0.5).round().astype(int).clip(0, h - 1)
    xs = ((np.arange(grid_w) + 0.5) / grid_w * w - 0.5).round().astype(int).clip(0, w - 1)
    return warp[np.ix_(ys, xs)]


def central_crop(img: np.ndarray, side: int) -> np.ndarray:
    h, w = img.shape[:2]
    if side >= min(h, w):
        return img
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img[y0:y0 + side, x0:x0 + side]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default="datasets/test_images.yaml")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--matcher", required=True,
                        choices=["roma", "minima_roma", "romav2"])
    parser.add_argument("--side-px", type=int, default=0,
                        help="Э2.2: центральная обрезка ОБЕИХ сторон до этого "
                             "размера (0 = родной размер окна)")
    parser.add_argument("--ref-zoom-offset", type=int, default=0,
                        help="Э2.3: сдвиг зума подложки (−1 грубее, +1 тоньше); "
                             "истина становится масштабированием вокруг центра")
    parser.add_argument("--dump-field", default="",
                        help="И3: каталог для .npz с полями warp/уверенности")
    parser.add_argument("--poses", default="eval_out/eval.csv")
    parser.add_argument("--pose-tolerance-m", type=float, default=150.0)
    parser.add_argument("--allow-partial-poses", action="store_true")
    parser.add_argument("--cache", default="tiles")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from aero_geoloc.matcher import RoMaMatcher, RoMaV2Matcher

    dataset = load_dataset(args.manifest)
    cases = [dataset.by_name(n.strip()) for n in args.cases.split(",") if n.strip()]
    required = {c.name for c in cases if not c.trust_yaw}
    try:
        poses, provenance = load_poses_with_provenance(
            args.poses, required=required, allow_partial=args.allow_partial_poses)
    except PosesError as exc:
        parser.error(str(exc))

    if args.matcher == "romav2":
        matcher = RoMaV2Matcher()
        runner = run_romav2
    else:
        matcher = RoMaMatcher(checkpoint=None if args.matcher == "roma"
                              else "minima_roma")
        runner = run_roma_v1

    basemap = TileBasemap(cache=TileCache(args.cache))
    max_zoom = ESRI_WORLD_IMAGERY.max_zoom
    rows = []
    for case in cases:
        align = alignment_for(case, poses, tolerance_m=args.pose_tolerance_m)
        if align is None:
            rows.append({**{f: "" for f in FIELDS}, "case": case.name,
                         "matcher": args.matcher, "pair_kind": "skipped_no_pose",
                         **provenance})
            continue
        z_fine = case.basemap_zoom(max_zoom=max_zoom)
        mpp_q = ground_mpp(case.prior.lat, z_fine)
        frame, _ = case.frame_at_mpp(mpp_q)
        query = north_up_crop(frame, align.yaw_deg)
        if args.side_px:
            query = central_crop(query, args.side_px)
        side_q = query.shape[0]

        z_ref = z_fine + args.ref_zoom_offset
        mpp_r = ground_mpp(case.prior.lat, z_ref)
        scale_q2r = mpp_q / mpp_r
        side_r = int(round(side_q * scale_q2r))
        ref, georef = basemap(align.lon, align.lat, z_ref, side_r, side_r)
        gray_ref = to_gray(ref)

        started = time.perf_counter()
        (s, warp_ab, conf_field, pts_q, pts_r, conf_pairs,
         s_name, s_shape) = runner(matcher, query, gray_ref)
        sec = time.perf_counter() - started

        grid_h, grid_w = s.shape[:2]
        expected = expected_target_norm(grid_h, grid_w, scale_q2r, side_q, side_r)
        tol = 2.0 / max(grid_h, grid_w)                  # ±1 патч спеки
        pred_argmax = argmax_targets(s)
        top10 = top_frac_mask(pool_to_grid(conf_field, grid_h, grid_w))
        warp_at_patches = sample_field_at_patches(warp_ab, grid_h, grid_w)

        # identity_frac_6px: сэмплированные пары против истины в исходных px.
        c_q = (side_q - 1) / 2.0
        c_r = (side_r - 1) / 2.0
        exp_r = (pts_q - c_q) * scale_q2r + c_r
        ident = (float(np.mean(np.linalg.norm(pts_r - exp_r, axis=1) < 6.0))
                 if len(pts_q) else "")

        row = {f: "" for f in FIELDS}
        row.update(
            case=case.name, matcher=args.matcher, pair_kind="oracle",
            side_px=side_q, side_ref_px=side_r, ref_zoom_offset=args.ref_zoom_offset,
            grid_n=grid_h * grid_w, s_tensor_name=s_name, s_tensor_shape=str(s_shape),
            argmax_hit_frac=round(hit_frac(pred_argmax, expected, tol), 4),
            argmax_hit_frac_top10=round(hit_frac(pred_argmax, expected, tol, top10), 4),
            warp_hit_frac=round(hit_frac(warp_at_patches, expected, tol), 4),
            identity_frac_6px=ident if ident == "" else round(ident, 4),
            n_sampled=len(pts_q), sec=round(sec, 2), **provenance,
        )
        row.update(consensus_row(pts_q, pts_r, conf_pairs, georef, side_q,
                                 align.lat, align.lon))
        rows.append(row)
        print(f"[{case.name}] {args.matcher}: argmax_hit={row['argmax_hit_frac']} "
              f"warp_hit={row['warp_hit_frac']} identity6px={row['identity_frac_6px']} "
              f"S={s_name}{s_shape}", flush=True)

        if args.dump_field:
            dump_dir = Path(args.dump_field)
            dump_dir.mkdir(parents=True, exist_ok=True)
            tag = f"{case.name}_{args.matcher}_side{side_q}_dz{args.ref_zoom_offset}"
            np.savez_compressed(
                dump_dir / f"{tag}.npz",
                warp_ab=warp_ab.astype(np.float32),
                conf_field=conf_field.astype(np.float32),
                argmax_targets=pred_argmax.astype(np.float32),
                expected_targets=expected.astype(np.float32),
                pts_q=pts_q.astype(np.float32), pts_r=pts_r.astype(np.float32),
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"готово: {out} ({len(rows)} строк)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
