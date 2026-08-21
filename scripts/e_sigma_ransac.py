"""Э-B: Σ⁻¹ RoMa v2 во взвешенном RANSAC — помогает ли ковариация пары позе.

Контекст (``docs/RESEARCH_A_ROMAV2_RECHECK.md`` §7, ``RESEARCH_A_PAPERS.md``
§1.2). RoMa v2 предсказывает на каждую пару матрицу точности 2×2 — обратную
ковариацию ошибки координаты. Авторы заводят её в скоринг MSAC внутри LO-RANSAC
и пост-рефайнмент и получают +21.5 AUC@1° на Hypersim — крупнейший одиночный
выигрыш их статьи. У нас матрица приходила наружу и сводилась к медиане следа.

Дизайн: на ОДНИХ И ТЕХ ЖЕ соответствиях v2 (и одних и тех же гипотезах —
общий сид) сравниваются три оценщика:

  cv2         штатный ``estimate_similarity`` (эталон реализации),
  msac        собственный MSAC без весов — контроль на «эффект самого оценщика»,
  msac_sigma  тот же MSAC, скоринг по Махаланобису с Σ⁻¹ + взвешенный перефит.

Режим свободный (масштаб 0.3–3, поворот без ограничений): вопрос эксперимента —
«сдвигает ли взвешивание ПОБЕДИТЕЛЯ к истине», в частности на DRZ-кейсах с
поворотным залипанием (+105°). Отдельно пишется, прошёл бы победитель гейты
пайплайна (масштаб 0.7–1.4, |поворот| ≤ 25°).

Про единицы. Абсолютный масштаб Σ⁻¹ в пикселях нашего входа не верифицирован
(матрицы предсказываются в координатах модельной сетки), поэтому матрицы
используются ОТНОСИТЕЛЬНО: нормируются на медианную по парам точность, а порог
инлайера задаётся эквивалентным пиксельным радиусом — у пары с медианной
точностью он совпадает с безвесовым 6 px. Абсолютная χ²-калибровка — отдельная
работа, здесь она не нужна: вопрос в относительном взвешивании.

    python scripts/e_sigma_ransac.py --out eval_out/e_sigma.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aero_geoloc.basemap import ESRI_WORLD_IMAGERY, TileBasemap, TileCache  # noqa: E402
from aero_geoloc.dataset import load_dataset  # noqa: E402
from aero_geoloc.geo import ground_mpp, haversine_m  # noqa: E402
from aero_geoloc.matcher import RoMaV2Matcher  # noqa: E402
from aero_geoloc.oracle import alignment_for, north_up_crop, offset_lonlat, to_gray  # noqa: E402
from aero_geoloc.pose import SimilarityTransform, estimate_similarity  # noqa: E402
from poses_provenance import PROVENANCE_FIELDS, PosesError, load_poses_with_provenance  # noqa: E402

#: Порог инлайера в эквивалентных пикселях (совпадает с ransac_threshold_px
#: пайплайна) и абсолютный колпак для взвешенного плеча: пара с крошечной
#: заявленной точностью не должна принимать километровые невязки — Σ⁻¹ обучена
#: только на ко-видимых участках с ‖r‖ < 8 px, для грубых выбросов она
#: бессмысленна (ограничение из статьи).
INLIER_PX = 6.0
ABS_CAP_PX = 12.0
MIN_INLIERS = 6

FIELDS = ["case", "status", "n_pairs", "prec_med_pos", "prec_med_false", "sec",
          *PROVENANCE_FIELDS]
for arm in ("cv2", "msac", "msac_sigma"):
    FIELDS += [f"{arm}_found", f"{arm}_inl", f"{arm}_scale", f"{arm}_rot",
               f"{arm}_err_m", f"{arm}_gate_ok"]


# --- взвешенный MSAC для подобия ---------------------------------------------


def _hypotheses(pts_q, pts_r, rng, iters):
    """Гипотезы подобия из минимальных наборов (2 пары), через комплексную форму."""
    n = len(pts_q)
    zq = pts_q[:, 0] + 1j * pts_q[:, 1]
    zr = pts_r[:, 0] + 1j * pts_r[:, 1]
    out = []
    idx = rng.integers(0, n, size=(iters, 2))
    for i, j in idx:
        if i == j:
            continue
        dq = zq[i] - zq[j]
        if abs(dq) < 1e-6:
            continue
        lam = (zr[i] - zr[j]) / dq          # s·e^{iθ}
        s = abs(lam)
        if not 0.05 < s < 20.0:
            continue
        t = zr[i] - lam * zq[i]
        out.append((lam.real, lam.imag, t.real, t.imag))
    return out


def _mahal2(params, pts_q, pts_r, W):
    """Квадрат Махаланобиса невязок для гипотезы ``(a, b, tx, ty)``."""
    a, b, tx, ty = params
    rx = pts_r[:, 0] - (a * pts_q[:, 0] - b * pts_q[:, 1] + tx)
    ry = pts_r[:, 1] - (b * pts_q[:, 0] + a * pts_q[:, 1] + ty)
    m2 = W[:, 0, 0] * rx * rx + 2.0 * W[:, 0, 1] * rx * ry + W[:, 1, 1] * ry * ry
    return m2, rx * rx + ry * ry


def _gls_fit(pts_q, pts_r, W):
    """Взвешенный МНК подобия: нормальные уравнения с попарными 2×2 весами."""
    n = len(pts_q)
    J = np.zeros((n, 2, 4))
    J[:, 0, 0] = pts_q[:, 0]; J[:, 0, 1] = -pts_q[:, 1]; J[:, 0, 2] = 1.0
    J[:, 1, 0] = pts_q[:, 1]; J[:, 1, 1] = pts_q[:, 0]; J[:, 1, 3] = 1.0
    WJ = np.einsum("nij,njk->nik", W, J)
    N = np.einsum("nji,njk->ik", J, WJ)
    rhs = np.einsum("nji,nj->i", WJ, pts_r)
    try:
        return np.linalg.solve(N, rhs)
    except np.linalg.LinAlgError:
        return None


def msac_similarity(pts_q, pts_r, W, hyps, *, tau=INLIER_PX ** 2):
    """MSAC по общему списку гипотез + два раунда взвешенного перефита."""
    best, best_score = None, np.inf
    for params in hyps:
        m2, r2 = _mahal2(params, pts_q, pts_r, W)
        score = np.minimum(m2, tau).sum()
        if score < best_score:
            best, best_score = params, score
    if best is None:
        return None
    params = np.asarray(best, float)
    for _ in range(2):
        m2, r2 = _mahal2(params, pts_q, pts_r, W)
        inl = (m2 <= tau) & (r2 <= ABS_CAP_PX ** 2)
        if inl.sum() < MIN_INLIERS:
            return None
        refit = _gls_fit(pts_q[inl], pts_r[inl], W[inl])
        if refit is None:
            return None
        params = refit
    m2, r2 = _mahal2(params, pts_q, pts_r, W)
    inl = (m2 <= tau) & (r2 <= ABS_CAP_PX ** 2)
    if inl.sum() < MIN_INLIERS:
        return None
    a, b, tx, ty = params
    matrix = np.array([[a, -b, tx], [b, a, ty]], float)
    return SimilarityTransform(matrix), int(inl.sum())


# --- прогон -------------------------------------------------------------------


def summarize(transform, n_inl, side, georef, align):
    if transform is None:
        return dict(found=0, inl="", scale="", rot="", err_m="", gate_ok="")
    c = transform.apply(np.array([(side - 1) / 2.0, (side - 1) / 2.0]))
    lon, lat = georef.pixel_to_lonlat(float(c[0]), float(c[1]))
    err = haversine_m(align.lat, align.lon, lat, lon)
    s, rot = transform.scale, transform.rotation_deg
    gate = (0.7 <= s <= 1.4) and (abs((rot + 180) % 360 - 180) <= 25.0)
    return dict(found=1, inl=n_inl, scale=round(s, 3), rot=round(rot, 1),
                err_m=round(err, 1), gate_ok=int(gate))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default="datasets/test_images.yaml")
    parser.add_argument("--cases", default="",
                        help="имена через запятую; пусто — все с оракульной позой")
    parser.add_argument("--offset-m", type=float, default=300.0,
                        help="сдвиг ложной пары для сигнала точности")
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--poses", default="eval_out/eval.csv")
    parser.add_argument("--allow-partial-poses", action="store_true",
                        help="работать при неполном файле поз (пропуски видны в CSV)")
    parser.add_argument("--cache", default="tiles")
    parser.add_argument("--out", default="eval_out/e_sigma.csv")
    args = parser.parse_args()

    dataset = load_dataset(args.manifest)
    if args.cases:
        cases = [dataset.by_name(n.strip()) for n in args.cases.split(",")]
    else:
        cases = dataset.cases
    required = {c.name for c in cases if not c.trust_yaw}
    try:
        poses, provenance = load_poses_with_provenance(
            args.poses, required=required, allow_partial=args.allow_partial_poses)
    except PosesError as exc:
        parser.error(str(exc))

    matcher = RoMaV2Matcher(keep_pair_precision=True)
    basemap = TileBasemap(cache=TileCache(args.cache))
    max_zoom = ESRI_WORLD_IMAGERY.max_zoom
    rng = np.random.default_rng(args.seed)

    rows = []
    for case in cases:
        align = alignment_for(case, poses)
        if align is None:
            row = {f: "" for f in FIELDS}
            row.update(case=case.name, status="skipped_no_pose", **provenance)
            rows.append(row)
            print(f"[{case.name}] пропуск: нет оракульной позы (в CSV — skipped_no_pose)")
            continue
        started = time.perf_counter()
        z_fine = case.basemap_zoom(max_zoom=max_zoom)
        frame, _ = case.frame_at_mpp(ground_mpp(case.prior.lat, z_fine))
        query = north_up_crop(frame, align.yaw_deg)
        side = query.shape[0]
        ref, georef = basemap(align.lon, align.lat, z_fine, side, side)
        corr = matcher.match(query, to_gray(ref))
        prec = matcher.last_precision_ab
        row = {f: "" for f in FIELDS}
        row.update(case=case.name, status="ok", **provenance, n_pairs=len(corr),
                   prec_med_pos=round(corr.evidence.get("precision_median", float("nan")), 4))

        if len(corr) >= MIN_INLIERS and prec is not None and len(prec) == len(corr):
            pts_q = corr.pts_q.astype(float)
            pts_r = corr.pts_r.astype(float)

            # cv2-эталон в том же свободном режиме
            pose = estimate_similarity(
                corr, ransac_threshold_px=INLIER_PX, min_inliers=MIN_INLIERS,
                scale_bounds=(0.3, 3.0), expected_rotation_deg=None)
            arm = summarize(pose.transform if pose else None,
                            pose.n_inliers if pose else 0, side, georef, align)
            row.update({f"cv2_{k}": v for k, v in arm.items()})

            # общие гипотезы для обоих кастомных плеч
            hyps = _hypotheses(pts_q, pts_r, rng, args.iters)

            W_id = np.tile(np.eye(2), (len(corr), 1, 1))
            res = msac_similarity(pts_q, pts_r, W_id, hyps)
            arm = summarize(*(res or (None, 0)), side, georef, align)
            row.update({f"msac_{k}": v for k, v in arm.items()})

            # Σ⁻¹, нормированная на медианную по парам точность (см. модуль-док):
            # у пары с медианной точностью порог совпадает с безвесовыми 6 px.
            W = prec.astype(float)
            per_axis = np.trace(W, axis1=1, axis2=2) / 2.0
            med = float(np.median(per_axis))
            if med > 0:
                res = msac_similarity(pts_q, pts_r, W / med, hyps)
                arm = summarize(*(res or (None, 0)), side, georef, align)
                row.update({f"msac_sigma_{k}": v for k, v in arm.items()})

        # ложная пара — только ради сигнала точности
        flat, flon = offset_lonlat(align.lat, align.lon, args.offset_m, 45.0)
        ref_f, _ = basemap(flon, flat, z_fine, side, side)
        corr_f = matcher.match(query, to_gray(ref_f))
        row["prec_med_false"] = round(
            corr_f.evidence.get("precision_median", float("nan")), 4)
        row["sec"] = round(time.perf_counter() - started, 1)
        rows.append(row)
        print(f"[{case.name}] пар={row['n_pairs']} | cv2: инл={row['cv2_inl'] or '—'} "
              f"err={row['cv2_err_m'] or '—'} rot={row['cv2_rot'] or '—'} | "
              f"msac: {row['msac_inl'] or '—'}/{row['msac_err_m'] or '—'} | "
              f"Σ⁻¹: {row['msac_sigma_inl'] or '—'}/{row['msac_sigma_err_m'] or '—'} "
              f"rot={row['msac_sigma_rot'] or '—'} | prec {row['prec_med_pos']} "
              f"vs {row['prec_med_false']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
