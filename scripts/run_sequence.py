"""Последовательностный режим на реальной серии кадров: VO + EKF, дрейф vs GPS.

Визуальная одометрия кадр-к-кадру (LightGlue) с опциональными абсолютными
привязками, оценённая по GPS (ground truth). Между соседними кадрами дрона
appearance gap мал (та же камера, тот же полёт), поэтому VO работает — в отличие
от дрон↔спутник. Курс приводится к истинному северу (``--declination``): DJI
отдаёт магнитный курс, и без поправки траектория систематически дрейфует.

    python scripts/run_sequence.py --data for_binding/ufa --start 0 --count 19 --declination 14.5

Нужны: torch + LightGlue, Pillow. Сеть не требуется (подложка не тянется).
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from aero_geoloc.camera import Camera  # noqa: E402
from aero_geoloc.drone import load_drone_shot  # noqa: E402
from aero_geoloc.geo import haversine_m  # noqa: E402
from aero_geoloc.localize import normalize_gray  # noqa: E402
from aero_geoloc.matcher import create_matcher  # noqa: E402
from aero_geoloc.sequence import AbsoluteFix, EKFState, localize_sequence  # noqa: E402


def _enu(lat0, lon0, lat, lon):
    """Локальные ENU-метры точки относительно начала (east, north)."""
    east = haversine_m(lat0, lon0, lat0, lon) * np.sign(lon - lon0)
    north = haversine_m(lat0, lon0, lat, lon0) * np.sign(lat - lat0)
    return float(east), float(north)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="for_binding/ufa", help="каталог серии кадров")
    parser.add_argument("--start", type=int, default=0, help="индекс первого кадра в отсортированном списке")
    parser.add_argument("--count", type=int, default=19, help="сколько кадров взять (одна полоса без разворотов)")
    parser.add_argument("--matcher", default="lightglue", choices=["sift", "akaze", "lightglue", "loftr"])
    parser.add_argument("--width", type=int, default=1024, help="ширина даунсемпла кадра для VO")
    parser.add_argument("--declination", type=float, default=0.0, help="магнитное склонение, ° (истинный север)")
    parser.add_argument("--anchor-every", type=int, default=0, help="абсолютная GPS-привязка каждые N кадров (0 = чистый VO)")
    parser.add_argument("--plot", default="", help="сохранить график траектории в PNG")
    args = parser.parse_args()

    files = sorted(Path(args.data).glob("*.JPG"))[args.start : args.start + args.count]
    shots = []
    for f in files:
        shot = load_drone_shot(f, magnetic_declination_deg=args.declination)
        if shot.is_nadir:
            shots.append(shot)
    if len(shots) < 3:
        print("недостаточно надирных кадров")
        return 1

    lat0, lon0 = shots[0].true_lat, shots[0].true_lon
    truth = [_enu(lat0, lon0, s.true_lat, s.true_lon) for s in shots]
    altitude = float(np.mean([s.altitude_m for s in shots]))

    scale = args.width / shots[0].camera.image_width
    frames = [
        normalize_gray(cv2.resize(s.image_bgr, (args.width, round(s.camera.image_height * scale)),
                                  interpolation=cv2.INTER_AREA))
        for s in shots
    ]
    camera = Camera(frames[0].shape[1], frames[0].shape[0], fov_deg=shots[0].camera.fov_deg)
    init = EKFState(0.0, 0.0, shots[0].yaw_deg, np.diag([1.0, 1.0, 1.0]))

    fix_fn = None
    if args.anchor_every > 0:
        def fix_fn(i, _state):
            if i % args.anchor_every:
                return None
            return AbsoluteFix(truth[i][0], truth[i][1], shots[i].yaw_deg,
                               position_sigma_m=3.0, heading_sigma_deg=2.0)

    print(f"серия {len(shots)} кадров, alt≈{altitude:.0f}м, склонение {args.declination:+.1f}°, "
          f"матчер {args.matcher}, привязка каждые {args.anchor_every or '—'}")
    states = localize_sequence(frames, camera, init, altitude_m=altitude,
                               matcher=create_matcher(args.matcher), min_inliers=15,
                               absolute_fix_fn=fix_fn)

    errors = [float(np.hypot(s.east_m - t[0], s.north_m - t[1])) for s, t in zip(states, truth)]
    path_len = sum(np.hypot(truth[i + 1][0] - truth[i][0], truth[i + 1][1] - truth[i][1])
                   for i in range(len(truth) - 1))
    print(f"\nдлина пути {path_len:.0f} м")
    print(f"дрейф: финальный {errors[-1]:.1f} м ({100 * errors[-1] / path_len:.1f}% пути), "
          f"макс {max(errors):.1f} м, медиана {statistics.median(errors):.1f} м")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 6))
            gt = np.array(truth)
            vo = np.array([[s.east_m, s.north_m] for s in states])
            ax.plot(gt[:, 0], gt[:, 1], "o-", label="GPS (истина)", color="tab:green")
            ax.plot(vo[:, 0], vo[:, 1], "s-", label="VO+EKF", color="tab:red", alpha=0.8)
            ax.set_xlabel("East, м"); ax.set_ylabel("North, м"); ax.axis("equal"); ax.legend()
            ax.set_title(f"Траектория: дрейф {100 * errors[-1] / path_len:.1f}% пути")
            fig.savefig(args.plot, dpi=110, bbox_inches="tight")
            print(f"график → {args.plot}")
        except ImportError:
            print("matplotlib не установлен — график пропущен")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
