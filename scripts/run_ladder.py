"""Лестница возмущений внешнего вида — точка перелома матчера (``docs/TESTING.md``).

Свипует уровни L1–L4 по силе возмущения и меряет долю успешных локализаций и
медианную ошибку. «Точка перелома» — первая сила, где доля успехов падает ниже
порога. Единый протокол для всех матчеров = честное A/B по устойчивости к
appearance gap, а не по вкусу.

    python scripts/run_ladder.py --matcher sift
    python scripts/run_ladder.py --matcher akaze --yaw-step 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aero_geoloc.geo import haversine_m  # noqa: E402
from aero_geoloc.matcher import create_matcher  # noqa: E402
from aero_geoloc.testbench import (  # noqa: E402
    SampleSpec,
    default_camera,
    generate_sample,
    make_synthetic_scene,
)
from aero_geoloc.localize import localize_against_reference  # noqa: E402

LEVEL_NAMES = {1: "L1 экспозиция", 2: "L2 блюр/шум/JPEG", 3: "L3 спектр", 4: "L4 объекты"}


def cell(scene, camera, matcher, level, strength, yaws, reference_size):
    ok, errs = 0, []
    for yaw in yaws:
        spec = SampleSpec(yaw_deg=float(yaw), appearance_level=level, appearance_strength=strength)
        sample = generate_sample(scene, camera, spec, reference_size=reference_size)
        result = localize_against_reference(
            sample.query, camera, sample.prior, sample.reference, sample.reference_georef,
            matcher=matcher, refine=True,
        )
        if result.is_localized:
            ok += 1
            errs.append(
                haversine_m(sample.true_lat, sample.true_lon, result.center_lat, result.center_lon)
            )
    return ok / len(yaws), (float(np.median(errs)) if errs else float("nan"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matcher", default="sift", choices=["sift", "akaze", "lightglue", "loftr"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scene-size", type=int, default=2048)
    parser.add_argument("--frame-size", type=int, default=512)
    parser.add_argument("--yaw-step", type=float, default=30.0)
    parser.add_argument("--strengths", type=float, nargs="*", default=[0.5, 1.0, 1.5, 2.0, 2.5])
    parser.add_argument("--breakpoint-rate", type=float, default=0.75,
                        help="доля успехов, ниже которой уровень считается сломанным")
    args = parser.parse_args()

    scene = make_synthetic_scene(args.scene_size, seed=args.seed)
    camera = default_camera(args.frame_size)
    yaws = np.arange(0.0, 360.0, args.yaw_step)
    matcher = create_matcher(args.matcher)

    print(f"матчер: {args.matcher}, сцена {args.scene_size}px, кадр {args.frame_size}px, "
          f"{len(yaws)} курсов, порог перелома {args.breakpoint_rate:.0%}")
    header = "  сила | " + " | ".join(f"{LEVEL_NAMES[lv]:>16}" for lv in (1, 2, 3, 4))
    print("\n" + header)
    print("-" * len(header))

    rates = {lv: {} for lv in (1, 2, 3, 4)}
    for strength in args.strengths:
        cells = []
        for level in (1, 2, 3, 4):
            rate, med = cell(scene, camera, matcher, level, strength, yaws, args.frame_size * 2 + 256)
            rates[level][strength] = rate
            cells.append(f"{rate * 100:3.0f}% {med * 100:6.1f}см" if rate else f"{rate * 100:3.0f}%      —")
        print(f"  {strength:4.1f} | " + " | ".join(f"{c:>16}" for c in cells))

    print("\nточка перелома (первая сила с долей успехов ниже порога):")
    for level in (1, 2, 3, 4):
        broke = next((s for s in args.strengths if rates[level][s] < args.breakpoint_rate), None)
        print(f"  {LEVEL_NAMES[level]:>16}: {'сила ' + str(broke) if broke is not None else 'не сломался'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
