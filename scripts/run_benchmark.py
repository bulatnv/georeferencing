"""Полный прогон синтетического стенда — CLI поверх :mod:`aero_geoloc.testbench`.

Тесты гоняют урезанную сетку, чтобы укладываться в секунды; здесь сетка
плотная, и именно её результат считается измерением точности фазы.

Единый протокол для всех матчеров (``--matcher sift|akaze``) — это и есть
честное A/B за одним интерфейсом из ``docs/TESTING.md``.

    python scripts/run_benchmark.py --matcher sift --yaw-step 30
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aero_geoloc.matcher import create_matcher  # noqa: E402
from aero_geoloc.testbench import (  # noqa: E402
    default_camera,
    iter_specs,
    make_synthetic_scene,
    run_grid,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matcher", default="sift", choices=["sift", "akaze"])
    parser.add_argument("--seed", type=int, default=0, help="seed процедурной сцены")
    parser.add_argument("--scene-size", type=int, default=2048)
    parser.add_argument("--frame-size", type=int, default=512)
    parser.add_argument("--reference-size", type=int, default=1280)
    parser.add_argument("--yaw-step", type=float, default=30.0, help="шаг сетки по курсу, °")
    parser.add_argument(
        "--scale-steps", type=int, default=5, help="узлов масштаба в диапазоне 0.8..1.2"
    )
    parser.add_argument("--prior-offset-m", type=float, nargs="*", default=[0.0, 20.0])
    parser.add_argument(
        "--no-trust-yaw", action="store_true", help="не использовать yaw как ограничение"
    )
    args = parser.parse_args()

    scene = make_synthetic_scene(args.scene_size, seed=args.seed)
    camera = default_camera(args.frame_size)
    specs = list(
        iter_specs(
            yaw_deg=tuple(np.arange(0.0, 360.0, args.yaw_step)),
            altitude_ratio=tuple(np.linspace(0.8, 1.2, args.scale_steps)),
            prior_offset_m=tuple(args.prior_offset_m),
        )
    )

    print(f"матчер: {args.matcher}, сцена {args.scene_size}px seed={args.seed}, "
          f"кадр {args.frame_size}px, окно {args.reference_size}px")
    print(f"сетка: {len(specs)} примеров")

    started = time.perf_counter()
    summary = run_grid(
        scene,
        camera,
        specs,
        matcher=create_matcher(args.matcher),
        reference_size=args.reference_size,
        trust_yaw=not args.no_trust_yaw,
    )
    elapsed = time.perf_counter() - started

    print()
    print(summary.format_report())
    print(f"время:                  {elapsed:.1f} с ({elapsed / len(specs):.2f} с/пример)")

    failures = [m for m in summary.metrics if not m.localized]
    if failures:
        print(f"\nотказы ({len(failures)}):")
        for reason in sorted({m.reason or "?" for m in failures}):
            print(f"  {sum(m.reason == reason for m in failures):3d} × {reason}")

    return 0 if summary.success_rate == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
