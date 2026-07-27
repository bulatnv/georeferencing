"""Проверка окружения: всё ли нужное установлено и **работает**.

Импорт пакета ещё не значит, что ядро поднимется: веса тянутся из сети, сборки
OpenCV затирают друг друга, torch может собраться без CUDA. Скрипт проверяет
каждый слой по факту, а не по наличию в `pip list`.

    python scripts/check_env.py              # быстрая проверка, без сети
    python scripts/check_env.py --weights    # + поднять веса (первый раз долго)

Код возврата: 0 — можно работать, 1 — есть блокирующая проблема.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OK, WARN, FAIL = "[+]", "[!]", "[-]"


class Report:
    """Копит строки отчёта и помнит, была ли блокирующая проблема."""

    def __init__(self) -> None:
        self.blocked = False

    def line(self, mark: str, name: str, detail: str) -> None:
        if mark == FAIL:
            self.blocked = True
        print(f"{mark} {name:22s} {detail}")


def _version(module: str) -> str:
    try:
        return str(getattr(importlib.import_module(module), "__version__", "?"))
    except Exception:
        return "?"


def check_core(rep: Report) -> None:
    print("\n── Ядро (без него пакет не импортируется) ──")
    for mod, pkg in (("numpy", "numpy"), ("cv2", "opencv-contrib-python")):
        if importlib.util.find_spec(mod) is None:
            rep.line(FAIL, pkg, "НЕТ → pip install -r requirements.txt")
        else:
            rep.line(OK, pkg, _version(mod))

    # Ловушка: две сборки OpenCV затирают друг друга в одном каталоге cv2.
    try:
        from importlib.metadata import distributions

        builds = sorted(
            d.metadata["Name"]
            for d in distributions()
            if (d.metadata["Name"] or "").lower() in ("opencv-python", "opencv-contrib-python")
        )
    except Exception:
        builds = []
    if len(builds) > 1:
        rep.line(
            FAIL,
            "сборки OpenCV",
            f"их {len(builds)}: {', '.join(builds)} — затирают друг друга!\n"
            "                       pip uninstall -y opencv-python opencv-contrib-python\n"
            "                       pip install opencv-contrib-python",
        )
    elif builds:
        rep.line(OK, "сборки OpenCV", f"одна: {builds[0]}")

    try:
        from aero_geoloc.matcher import AKAZEMatcher, SIFTMatcher

        SIFTMatcher()
        AKAZEMatcher()
        rep.line(OK, "SIFT / AKAZE", "оба детектора создаются")
    except Exception as exc:
        rep.line(FAIL, "SIFT / AKAZE", f"{type(exc).__name__}: {str(exc).splitlines()[0]}")


def check_real(rep: Report) -> None:
    print("\n── Реальные снимки ──")
    if importlib.util.find_spec("PIL") is None:
        rep.line(FAIL, "Pillow", "НЕТ → EXIF/XMP бортовых снимков не прочитать")
    else:
        rep.line(OK, "Pillow", _version("PIL"))


def check_torch(rep: Report) -> None:
    print("\n── Обучаемые ядра ──")
    if importlib.util.find_spec("torch") is None:
        rep.line(FAIL, "torch", "НЕТ → оба обучаемых этажа недоступны")
        return
    import torch

    rep.line(OK, "torch", torch.__version__)
    if torch.cuda.is_available():
        gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        rep.line(OK, "CUDA", f"{torch.cuda.get_device_name(0)}, {gb:.1f} ГБ")
    else:
        rep.line(WARN, "CUDA", "нет — всё считается на CPU, индекс будет очень долгим")

    for mod, why in (
        ("lightglue", "Этаж 2: основной матчер на кросс-домене"),
        ("huggingface_hub", "Этаж 1: веса MegaLoc через torch.hub"),
        ("safetensors", "Этаж 1: веса MegaLoc"),
    ):
        mark = OK if importlib.util.find_spec(mod) is not None else FAIL
        rep.line(mark, mod, why if mark == OK else f"НЕТ → {why}")

    for mod, why in (("faiss", "ANN-поиск; без него точный numpy-kNN"),
                     ("kornia", "LoFTR; не обязателен")):
        mark = OK if importlib.util.find_spec(mod) is not None else WARN
        rep.line(mark, mod, why)


def check_weights(rep: Report) -> None:
    """Поднять ядра по-настоящему: веса тянутся из сети, первый раз это долго."""
    print("\n── Веса (сеть) ──")
    import numpy as np

    from aero_geoloc.testbench import make_synthetic_scene

    scene = make_synthetic_scene(512, seed=7).image
    query = scene[100:420, 100:420].copy()

    try:
        from aero_geoloc.matcher import create_matcher

        t = time.perf_counter()
        corr = create_matcher("lightglue").match(query, scene)
        shift = np.median(corr.pts_r - corr.pts_q, axis=0)
        good = abs(shift[0] - 100) < 2 and abs(shift[1] - 100) < 2
        rep.line(
            OK if good else FAIL,
            "LightGlue",
            f"{len(corr)} соответствий, сдвиг {shift[0]:.0f},{shift[1]:.0f} "
            f"(ожидается 100,100) за {time.perf_counter() - t:.1f} с",
        )
    except Exception as exc:
        rep.line(FAIL, "LightGlue", f"{type(exc).__name__}: {str(exc).splitlines()[0]}")

    for name, cls_path, dim in (
        ("DINOv2", "DinoV2Encoder", 384),
        ("MegaLoc", "MegaLocEncoder", 8448),
    ):
        try:
            from aero_geoloc import retrieval

            enc = getattr(retrieval, cls_path)()
            t = time.perf_counter()
            vec = enc.encode(np.ascontiguousarray(scene[:322, :322]))
            ok = vec.shape[0] == dim and abs(float(np.linalg.norm(vec)) - 1.0) < 1e-3
            rep.line(
                OK if ok else FAIL,
                name,
                f"dim={vec.shape[0]} (ожидается {dim}), норма 1.0, "
                f"{time.perf_counter() - t:.1f} с",
            )
        except Exception as exc:
            rep.line(FAIL, name, f"{type(exc).__name__}: {str(exc).splitlines()[0]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--weights", action="store_true", help="поднять веса моделей (нужна сеть; первый раз долго)"
    )
    args = parser.parse_args()

    rep = Report()
    check_core(rep)
    check_real(rep)
    check_torch(rep)
    if args.weights:
        check_weights(rep)
    else:
        print("\n(веса не проверялись — запустите с --weights)")

    print()
    if rep.blocked:
        print("ЕСТЬ БЛОКИРУЮЩИЕ ПРОБЛЕМЫ — см. строки [-] выше.")
        print("Обычно чинится: pip install -r requirements-real.txt")
        return 1
    print("Окружение готово.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
