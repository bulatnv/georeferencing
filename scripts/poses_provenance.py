"""Позы для оракульных проб — с провенансом и отказом вместо тишины.

Модуль существует из-за конкретной утечки (``docs/FIX_EVAL_ARTIFACT_LEAK.md``):
частичный прогон стенда молча перезаписывал канонический ``eval_out/eval.csv``,
а оба потребителя (`probe_matcher.py`, `e_sigma_ransac.py`) читали его как
``--poses`` и на отсутствующих кейсах **молча пропускали** строки — либо, хуже,
брали позы, порождённые другим ядром, и это не было видно нигде.

Правило регламента, которое здесь реализовано: **вход эксперимента обязан нести
провенанс наравне с весами** — «кем, когда и каким ядром сделан», и эти ответы
обязаны доезжать до выходной таблицы.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

#: Боевое ядро — с ним сравнивается матчер из сайдкара поз. Позы от другого
#: ядра легитимны (мы ими пользуемся), но обязаны быть видны в логе и в CSV.
BATTLE_MATCHER = "minima_roma"

#: Колонки провенанса — добавляются в FIELDS обоих потребителей.
PROVENANCE_FIELDS = ["poses_src", "poses_matcher", "poses_n_cases", "poses_mtime"]


class PosesError(RuntimeError):
    """Вход эксперимента непригоден — отказ с подсказкой, а не тихий пропуск."""


def load_poses_with_provenance(
    path: str | Path,
    *,
    required: set[str] | None = None,
    allow_partial: bool = False,
    battle_matcher: str = BATTLE_MATCHER,
) -> tuple[dict[str, tuple[float, float, float]], dict[str, object]]:
    """Позы пайплайна + провенанс файла, из которого они взяты.

    Args:
        path: CSV прогона стенда (``found_lat``/``found_lon``/``heading_deg``).
        required: имена кейсов, которым поза обязательна (manual-истина без
            EXIF-курса). Пустое множество — файл не обязателен вовсе.
        allow_partial: разрешить работу при неполном покрытии ``required``
            (эскейп-хатч Ф4; пропущенные кейсы всё равно видны в CSV строками
            ``skipped_no_pose``).

    Returns:
        ``(poses, provenance)``; провенанс — значения для
        :data:`PROVENANCE_FIELDS`.

    Raises:
        PosesError: файла нет (при непустом ``required``) либо покрытие неполно
            без ``allow_partial``.
    """
    path = Path(path)
    required = set(required or ())
    provenance: dict[str, object] = {
        "poses_src": str(path), "poses_matcher": "unknown",
        "poses_n_cases": 0, "poses_mtime": "unknown",
    }

    if not path.exists():
        if not required:
            return {}, provenance
        raise PosesError(
            f"файла поз нет: {path}. Оракул manual-кейсов ({', '.join(sorted(required))}) "
            f"без него слеп. Создать полным прогоном боевого ядра: "
            f".venv/Scripts/python.exe scripts/eval_dataset.py")

    poses: dict[str, tuple[float, float, float]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                poses[row["case"]] = (float(row["found_lat"]), float(row["found_lon"]),
                                      float(row["heading_deg"]))
            except (KeyError, ValueError):
                continue
    provenance["poses_n_cases"] = len(poses)
    provenance["poses_mtime"] = datetime.fromtimestamp(
        path.stat().st_mtime).isoformat(timespec="seconds")

    sidecar = path.with_suffix(".config.json")
    if sidecar.exists():
        try:
            provenance["poses_matcher"] = str(
                json.load(open(sidecar, encoding="utf-8")).get("matcher", "unknown"))
        except (json.JSONDecodeError, OSError):
            provenance["poses_matcher"] = "unknown"
    if provenance["poses_matcher"] == "unknown":
        print(f"ПРЕДУПРЕЖДЕНИЕ: рядом с {path.name} нет читаемого сайдкара "
              f"*.config.json — провенанс поз неизвестен")
    elif provenance["poses_matcher"] != battle_matcher:
        print(f"ПРЕДУПРЕЖДЕНИЕ: позы в {path.name} порождены ядром "
              f"{provenance['poses_matcher']!r}, боевое — {battle_matcher!r}. "
              f"Это легитимно, но должно быть осознанно; матчер попадёт в CSV")

    missing = sorted(required - set(poses))
    if missing:
        message = (f"в {path.name} нет поз для {len(missing)} manual-кейсов из "
                   f"запрошенных: {', '.join(missing)} (в файле {len(poses)} кейсов). "
                   f"Частичный прогон стенда мог затереть канонический eval.csv — "
                   f"пересоздайте его полным прогоном боевого ядра")
        if not allow_partial:
            raise PosesError(message + ". Осознанно продолжить: --allow-partial-poses")
        print(f"ПРЕДУПРЕЖДЕНИЕ: {message}; пропуски попадут в CSV строками "
              f"skipped_no_pose")
    return poses, provenance
