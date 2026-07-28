"""Датасет оценки как объект первого класса: манифест → готовые кейсы.

Зачем модуль ([EVAL_PLAN.md](../docs/EVAL_PLAN.md), этап A): раньше каждый скрипт
сам сканировал папку со снимками и сам собирал приор, а знание «этот кадр снят в
горизонт, его нельзя» жило в голове автора, а не в данных. Из-за этого метрику
можно было посчитать по кадрам, которые не локализуемы в принципе.

Здесь единственный источник правды — YAML-манифест (``datasets/*.yaml``), а
модуль отдаёт **однородный** :class:`EvalCase` независимо от того, откуда взялись
параметры: из EXIF снимка или из манифеста.

Два класса входа
----------------
*Снимок с метаданными* (``truth: exif``) — камера, высота, курс и истина берутся
из EXIF/XMP через :mod:`aero_geoloc.drone`.

*Снимок без метаданных* (EXIF срезан) — камера восстанавливается из **известного
GSD** (:meth:`Camera.from_gsd`), приор задаётся в манифесте, а истина
размечается по подложке и подтверждается владельцем (``truth: manual``). Это
самый реалистичный операционный случай: снимок из неизвестного источника, известен
лишь примерный район.

Кадр отдаётся тем же контрактом, что у :func:`aero_geoloc.drone.frame_at_mpp` —
ресемпл до разрешения подложки с сохранением FOV, — поэтому оркестрация
(``localize``) не видит разницы между классами входа.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .camera import Camera
from .geo import zoom_for_mpp
from .types import Prior

__all__ = ["EvalCase", "ExcludedCase", "Dataset", "load_dataset"]

#: Дефолты приора, когда манифест их не задаёт.
_DEFAULT_SIGMA_M = 2000.0
_DEFAULT_ALTITUDE_SIGMA_M = 50.0


@dataclass(frozen=True)
class EvalCase:
    """Один кейс оценки: чем локализовать и с чем сравнивать.

    Attributes:
        name: короткое имя кейса (ключ в отчётах и именах оверлеев).
        path: путь к снимку.
        camera: камера в **нативном** разрешении снимка.
        prior: приор позиции/высоты/курса — ограничение, а не подсказка.
        gsd_m: разрешение кадра на земле, м/пиксель.
        truth_lat, truth_lon: истина или ``None``, если её ещё нет.
        truth_source: ``exif`` | ``manual`` | ``none`` — откуда истина.
        trust_yaw: известен ли курс. У снимков без метаданных — ``False``,
            и тогда предповорот кадра невозможен (см. EVAL_PLAN, Б1/Б3).
        regime: ``in_season`` | ``cross_season`` — совпадает ли сезон съёмки с
            сезоном подложки. Свойство ДАННЫХ, а не алгоритма, поэтому живёт в
            манифесте: без него анализ сигналов усредняет два разных режима в
            одну кучу и получает бессмысленную середину.
        notes: заметка из манифеста (чем кейс интересен или труден).
    """

    name: str
    path: Path
    camera: Camera
    prior: Prior
    gsd_m: float
    truth_lat: float | None = None
    truth_lon: float | None = None
    truth_source: str = "none"
    trust_yaw: bool = True
    regime: str = "in_season"
    notes: str = ""

    @property
    def has_truth(self) -> bool:
        """Есть ли с чем сравнивать результат (иначе ошибку в метрах не измерить)."""
        return self.truth_lat is not None and self.truth_lon is not None

    def basemap_zoom(self, *, max_zoom: int) -> int:
        """Зум подложки под GSD кадра, клампованный к максимуму провайдера."""
        return zoom_for_mpp(self.gsd_m, self.prior.lat, max_zoom=max_zoom)

    def frame_at_mpp(self, target_mpp: float) -> tuple[np.ndarray, Camera]:
        """Кадр, ресемплированный до разрешения подложки, + согласованная камера.

        Тот же контракт, что у :func:`aero_geoloc.drone.frame_at_mpp`: масштаб
        приводится к ``≈1``, FOV сохраняется, камера пересобирается под новый
        размер. Снимок читается с диска здесь, а не в конструкторе, — кадры
        бывают по 40 Мп, и держать их все в памяти незачем.
        """
        image = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"{self.path}: не читается как изображение")
        scale = self.gsd_m / target_mpp
        new_w = max(16, round(self.camera.image_width * scale))
        new_h = max(16, round(self.camera.image_height * scale))
        frame = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return frame, Camera(new_w, new_h, fov_deg=self.camera.fov_deg)


@dataclass(frozen=True)
class ExcludedCase:
    """Снимок, намеренно исключённый из оценки, с причиной.

    Причина хранится в манифесте, а не в коде: исключения — свойство данных.
    """

    name: str
    path: Path
    reason: str


@dataclass(frozen=True)
class Dataset:
    """Загруженный манифест: годные кейсы и исключённые с причинами."""

    name: str
    cases: list[EvalCase] = field(default_factory=list)
    excluded: list[ExcludedCase] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.cases)

    def by_name(self, name: str) -> EvalCase:
        for case in self.cases:
            if case.name == name:
                return case
        raise KeyError(f"кейс {name!r} не найден (есть: {[c.name for c in self.cases]})")

    @property
    def with_truth(self) -> list[EvalCase]:
        """Кейсы, по которым можно измерить ошибку в метрах."""
        return [c for c in self.cases if c.has_truth]


def _case_from_exif(entry: dict, path: Path, root: Path) -> EvalCase:
    """Кейс из снимка с метаданными: камера/высота/курс/истина из EXIF+XMP."""
    from .drone import load_drone_shot  # локально: тянет Pillow

    shot = load_drone_shot(path)
    prior_cfg = entry.get("prior") or {}
    # Приор по умолчанию стоит на истине (её загрубляет уже харнесс свипа);
    # манифест может сдвинуть его явно.
    prior = Prior(
        lat=float(prior_cfg.get("lat", shot.true_lat)),
        lon=float(prior_cfg.get("lon", shot.true_lon)),
        sigma_m=float(prior_cfg.get("sigma_m", _DEFAULT_SIGMA_M)),
        altitude_m=shot.altitude_m,
        altitude_sigma_m=float(prior_cfg.get("altitude_sigma_m", _DEFAULT_ALTITUDE_SIGMA_M)),
        yaw_deg=shot.yaw_deg,
        pitch_deg=shot.pitch_from_nadir_deg,
        roll_deg=shot.roll_deg,
    )
    return EvalCase(
        name=entry["name"],
        path=path,
        camera=shot.camera,
        prior=prior,
        gsd_m=shot.camera.gsd(shot.altitude_m),
        truth_lat=shot.true_lat,
        truth_lon=shot.true_lon,
        truth_source="exif",
        trust_yaw=True,
        regime=str(entry.get("regime", "in_season")).strip(),
        notes=str(entry.get("notes", "")).strip(),
    )


def _case_from_manifest(entry: dict, path: Path) -> EvalCase:
    """Кейс без метаданных: камера из GSD, приор и истина — из манифеста."""
    if "gsd_m" not in entry:
        raise ValueError(
            f"кейс {entry['name']!r}: без EXIF нужен gsd_m в манифесте "
            f"(камеру не из чего собрать)"
        )
    prior_cfg = entry.get("prior") or {}
    if "lat" not in prior_cfg or "lon" not in prior_cfg:
        raise ValueError(f"кейс {entry['name']!r}: без EXIF нужен prior.lat/lon в манифесте")

    image = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_8)  # только ради размера
    if image is None:
        raise ValueError(f"{path}: не читается как изображение")
    height, width = (s * 8 for s in image.shape[:2])

    gsd_m = float(entry["gsd_m"])
    altitude_m = float(entry.get("altitude_m", 500.0))
    camera = Camera.from_gsd(width, height, gsd_m=gsd_m, altitude_m=altitude_m)

    truth = entry.get("truth")
    truth_lat = truth_lon = None
    truth_source = "none"
    if isinstance(truth, dict):  # truth: {lat: ..., lon: ...} — ручная разметка
        truth_lat, truth_lon = float(truth["lat"]), float(truth["lon"])
        truth_source = "manual"

    # Курс неизвестен: yaw=0 — это заглушка для Prior, а не знание. Отсюда
    # trust_yaw=False, и предповорот кадра делать нельзя (EVAL_PLAN, Б1/Б3).
    prior = Prior(
        lat=float(prior_cfg["lat"]),
        lon=float(prior_cfg["lon"]),
        sigma_m=float(prior_cfg.get("sigma_m", _DEFAULT_SIGMA_M)),
        altitude_m=altitude_m,
        altitude_sigma_m=float(prior_cfg.get("altitude_sigma_m", _DEFAULT_ALTITUDE_SIGMA_M)),
        yaw_deg=0.0,
    )
    return EvalCase(
        name=entry["name"],
        path=path,
        camera=camera,
        prior=prior,
        gsd_m=gsd_m,
        truth_lat=truth_lat,
        truth_lon=truth_lon,
        truth_source=truth_source,
        trust_yaw=False,
        regime=str(entry.get("regime", "in_season")).strip(),
        notes=str(entry.get("notes", "")).strip(),
    )


def load_dataset(manifest_path, *, repo_root=None) -> Dataset:
    """Прочитать манифест и собрать кейсы (EXIF сливается с записями манифеста).

    Args:
        manifest_path: путь к YAML-манифесту (``datasets/test_images.yaml``).
        repo_root: корень, от которого отсчитываются пути снимков. По умолчанию —
            родитель каталога манифеста (то есть корень репозитория).

    Returns:
        :class:`Dataset` с годными кейсами и исключёнными (с причиной).

    Кейс, который не удалось собрать (нет файла, нет GSD, битый EXIF), не роняет
    загрузку: он уходит в ``excluded`` с текстом ошибки — иначе один плохой
    снимок закрывал бы прогон по всему набору.
    """
    import yaml

    manifest_path = Path(manifest_path)
    root_dir = Path(repo_root) if repo_root is not None else manifest_path.parent.parent
    with open(manifest_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    images_root = root_dir / str(data.get("root", "."))
    cases: list[EvalCase] = []
    excluded: list[ExcludedCase] = []

    for entry in data.get("cases", []):
        path = images_root / entry["path"]
        if entry.get("exclude"):
            excluded.append(
                ExcludedCase(entry["name"], path, " ".join(str(entry["exclude"]).split()))
            )
            continue
        if not path.exists():
            excluded.append(ExcludedCase(entry["name"], path, f"файла нет: {path}"))
            continue
        try:
            if entry.get("truth") == "exif":
                cases.append(_case_from_exif(entry, path, images_root))
            else:
                cases.append(_case_from_manifest(entry, path))
        except Exception as exc:  # noqa: BLE001 — причина уходит в отчёт, а не в стек
            excluded.append(ExcludedCase(entry["name"], path, f"{type(exc).__name__}: {exc}"))

    return Dataset(name=str(data.get("dataset", manifest_path.stem)), cases=cases, excluded=excluded)
