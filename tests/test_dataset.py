"""Тесты датасета оценки: манифест — единственный источник правды о кейсах.

Основная часть идёт на **синтетическом** манифесте во временной папке: реальные
снимки — чужие данные вне репозитория, и тест не должен от них зависеть. Прогон по
настоящему `datasets/test_images.yaml` — отдельный, gated по наличию файлов.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from aero_geoloc.dataset import Dataset, EvalCase, load_dataset

_REAL_MANIFEST = Path(__file__).resolve().parents[1] / "datasets" / "test_images.yaml"
_REAL_IMAGES = Path(__file__).resolve().parents[1] / "test_images"


def _write_case_image(path: Path, width: int = 800, height: int = 600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    cv2.imwrite(str(path), rng.integers(0, 255, (height, width, 3), dtype=np.uint8))


@pytest.fixture()
def synthetic_manifest(tmp_path: Path) -> Path:
    """Манифест с кейсом без EXIF, исключённым кейсом и битой ссылкой."""
    _write_case_image(tmp_path / "images" / "plain.jpg")
    manifest = tmp_path / "datasets" / "sample.yaml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        """
dataset: sample
root: images
cases:
  - name: plain
    path: plain.jpg
    truth: none
    gsd_m: 0.065
    altitude_m: 500.0
    prior: {lat: 51.5, lon: 46.0, sigma_m: 1500.0}
    notes: кейс без метаданных
  - name: skipped
    path: whatever.jpg
    exclude: снят в горизонт
  - name: missing
    path: no_such_file.jpg
    truth: none
    gsd_m: 0.065
    prior: {lat: 51.5, lon: 46.0}
""".lstrip(),
        encoding="utf-8",
    )
    return manifest


def test_loads_case_without_exif_from_gsd(synthetic_manifest):
    """Снимок без метаданных собирается из GSD манифеста — класс входа поддержан."""
    ds = load_dataset(synthetic_manifest, repo_root=synthetic_manifest.parent.parent)
    case = ds.by_name("plain")
    assert isinstance(ds, Dataset) and isinstance(case, EvalCase)
    assert (case.camera.image_width, case.camera.image_height) == (800, 600)
    # Камера восстановлена так, что её GSD равен заявленному в манифесте.
    assert case.camera.gsd(case.prior.altitude_m) == pytest.approx(0.065, rel=1e-9)
    assert case.gsd_m == pytest.approx(0.065)
    assert case.prior.lat == 51.5 and case.prior.sigma_m == 1500.0


def test_case_without_metadata_does_not_claim_yaw_or_truth(synthetic_manifest):
    """Курса и истины у такого снимка нет — и кейс об этом честно сообщает."""
    case = load_dataset(
        synthetic_manifest, repo_root=synthetic_manifest.parent.parent
    ).by_name("plain")
    assert case.trust_yaw is False  # yaw=0 в приоре — заглушка, а не знание
    assert case.has_truth is False and case.truth_source == "none"


def test_excluded_case_carries_reason(synthetic_manifest):
    ds = load_dataset(synthetic_manifest, repo_root=synthetic_manifest.parent.parent)
    reasons = {e.name: e.reason for e in ds.excluded}
    assert "снят в горизонт" in reasons["skipped"]
    assert "plain" not in reasons


def test_broken_entry_is_excluded_not_fatal(synthetic_manifest):
    """Один плохой снимок не должен закрывать прогон по всему набору."""
    ds = load_dataset(synthetic_manifest, repo_root=synthetic_manifest.parent.parent)
    assert [c.name for c in ds.cases] == ["plain"]
    assert "файла нет" in {e.name: e.reason for e in ds.excluded}["missing"]


def test_frame_at_mpp_matches_drone_contract(synthetic_manifest):
    """Ресемпл до mpp подложки: масштаб ≈1, FOV сохранён (как во frame_at_mpp)."""
    case = load_dataset(
        synthetic_manifest, repo_root=synthetic_manifest.parent.parent
    ).by_name("plain")
    target_mpp = 0.13  # вдвое грубее GSD кадра → кадр ужимается вдвое
    frame, camera = case.frame_at_mpp(target_mpp)
    assert (camera.image_width, camera.image_height) == (400, 300)
    assert frame.shape[:2] == (300, 400)
    assert camera.fov_deg == pytest.approx(case.camera.fov_deg)
    # После ресемпла GSD кадра равен разрешению подложки — то, ради чего он и делается.
    assert camera.gsd(case.prior.altitude_m) == pytest.approx(target_mpp, rel=1e-6)


def test_manual_truth_is_read_from_manifest(tmp_path):
    """Ручная разметка (truth: {lat, lon}) читается как истина с источником manual."""
    _write_case_image(tmp_path / "images" / "annotated.jpg")
    manifest = tmp_path / "datasets" / "m.yaml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        """
dataset: m
root: images
cases:
  - name: annotated
    path: annotated.jpg
    truth: {lat: 51.54, lon: 46.01}
    gsd_m: 0.065
    prior: {lat: 51.54, lon: 46.01}
""".lstrip(),
        encoding="utf-8",
    )
    case = load_dataset(manifest, repo_root=tmp_path).by_name("annotated")
    assert case.truth_source == "manual"
    assert case.has_truth and case.truth_lat == pytest.approx(51.54)


def test_case_without_gsd_and_without_exif_is_rejected(tmp_path):
    """Без EXIF и без GSD камеру собрать не из чего — кейс уходит с внятной причиной."""
    _write_case_image(tmp_path / "images" / "bare.jpg")
    manifest = tmp_path / "datasets" / "m.yaml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "dataset: m\nroot: images\ncases:\n  - name: bare\n    path: bare.jpg\n"
        "    truth: none\n    prior: {lat: 51.5, lon: 46.0}\n",
        encoding="utf-8",
    )
    ds = load_dataset(manifest, repo_root=tmp_path)
    assert not ds.cases
    assert "gsd_m" in ds.excluded[0].reason


# --- реальный манифест проекта (gated: снимки лежат вне репозитория) ----------


@pytest.mark.skipif(not _REAL_IMAGES.exists(), reason="нужны снимки test_images/")
def test_real_manifest_splits_usable_and_excluded():
    """Реальный набор: годные кейсы собраны, исключённые названы с причиной."""
    ds = load_dataset(_REAL_MANIFEST)
    names = {c.name for c in ds.cases}
    # Кадры с EXIF дают истину сразу; снимки без метаданных — пока без неё.
    assert {"00049", "Ufa2", "Ufa3", "train_SB_0023"} <= names
    assert {"Saratov", "Volgograd"} <= names
    assert {c.name for c in ds.with_truth} == {"00049", "Ufa2", "Ufa3", "train_SB_0023"}
    # Непригодные исключены именно как данные, а не молчаливым пропуском.
    excluded = {e.name: e.reason for e in ds.excluded}
    assert "Ufa" in excluded and "горизонт" in excluded["Ufa"]
    assert "DSC00045" in excluded
    # У снимков без метаданных курс неизвестен — это должно быть видно из кейса.
    assert ds.by_name("Saratov").trust_yaw is False
    assert ds.by_name("00049").trust_yaw is True
