"""Сменная мера согласия в связке качества (E3 из `docs/ROADMAP.md`).

Смысл изменения: связка перестала знать, что её фотометрический член — именно
NCC. Теперь оркестрация выбирает МЕРУ, а :func:`quality.assess` знает только её
значение, имя и порог. Тесты закрепляют оба свойства, на которых это держится:
имя меры едет вместе со значением (иначе таблицу чисел нельзя прочитать), и
порог берётся по имени, а не литералом.
"""

from __future__ import annotations

import numpy as np
import pytest

from aero_geoloc import similarity
from aero_geoloc.localize import photometric_measure
from aero_geoloc.matcher import Correspondences
from aero_geoloc.pose import PoseEstimate, SimilarityTransform
from aero_geoloc.quality import (
    MIN_DINO,
    MIN_NCC,
    PHOTOMETRIC_THRESHOLDS,
    align_reference,
    aligned_ncc,
    aligned_structural,
    assess,
)
from aero_geoloc.types import Status

IDENTITY = SimilarityTransform.from_params(1.0, 0.0, 0.0, 0.0)


def scene(seed: int = 0, size: int = 128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (size // 4, size // 4), dtype=np.uint8)
    return np.repeat(np.repeat(base, 4, 0), 4, 1)


def _pose(n: int = 40, seed: int = 0) -> tuple[PoseEstimate, Correspondences]:
    rng = np.random.default_rng(seed)
    pts_q = rng.uniform(0, 120, (n, 2)).astype(np.float32)
    pts_r = (pts_q + rng.normal(0.0, 0.2, pts_q.shape)).astype(np.float32)
    corr = Correspondences(pts_q, pts_r, np.ones(n, np.float32))
    return PoseEstimate(IDENTITY, np.ones(n, bool), 0.2), corr


# --- выравнивание подложки --------------------------------------------------

def test_structural_matches_ncc_on_full_overlap():
    """При полном перекрытии обрезка ничего не меняет — сверка двух путей."""
    img = scene()
    assert aligned_structural(img, img, IDENTITY, similarity.ncc) == pytest.approx(
        aligned_ncc(img, img, IDENTITY), abs=1e-5)


def test_structural_crops_away_the_invalid_border():
    """Ключевое отличие от NCC: структурной мере нужна СВЯЗНАЯ картинка.

    NCC можно посчитать по разрозненным валидным пикселям, а градиентам и
    дескрипторам — нет. Если бы невалидную зону не обрезали, а занулили, на её
    границе возник бы искусственный край, которого нет ни на одной из картинок.
    """
    img = scene()
    shifted = SimilarityTransform.from_params(1.0, 0.0, 40.0, 25.0)
    _, valid = align_reference(img, img, shifted)
    assert not valid.all()                       # часть кадра осталась без данных
    value = aligned_structural(img, img, shifted, similarity.ncc)
    assert -1.0 <= value <= 1.0


def test_no_overlap_reports_nothing_to_compare():
    """«Сравнивать нечего» — это −1, тот же признак, что и у aligned_ncc."""
    img = scene()
    far = SimilarityTransform.from_params(1.0, 0.0, 5000.0, 5000.0)
    assert aligned_structural(img, img, far, similarity.ncc) == -1.0
    assert aligned_ncc(img, img, far) == -1.0


# --- связка качества --------------------------------------------------------

def test_threshold_comes_from_the_measure_name():
    """0.31 — «отлично» для NCC и «на грани» для dino. Порог обязан идти за именем."""
    pose, corr = _pose()
    for kind, threshold in PHOTOMETRIC_THRESHOLDS.items():
        result = assess(pose, corr, (60.0, 60.0), mpp=0.3,
                        photometric=threshold, photometric_kind=kind)
        assert result.signals["photometric_threshold"] == threshold
        assert result.signals["photometric_kind"] == kind
        assert result.status is Status.LOCALIZED

        below = assess(pose, corr, (60.0, 60.0), mpp=0.3,
                       photometric=threshold - 0.01, photometric_kind=kind)
        assert below.status is Status.LOW_CONFIDENCE


def test_dino_threshold_is_stricter_than_ncc():
    """Шкалы разные: у dino ноль означает «связи нет», у NCC — тоже, но пол выше."""
    assert MIN_DINO > MIN_NCC


def test_the_00049_defect_is_what_the_swap_fixes():
    """Живой дефект в одном тесте: место верное, NCC −0.018, dino 0.49.

    Числа взяты из прогона (веха E3 в `docs/JOURNAL.md`). Смысл проверки — не
    «эти константы такие», а что связка на них ведёт себя по-разному: та же поза
    режется по NCC и проходит по dino.
    """
    pose, corr = _pose()
    by_ncc = assess(pose, corr, (60.0, 60.0), mpp=0.3,
                    photometric=-0.018, photometric_kind="ncc")
    by_dino = assess(pose, corr, (60.0, 60.0), mpp=0.3,
                     photometric=0.491, photometric_kind="dino")
    assert by_ncc.status is Status.LOW_CONFIDENCE
    assert by_dino.status is Status.LOCALIZED


def test_explicit_threshold_wins_over_the_calibrated_default():
    pose, corr = _pose()
    result = assess(pose, corr, (60.0, 60.0), mpp=0.3, photometric=0.2,
                    photometric_kind="dino", min_photometric=0.1)
    assert result.status is Status.LOCALIZED
    assert result.signals["photometric_threshold"] == 0.1


def test_measure_stays_a_conjunction_member_not_a_veto():
    """Хорошая мера не спасает позу с недостаточным числом инлайеров."""
    pose, corr = _pose(n=5)
    result = assess(pose, corr, (60.0, 60.0), mpp=0.3,
                    photometric=0.9, photometric_kind="dino")
    assert result.status is Status.LOW_CONFIDENCE


# --- выбор меры оркестрацией ------------------------------------------------

def test_ncc_measure_is_the_plain_function():
    assert photometric_measure("ncc") is aligned_ncc


def test_unknown_measure_names_itself_and_the_alternatives():
    with pytest.raises(ValueError, match="неизвестная мера"):
        photometric_measure("вкусовщина")
