"""Регрессионный сторож: проверяем сам сторож.

Тесты написаны против **сценариев**, а не против чисел текущего прогона: каждый
воспроизводит конкретную ситуацию, ради которой этап F вообще затевался (см.
`docs/ROADMAP.md`, фаза 0), — прежде всего два случая, когда тихая регрессия уже
почти прошла.
"""

from __future__ import annotations

import pytest

from aero_geoloc.regression import (
    OUTCOME_RANK,
    CaseExpectation,
    Golden,
    compare,
    freeze,
    load_golden,
    outcome_of,
    save_golden,
)

CONFIG = {"matcher": "lightglue", "radius_km": 1.5, "top_k": 15}


def row(case, status, *, accepted="", correct="", error_m="", blame="", rank="",
        inliers="", ncc=""):
    return {
        "case": case, "status": status, "accepted": accepted, "correct": correct,
        "error_m": error_m, "blame": blame, "true_cell_rank": rank,
        "n_inliers": inliers, "photometric": ncc,
    }


def ok_row(case, error_m=1.6, rank=2):
    return row(case, "localized", accepted=1, correct=1, error_m=error_m, blame="ok",
               rank=rank, inliers=18, ncc=0.58)


# --- классификация исхода ---------------------------------------------------

@pytest.mark.parametrize("r,expected", [
    (ok_row("a"), "accepted_correct"),
    (row("a", "localized", accepted=1, correct=0, error_m=2126), "accepted_wrong"),
    (row("a", "low_confidence", accepted=0, correct=1, error_m=8.5), "pose_correct_gated"),
    (row("a", "low_confidence", accepted=0, correct=0, error_m=900), "pose_wrong_gated"),
    (row("a", "localized", accepted=1), "accepted_unverified"),
    (row("a", "low_confidence", accepted=0), "pose_gated_unverified"),
    (row("a", "not_localized", blame="Этаж 1 (верная клетка на 661, top-K=15)"), "refused_floor1"),
    (row("a", "not_localized", blame="Этаж 2 (клетка доставлена, поза не сошлась)"), "refused_floor2"),
    (row("a", "not_localized", blame="поза не найдена"), "refused"),
    (row("a", "ошибка", blame="SSLError"), "error"),
])
def test_outcome_classification(r, expected):
    assert outcome_of(r) == expected


def test_false_positive_is_worst_outcome():
    """Ложное срабатывание — ниже отказа и ниже аварии.

    Инвариант «честный отказ дороже красивой точки»: уверенно-неверная точка
    вреднее молчания, потому что на неё полагаются.
    """
    assert OUTCOME_RANK["accepted_wrong"] < OUTCOME_RANK["error"]
    assert OUTCOME_RANK["accepted_wrong"] < OUTCOME_RANK["refused_floor2"]
    assert OUTCOME_RANK["accepted_correct"] > OUTCOME_RANK["pose_correct_gated"]
    assert OUTCOME_RANK["pose_correct_gated"] > OUTCOME_RANK["refused_floor2"]
    # Гейт отверг неверную позу — снаружи это тот же отказ, не хуже и не лучше.
    assert OUTCOME_RANK["pose_wrong_gated"] == OUTCOME_RANK["refused_floor2"]


# --- сценарии, ради которых всё затевалось ----------------------------------

def test_lost_localization_is_regression():
    """Сценарий Volgograd3: авто-перекрытие превратило 0.8 м в отказ.

    Это ровно тот случай, который прошёл бы незамеченным: одна строка из
    шестнадцати, а сводка «верных 7/7» выглядит здоровой.
    """
    golden = freeze([ok_row("Volgograd3", error_m=0.8, rank=9)], CONFIG)
    now = [row("Volgograd3", "not_localized", blame="Этаж 2 (клетка доставлена, поза не сошлась)")]
    report = compare(now, golden, CONFIG)
    assert not report.passed
    assert [v.name for v in report.failures] == ["Volgograd3"]


def test_false_positive_fails_even_with_improvements():
    """Размен «+1 верная ценой одной ложной» не является улучшением.

    Если бы вердикт считался по сводным числам, такой прогон выглядел бы
    нейтральным. Он неприемлем.
    """
    golden = freeze([
        ok_row("good"),
        row("bad", "not_localized", blame="Этаж 2 (клетка доставлена, поза не сошлась)"),
    ], CONFIG)
    now = [
        row("good", "localized", accepted=1, correct=0, error_m=2126, blame="ЛОЖНОЕ"),
        ok_row("bad", error_m=3.0),
    ]
    report = compare(now, golden, CONFIG)
    assert not report.passed
    assert [v.name for v in report.failures] == ["good"]
    assert [v.name for v in report.improvements] == ["bad"]


def test_config_mismatch_blocks_verdict():
    """Сценарий ключа кэша: сравнение при других параметрах — не доказательство.

    Прогон с другим радиусом физически другой эксперимент; выдать по нему
    «регрессий нет» значит соврать тем же способом, каким соврал кэш карт.
    """
    golden = freeze([ok_row("a")], CONFIG)
    report = compare([ok_row("a")], golden, {**CONFIG, "radius_km": 2.0})
    assert not report.passed
    assert not report.failures          # сами кейсы в порядке
    assert report.config_diff           # но вердикт не выдаётся
    assert "radius_km" in report.config_diff[0]


def test_gate_release_is_improvement_not_noise():
    """Цель фазы 1: 00049 перестаёт резаться гейтом. Это улучшение, а не «изменилось»."""
    golden = freeze([row("00049", "low_confidence", accepted=0, correct=1, error_m=8.5,
                         blame="ГЕЙТ отверг ВЕРНУЮ (NCC 0.113<0.12)", rank=2,
                         inliers=49, ncc=0.1129)], CONFIG)
    report = compare([ok_row("00049", error_m=8.5)], golden, CONFIG)
    assert report.passed
    assert [v.severity for v in report.verdicts] == ["улучшение"]


# --- полоса допуска по ошибке -----------------------------------------------

def test_error_noise_within_band_is_ok():
    """RANSAC недетерминирован: метры гуляют сами по себе, это не регрессия."""
    golden = freeze([ok_row("a", error_m=8.5)], CONFIG)
    report = compare([ok_row("a", error_m=11.0)], golden, CONFIG)
    assert report.passed
    assert report.verdicts[0].severity == "ok"


def test_error_blowup_within_tolerance_is_warning():
    """0.8 м → 45 м формально «верно», фактически поза поехала. Предупреждение."""
    golden = freeze([ok_row("a", error_m=0.8)], CONFIG)
    report = compare([ok_row("a", error_m=45.0)], golden, CONFIG)
    assert report.passed                       # класс исхода не изменился
    assert report.verdicts[0].severity == "ухудшение"
    assert [v.name for v in report.warnings] == ["a"]


def test_error_drop_is_improvement():
    golden = freeze([ok_row("a", error_m=60.0)], CONFIG)
    report = compare([ok_row("a", error_m=2.0)], golden, CONFIG)
    assert report.verdicts[0].severity == "улучшение"


# --- состав набора ----------------------------------------------------------

def test_missing_and_new_cases_are_flagged_but_not_failures():
    """Кейс мог не прогоняться (--cases) — это повод сказать, а не рушить сборку."""
    golden = freeze([ok_row("a"), ok_row("b")], CONFIG)
    report = compare([ok_row("a"), ok_row("c")], golden, CONFIG)
    assert report.passed
    severities = {v.name: v.severity for v in report.verdicts}
    assert severities == {"a": "ok", "b": "пропал", "c": "новый"}


def test_roundtrip_through_yaml(tmp_path):
    golden = freeze([ok_row("a"), row("b", "not_localized", blame="Этаж 1 (…)")],
                    CONFIG, note="проверка")
    path = tmp_path / "golden.yaml"
    save_golden(golden, path, header="# комментарий")
    restored = load_golden(path)
    assert restored.config == golden.config
    assert set(restored.cases) == set(golden.cases)
    assert restored.cases["a"].outcome == "accepted_correct"
    assert restored.cases["a"].error_m == pytest.approx(1.6)
    assert restored.cases["b"].outcome == "refused_floor1"
    assert path.read_text(encoding="utf-8").startswith("# комментарий")


def test_summary_counts_outcomes():
    golden = Golden(cases={
        "a": CaseExpectation("a", "accepted_correct"),
        "b": CaseExpectation("b", "accepted_correct"),
        "c": CaseExpectation("c", "refused_floor2"),
    })
    assert golden.summary == {"accepted_correct": 2, "refused_floor2": 1}
