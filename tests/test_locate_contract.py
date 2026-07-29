"""Контракт между локализацией одного снимка и сводкой по пачке.

Тест существует из-за конкретной поломки: ``locate_one`` возвращал кортеж вместо
словаря, и сводка по пачке падала **на каждом** снимке. Отдельные отчёты при этом
писались, консоль печатала успех — а `summary.html` не собирался.

Тесты этого не видели, потому что проверяли ``save_summary`` на рукописных
словарях, в отрыве от её единственного настоящего источника. Здесь склейка
проверяется целиком и без torch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from locate import summary_row  # noqa: E402

from aero_geoloc.report import save_summary  # noqa: E402
from aero_geoloc.types import LocalizationResult, Status  # noqa: E402

TIMINGS = {"подготовка кадра": 0.2, "карта района": 19.0, "локализация": 25.3}


def localized(ellipse=(0.9, 0.6, 12.0)) -> LocalizationResult:
    return LocalizationResult(
        status=Status.LOCALIZED, center_lat=54.810675, center_lon=56.087295,
        heading_deg=67.1, altitude_est_m=300.0, error_ellipse_m=ellipse,
        footprint_lonlat=[(56.086, 54.809)] * 4,
        diagnostics={"n_inliers": 412, "photometric": 0.63},
    )


def test_row_feeds_the_summary_end_to_end(tmp_path):
    """ГЛАВНЫЙ тест: то, что отдаёт локализация, сводка обязана принять."""
    rows = [summary_row("a", localized(), TIMINGS),
            summary_row("b", LocalizationResult.failed("низкая уникальность"), TIMINGS)]
    html = save_summary(tmp_path, rows).read_text(encoding="utf-8")
    assert "локализовано 1" in html and "не принято 1" in html
    assert "a/report.html" in html and "b/report.html" in html
    assert "низкая уникальность" in html


def test_row_has_every_key_the_summary_reads():
    """Ключи перечислены явно: молчаливое исчезновение любого ломает сводку."""
    row = summary_row("a", localized(), TIMINGS)
    for key in ("name", "status", "lat", "lon", "ellipse", "inliers", "reason", "seconds"):
        assert key in row, key


def test_console_line_is_readable_for_a_success():
    line = summary_row("a", localized(), TIMINGS)["line"]
    assert "localized" in line and "54.810675" in line and "курс 67°" in line


def test_console_line_of_a_refusal_names_the_reason():
    line = summary_row("a", LocalizationResult.failed("решение вне диска приора"), TIMINGS)["line"]
    assert "not_localized" in line and "вне диска приора" in line


def test_subcentimetre_ellipse_is_not_shown_as_zero():
    """«0.0 м» читается как «ошибки нет» — это вводит в заблуждение."""
    assert summary_row("a", localized(ellipse=(0.03, 0.02, 5.0)), TIMINGS)["ellipse"] == "<0.1 м"
    assert summary_row("a", localized(ellipse=(0.9, 0.6, 5.0)), TIMINGS)["ellipse"] == "0.90 м"


def test_rejected_pose_shows_no_ellipse_in_the_summary():
    """У непринятой позы эллипс — разброс подгонки, а не точность места."""
    rejected = LocalizationResult(
        status=Status.LOW_CONFIDENCE, center_lat=56.7747, center_lon=52.7791,
        heading_deg=229.6, altitude_est_m=382.0, error_ellipse_m=(0.454, 0.454, -45.0),
        footprint_lonlat=[(52.77, 56.77)] * 4, diagnostics={"n_inliers": 22},
    )
    row = summary_row("a", rejected, TIMINGS)
    assert row["ellipse"] == "—" and "эллипс" not in row["line"]


def test_seconds_are_summed_across_stages():
    assert summary_row("a", localized(), TIMINGS)["seconds"] == pytest.approx(44.5)


def test_refusal_carries_no_coordinates():
    row = summary_row("a", LocalizationResult.failed("нет позы"), TIMINGS)
    assert row["lat"] is None and row["inliers"] == "—"
