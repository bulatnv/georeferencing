"""Отчёт: он обязан быть понятным человеку и честным про отказ.

Отдельно проверяется то, что легко потерять при рефакторинге: **отказ формирует
такой же полноценный отчёт, как успех**, и совет «что поменять» выводится из
причины, а не выдаётся списком на все случаи (`docs/TOOL_PLAN.md`, T3).
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from aero_geoloc.report import (
    advice_for,
    save_summary,
    footprint_geojson,
    footprint_kml,
    result_payload,
    save_report,
)
from aero_geoloc.request import build_request
from aero_geoloc.types import LocalizationResult, Status


@pytest.fixture
def request_obj(tmp_path):
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), np.full((600, 800, 3), 128, np.uint8))
    return build_request(path, lat=54.81, lon=56.09, sigma_m=1500.0,
                         gsd_m=0.065, yaw_deg=30.0)


FOOTPRINT = [(56.088, 54.809), (56.092, 54.809), (56.092, 54.811), (56.088, 54.811)]


def ok_result() -> LocalizationResult:
    return LocalizationResult(
        status=Status.LOCALIZED, center_lat=54.8100, center_lon=56.0900,
        heading_deg=30.0, altitude_est_m=500.0, error_ellipse_m=(0.9, 0.6, 12.0),
        footprint_lonlat=FOOTPRINT,
        diagnostics={"n_inliers": 412, "photometric": 0.63, "photometric_kind": "dino"},
    )


def refusal(reason: str) -> LocalizationResult:
    return LocalizationResult.failed(reason)


# --- отказ — полноценный результат ------------------------------------------

def test_refusal_produces_a_full_report(tmp_path, request_obj):
    """NOT_LOCALIZED — легитимный исход, а не сбой запуска."""
    report = save_report(tmp_path, request_obj,
                         refusal("точный уровень не сошёлся ни на одном кандидате"),
                         matcher="minima_roma")
    assert report.exists()
    for name in ("result.json", "footprint.kml", "footprint.geojson"):
        assert (tmp_path / name).exists()
    html = report.read_text(encoding="utf-8")
    assert "НЕ ЛОКАЛИЗОВАНО" in html
    assert "Что можно поменять" in html


@pytest.mark.parametrize("reason,expected", [
    ("низкая уникальность (самоподобие)", "однородна"),
    ("решение вне диска приора", "--sigma-km"),
    ("точный уровень не сошёлся ни на одном кандидате", "GSD"),
    ("retrieval не дал кандидатов", "--radius-km"),
    ("у подложки нет съёмки в этом районе (проверено с zoom 19 до 14)", "заглушку"),
])
def test_advice_follows_the_reason(request_obj, reason, expected):
    """Совет выводится из причины. Список «на все случаи» бесполезен."""
    tips = " ".join(advice_for(refusal(reason), request_obj))
    assert expected in tips


def test_success_needs_no_advice(request_obj):
    assert advice_for(ok_result(), request_obj) == []


def test_unknown_reason_still_says_something_useful(request_obj):
    tips = advice_for(refusal("нечто новое"), request_obj)
    assert tips and "оверлей" in " ".join(tips)


# --- содержание отчёта ------------------------------------------------------

def test_report_shows_the_footprint_for_eyeballing(tmp_path, request_obj):
    """Неверный GSD — самая частая ошибка ввода, и она видна только по отпечатку."""
    html = save_report(tmp_path, request_obj, ok_result(),
                       matcher="minima_roma").read_text(encoding="utf-8")
    assert "Кадр покрывает" in html and "52 × 39 м" in html


def test_report_is_self_contained(tmp_path, request_obj):
    """Открывается двойным кликом: картинка внутри, внешних запросов нет."""
    overlay = np.full((80, 160, 3), 200, np.uint8)
    html = save_report(tmp_path, request_obj, ok_result(), overlay=overlay,
                       matcher="minima_roma").read_text(encoding="utf-8")
    assert "data:image/png;base64," in html
    assert "<script" not in html and "http://" not in html
    assert (tmp_path / "overlay.png").exists()


def test_ellipse_is_explained_not_just_printed(tmp_path, request_obj):
    """Голое «0.9 м» вводит в заблуждение: это не абсолютная точность."""
    html = save_report(tmp_path, request_obj, ok_result(),
                       matcher="minima_roma").read_text(encoding="utf-8")
    assert "случайная" in html and "подложки" in html


def test_rejected_pose_shows_no_ellipse(tmp_path, request_obj):
    """Субметровый эллипс рядом с отказом читается как «зато очень точно».

    Случай DSC00045: связка отвергла позу, построенную по пустой подложке, а в
    шапке стояло «эллипс 0.45 м» — то есть отчёт приглашал поверить отказу.
    """
    rejected = LocalizationResult(
        status=Status.LOW_CONFIDENCE, center_lat=56.7747, center_lon=52.7791,
        heading_deg=229.6, altitude_est_m=382.0, error_ellipse_m=(0.454, 0.454, -45.0),
        footprint_lonlat=FOOTPRINT, diagnostics={"n_inliers": 22},
    )
    html = save_report(tmp_path, request_obj, rejected,
                       matcher="minima_roma").read_text(encoding="utf-8")
    assert "0.45" not in html and "поза не принята" in html


def test_prior_provenance_reaches_the_report(tmp_path, request_obj):
    html = save_report(tmp_path, request_obj, ok_result(),
                       matcher="minima_roma").read_text(encoding="utf-8")
    assert "задан аргументами" in html


# --- машиночитаемое ---------------------------------------------------------

def test_payload_is_json_serialisable_with_numpy_inside(request_obj):
    """Диагностика приходит из numpy — сериализация не должна падать на этом."""
    result = LocalizationResult(
        status=Status.LOCALIZED, center_lat=54.81, center_lon=56.09,
        heading_deg=30.0, altitude_est_m=500.0, error_ellipse_m=(0.9, 0.6, 12.0),
        footprint_lonlat=FOOTPRINT,
        diagnostics={"n_inliers": np.int64(412), "scale": np.float32(1.01),
                     "scale_bounds": (0.7, 1.4), "cov": np.zeros((2, 2))},
    )
    payload = result_payload(request_obj, result, matcher="minima_roma")
    json.dumps(payload, ensure_ascii=False)      # не должно бросить
    assert payload["статус"] == "localized"


def test_geojson_and_kml_carry_both_shapes():
    gj = footprint_geojson(ok_result(), name="frame")
    kinds = {f["geometry"]["type"] for f in gj["features"]}
    assert kinds == {"Polygon", "Point"}
    kml = footprint_kml(ok_result(), name="frame")
    assert "<Polygon>" in kml and "<Point>" in kml and kml.startswith("<?xml")


def test_geojson_of_a_refusal_is_empty_but_valid():
    gj = footprint_geojson(refusal("нет позы"), name="frame")
    assert gj["type"] == "FeatureCollection" and gj["features"] == []


# --- сводка по пачке --------------------------------------------------------

def test_summary_counts_and_links(tmp_path):
    """Главный вопрос при прогоне серии — «сколько взято и что с остальными»."""
    rows = [
        {"name": "a", "status": "localized", "lat": 54.8, "lon": 56.1,
         "ellipse": "<0.1 м", "inliers": 400, "seconds": 30.0},
        {"name": "b", "status": "not_localized", "lat": None, "lon": None,
         "reason": "низкая уникальность (самоподобие)", "seconds": 12.0},
    ]
    html = save_summary(tmp_path, rows).read_text(encoding="utf-8")
    assert "2 снимков" in html and "локализовано 1" in html and "не принято 1" in html
    assert "a/report.html" in html and "b/report.html" in html


def test_summary_shows_the_reason_of_a_refusal(tmp_path):
    """Отказ в сводке — полноправная строка: по ней видно, повторяется ли причина."""
    rows = [{"name": "b", "status": "not_localized", "lat": None, "lon": None,
             "reason": "решение вне диска приора", "seconds": 5.0}]
    html = save_summary(tmp_path, rows).read_text(encoding="utf-8")
    assert "вне диска приора" in html
    assert "отказ легитимен" in html.lower() or "лучше уверенно-неверной" in html
