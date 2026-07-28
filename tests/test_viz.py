"""Тесты визуализации: оверлей должен собираться и на успехе, и на отказе.

Проверяется структура (панели, шапка, подпись) и устойчивость к краевым входам —
содержимое пикселей не сравнивается: оверлей нужен человеку, а не автомату.
"""

from __future__ import annotations

import numpy as np
import pytest

from aero_geoloc.geo import Georef
from aero_geoloc.types import LocalizationResult, Status
from aero_geoloc.viz import OverlayStyle, render_localization, save_localization_overlay

STYLE = OverlayStyle(panel_px=200, checker=40, header_px=50, footer_px=20)
GEOREF = Georef(37.6173, 55.7558, 18, 400, 400)


def _frame(width=320, height=240):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (height, width, 3), dtype=np.uint8)


def _reference(side=400):
    rng = np.random.default_rng(1)
    return rng.integers(0, 255, (side, side, 3), dtype=np.uint8)


def _result(**over):
    base = dict(
        status=Status.LOCALIZED,
        center_lat=55.7558,
        center_lon=37.6173,
        transform=np.array([[1.0, 0.0, 40.0], [0.0, 1.0, 30.0]]),
        footprint_lonlat=[(37.616, 55.755), (37.618, 55.755), (37.618, 55.757), (37.616, 55.757)],
        error_ellipse_m=(0.4, 0.3, 12.0),
        diagnostics={"n_inliers": 42, "photometric": 0.31},
    )
    base.update(over)
    return LocalizationResult(**base)


def test_overlay_has_four_panels_and_bars():
    """Ширина — четыре панели, высота — шапка + панель + подпись."""
    frame = _frame()
    out = render_localization(frame, _reference(), GEOREF, _result(), title="case", style=STYLE)
    panel_h = round(STYLE.panel_px * frame.shape[0] / frame.shape[1])
    assert out.shape[1] == 4 * STYLE.panel_px
    assert out.shape[0] == STYLE.header_px + panel_h + STYLE.footer_px
    assert out.dtype == np.uint8 and out.ndim == 3


def test_panels_follow_frame_aspect_not_square():
    """Панели идут в пропорциях кадра — иначе треть площади уходит в чёрные поля."""
    wide = render_localization(_frame(400, 200), _reference(), GEOREF, _result(), style=STYLE)
    tall = render_localization(_frame(200, 400), _reference(), GEOREF, _result(), style=STYLE)
    assert wide.shape[0] < tall.shape[0]


def test_overlay_survives_failed_result():
    """Отказ — легитимный исход: оверлей рисуется без позы и не падает."""
    failed = LocalizationResult.failed("нет устойчивой модели подобия")
    out = render_localization(_frame(), _reference(), GEOREF, failed, title="fail", style=STYLE)
    assert out.shape[1] == 4 * STYLE.panel_px
    assert out.shape[0] > STYLE.header_px  # шапка с причиной + пустые панели


def test_overlay_accepts_grayscale_inputs():
    gray_frame = np.full((240, 320), 128, np.uint8)
    gray_ref = np.full((400, 400), 90, np.uint8)
    out = render_localization(gray_frame, gray_ref, GEOREF, _result(), style=STYLE)
    assert out.shape[2] == 3  # на выходе всегда цветной холст


def test_overlay_with_truth_renders_same_shape():
    """Истина добавляет метку и ошибку в подпись, но не меняет раскладку."""
    without = render_localization(_frame(), _reference(), GEOREF, _result(), style=STYLE)
    with_truth = render_localization(
        _frame(), _reference(), GEOREF, _result(), truth_lat=55.7560, truth_lon=37.6175, style=STYLE
    )
    assert without.shape == with_truth.shape
    assert not np.array_equal(without, with_truth)  # метка истины реально нарисована


def test_save_writes_png(tmp_path):
    path = tmp_path / "sub" / "overlay.png"
    image = save_localization_overlay(
        path, _frame(), _reference(), GEOREF, _result(), title="saved", style=STYLE
    )
    assert path.exists() and path.stat().st_size > 0
    assert image.shape[1] == 4 * STYLE.panel_px


def test_status_color_differs_by_status():
    """Статус читается цветом заголовка — иначе оверлеи неразличимы с первого взгляда."""
    ok = render_localization(_frame(), _reference(), GEOREF, _result(), title="t", style=STYLE)
    low = render_localization(
        _frame(), _reference(), GEOREF, _result(status=Status.LOW_CONFIDENCE), title="t", style=STYLE
    )
    header_ok, header_low = ok[: STYLE.header_px], low[: STYLE.header_px]
    assert not np.array_equal(header_ok, header_low)
