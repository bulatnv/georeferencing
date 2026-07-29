"""Визуализация результата локализации — чтобы место можно было проверить глазами.

Зачем модуль ([EVAL_PLAN.md](../docs/EVAL_PLAN.md), блокер Б3): у георефа не было
ни одного способа посмотреть на результат. Нельзя было ни отладить провал, ни
подтвердить успех, ни разметить истину там, где её нет (снимки без GPS).

Что рисуется — четыре панели в один PNG:

1. **Кадр** — что искали.
2. **Подложка в геометрии кадра** — что нашли: окно подложки, отображённое
   найденным преобразованием в систему кадра. Если позиция верна, панели 1 и 2
   показывают одно и то же место (с точностью до сезона и даты съёмки).
3. **Наложение** — шахматная склейка тех же двух панелей: видно, сходятся ли
   дороги и здания на стыках клеток. Это проверка **точности**, тогда как панели
   1–2 отвечают на вопрос «то ли это вообще место».
4. **Контекст** — окно подложки целиком с контуром отпечатка кадра: где именно в
   районе поиска система поставила кадр.

Подпись несёт числа, по которым результат принимают или отвергают: статус, ошибка
против истины (если она есть), инлайеры, NCC, эллипс.

Текст рисуется через OpenCV (шрифт Hershey), а он не умеет кириллицу — поэтому
подписи латиницей, в отличие от остального вывода проекта.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geo import Georef, haversine_m
from .types import LocalizationResult, Status

__all__ = ["OverlayStyle", "render_localization", "save_localization_overlay"]

_FONT = cv2.FONT_HERSHEY_SIMPLEX
#: Цвета статусов (BGR): зелёный — принято, жёлтый — слабо, красный — отказ.
_STATUS_BGR = {
    Status.LOCALIZED: (80, 200, 80),
    Status.LOW_CONFIDENCE: (60, 200, 240),
    Status.NOT_LOCALIZED: (70, 70, 230),
}


@dataclass(frozen=True)
class OverlayStyle:
    """Оформление оверлея.

    Attributes:
        panel_px: сторона одной панели в пикселях.
        checker: размер клетки шахматной склейки, пикселей.
        header_px: высота полосы с подписью.
        footer_px: высота полосы с пояснением панелей.
    """

    panel_px: int = 420
    checker: int = 60
    header_px: int = 64
    footer_px: int = 26


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def _fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Вписать картинку в ``width×height``, сохранив пропорции (letterbox)."""
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), np.uint8)
    y0, x0 = (height - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
    canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    return canvas


def _checkerboard(a: np.ndarray, b: np.ndarray, cell: int) -> np.ndarray:
    """Шахматная склейка двух картинок — стыки показывают рассогласование."""
    out = a.copy()
    h, w = a.shape[:2]
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            if ((x // cell) + (y // cell)) % 2:
                out[y:y + cell, x:x + cell] = b[y:y + cell, x:x + cell]
    return out


def _label(canvas: np.ndarray, text: str, org: tuple[int, int], *,
           color: tuple[int, int, int] = (235, 235, 235), scale: float = 0.44) -> None:
    """Подпись с тёмной обводкой — читается на любом фоне."""
    cv2.putText(canvas, text, org, _FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, org, _FONT, scale, color, 1, cv2.LINE_AA)


def render_localization(
    frame: np.ndarray,
    reference: np.ndarray,
    reference_georef: Georef,
    result: LocalizationResult,
    *,
    title: str = "",
    truth_lat: float | None = None,
    truth_lon: float | None = None,
    style: OverlayStyle | None = None,
) -> np.ndarray:
    """Собрать оверлей результата локализации.

    Args:
        frame: кадр, который локализовали (BGR или grayscale).
        reference: окно подложки, против которого шёл матчинг.
        reference_georef: привязка этого окна — по ней контур отпечатка ложится
            в пиксели подложки.
        result: результат локализации (может быть и отказом — тогда рисуются
            только кадр и контекст, а панели совмещения пустые).
        title: имя кейса для подписи.
        truth_lat, truth_lon: истина, если известна, — тогда в подписи будет
            ошибка в метрах, а на контексте появится метка истины.
        style: оформление.

    Returns:
        Изображение оверлея (BGR).
    """
    style = style or OverlayStyle()
    side = style.panel_px
    frame_bgr, ref_bgr = _to_bgr(frame), _to_bgr(reference)
    # Панели идут в пропорциях кадра: у надирных снимков они 4:3 или шире, и
    # квадратные панели съедали бы треть площади чёрными полями.
    aspect = frame_bgr.shape[0] / frame_bgr.shape[1]
    panel_h = max(120, min(side, int(round(side * aspect))))

    panel_frame = _fit(frame_bgr, side, panel_h)
    if result.transform is not None:
        # Подложка → система кадра тем же преобразованием, что и в quality.aligned_ncc.
        warped = cv2.warpAffine(
            ref_bgr, np.asarray(result.transform, dtype=np.float32),
            (frame_bgr.shape[1], frame_bgr.shape[0]),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT,
        )
        panel_ref = _fit(warped, side, panel_h)
        panel_mix = _checkerboard(panel_frame, panel_ref, style.checker)
    else:
        panel_ref = np.zeros((panel_h, side, 3), np.uint8)
        panel_mix = np.zeros((panel_h, side, 3), np.uint8)
        _label(panel_ref, "no pose", (side // 2 - 40, panel_h // 2), color=(90, 90, 220), scale=0.6)

    # Контекст: всё окно подложки + контур отпечатка + метки центра и истины.
    context = _to_bgr(ref_bgr).copy()
    scale_ctx = side / max(context.shape[:2])
    context = cv2.resize(context, None, fx=scale_ctx, fy=scale_ctx, interpolation=cv2.INTER_AREA)

    def to_ctx(lon: float, lat: float) -> tuple[int, int] | None:
        px, py = reference_georef.lonlat_to_pixel(lon, lat)
        x, y = int(round(float(px) * scale_ctx)), int(round(float(py) * scale_ctx))
        if -10**6 < x < 10**6 and -10**6 < y < 10**6:
            return x, y
        return None

    if result.footprint_lonlat:
        pts = [to_ctx(lon, lat) for lon, lat in result.footprint_lonlat]
        if all(p is not None for p in pts):
            cv2.polylines(context, [np.array(pts, np.int32)], True, (80, 220, 80), 2, cv2.LINE_AA)
    if result.center_lat is not None:
        c = to_ctx(result.center_lon, result.center_lat)
        if c:
            cv2.drawMarker(context, c, (80, 220, 80), cv2.MARKER_CROSS, 18, 2)
    if truth_lat is not None and truth_lon is not None:
        t = to_ctx(truth_lon, truth_lat)
        if t:
            cv2.drawMarker(context, t, (60, 200, 240), cv2.MARKER_TILTED_CROSS, 18, 2)
    panel_ctx = _fit(context, side, panel_h)

    panels = [panel_frame, panel_ref, panel_mix, panel_ctx]
    captions = ["1. frame (query)", "2. basemap at found pose", "3. checkerboard blend",
                "4. context + footprint"]
    body = np.hstack(panels)

    header = np.full((style.header_px, body.shape[1], 3), 24, np.uint8)
    status_color = _STATUS_BGR.get(result.status, (200, 200, 200))
    _label(header, f"{title}  [{result.status.value}]", (12, 24), color=status_color, scale=0.6)

    diag = result.diagnostics or {}
    bits: list[str] = []
    if truth_lat is not None and truth_lon is not None and result.center_lat is not None:
        err = haversine_m(truth_lat, truth_lon, result.center_lat, result.center_lon)
        bits.append(f"error vs truth: {err:.1f} m")
    if result.center_lat is not None:
        bits.append(f"found: {result.center_lat:.6f}, {result.center_lon:.6f}")
    if diag.get("n_inliers") is not None:
        bits.append(f"inliers: {diag['n_inliers']}")
    if diag.get("photometric") is not None:
        bits.append(f"{diag.get('photometric_kind', 'NCC')}: "
                    f"{float(diag['photometric']):.3f}")
    # Эллипс — только у принятой позы. У отвергнутой это разброс подгонки, которая
    # проверку не прошла, и «ellipse: 0.45 m» рядом со словом low_confidence
    # читается как «зато очень точно» (поймано на DSC00045).
    if result.error_ellipse_m and result.status is Status.LOCALIZED:
        bits.append(f"ellipse: {result.error_ellipse_m[0]:.2f} m")
    if not result.is_localized and diag.get("reason"):
        bits.append(f"reason: {diag['reason']}")
    _label(header, "   ".join(bits), (12, 48))

    footer = np.full((style.footer_px, body.shape[1], 3), 24, np.uint8)
    for i, caption in enumerate(captions):
        _label(footer, caption, (12 + i * side, 18), color=(170, 170, 170), scale=0.4)

    out = np.vstack([header, body, footer])
    for i in range(1, len(panels)):  # разделители панелей
        cv2.line(out, (i * side, style.header_px), (i * side, style.header_px + panel_h),
                 (40, 40, 40), 1)
    return out


def save_localization_overlay(path, *args, **kwargs) -> np.ndarray:
    """Отрисовать оверлей и записать PNG. Аргументы — как у :func:`render_localization`."""
    from pathlib import Path

    image = render_localization(*args, **kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"не удалось записать оверлей: {path}")
    return image
