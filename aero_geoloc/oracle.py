"""Оракульное выравнивание: верная поза кадра, построенная БЕЗ матчера.

Зачем ([ROADMAP.md](../docs/ROADMAP.md), фаза 1). Пайплайн не даёт ни одной
верной кросс-сезонной позы — именно в этом проблема. Значит, изучать сигналы
качества «по результатам пайплайна» нельзя: выборка позитивов окажется ровно той,
где всё и так работает. Оракул решает это в лоб: у кейса есть истина, значит
верную позу можно **построить из неё**, а не найти.

Рецепт умышленно повторяет операцию пайплайна, а не изобретает конвенцию углов
заново: кадр приводится к разрешению подложки, поворачивается на ``−yaw`` (то
самое, что делает предповорот в :func:`aero_geoloc.localize._match_prerotated`) и
обрезается вписанным квадратом. Окно подложки берётся того же размера и с тем же
центром — после этого пиксель ``(i, j)`` обеих картинок соответствует одной точке
земли.

Откуда берутся центр и курс — см. :func:`alignment_for`.

Модулем пользуются оба эксперимента трека B: ``scripts/e1_signals.py`` (меры
согласия) и ``scripts/e2_geometry.py`` (геометрические сигналы).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .dataset import EvalCase
from .geo import haversine_m

__all__ = ["Alignment", "to_gray", "north_up_crop", "offset_lonlat", "alignment_for"]


@dataclass(frozen=True)
class Alignment:
    """Оракульная поза кадра: где центр, куда смотрит и откуда мы это знаем.

    Attributes:
        lat, lon: центр кадра на земле.
        yaw_deg: курс кадра относительно севера.
        source: ``exif`` — из метаданных снимка, ни одного пикселя подложки не
            использовано; ``pose`` — из позы, найденной пайплайном и сверенной с
            ручной истиной (для кадров без EXIF курс иначе не восстановить).
        drift_m: расхождение центра с истиной; у ``exif`` равно нулю.
    """

    lat: float
    lon: float
    yaw_deg: float
    source: str
    drift_m: float = 0.0


def to_gray(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def north_up_crop(frame: np.ndarray, yaw_deg: float) -> np.ndarray:
    """Кадр, повёрнутый к северу и обрезанный **вписанным** квадратом.

    Сторона квадрата ``min(W, H)/√2`` гарантированно лежит внутри исходного
    прямоугольника при любом угле поворота. Обрезка обязательна: чёрные углы,
    попав в статистику, подтягивают вверх любую меру согласия — два чёрных поля
    прекрасно коррелируют между собой.
    """
    gray = to_gray(frame)
    h, w = gray.shape[:2]
    centre = ((w - 1) / 2.0, (h - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, -yaw_deg, 1.0)
    rotated = cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_LINEAR)
    side = int(min(w, h) / math.sqrt(2.0))
    side -= side % 2
    x0, y0 = int(round(centre[0] - side / 2)), int(round(centre[1] - side / 2))
    return rotated[y0:y0 + side, x0:x0 + side]


def offset_lonlat(lat: float, lon: float, distance_m: float,
                  bearing_deg: float) -> tuple[float, float]:
    """Точка в ``distance_m`` метрах по азимуту ``bearing_deg`` — для ложных пар."""
    north = distance_m * math.cos(math.radians(bearing_deg))
    east = distance_m * math.sin(math.radians(bearing_deg))
    return (lat + north / 111320.0,
            lon + east / (111320.0 * math.cos(math.radians(lat))))


def alignment_for(case: EvalCase, poses: dict[str, tuple[float, float, float]],
                  *, tolerance_m: float = 150.0) -> Alignment | None:
    """Оракульная поза кейса либо ``None``, если построить её честно нельзя.

    ``exif``: центр — GPS кадра, курс — из метаданных. Подложка в построении не
    участвует вообще, матчер не вызывается. **Кросс-сезонные кейсы попадают
    именно сюда**, поэтому главная группа эксперимента чиста.

    ``manual``: курс по карте не восстановить — берётся у позы, найденной
    пайплайном. Чтобы «оракул» не выродился в очередной результат матчера, поза
    принимается только при совпадении её центра с ручной истиной; расхождение
    возвращается наружу и попадает в отчёт.
    """
    if not case.has_truth:
        return None
    if case.trust_yaw:
        return Alignment(case.truth_lat, case.truth_lon, case.prior.yaw_deg, "exif", 0.0)
    pose = poses.get(case.name)
    if pose is None:
        return None
    lat, lon, yaw = pose
    drift = haversine_m(case.truth_lat, case.truth_lon, lat, lon)
    if drift > tolerance_m:
        return None
    return Alignment(lat, lon, yaw, "pose", drift)
