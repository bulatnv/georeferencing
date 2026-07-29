"""Обзор снимков: регламент σ и правила построения готовой команды.

Файл-обзор владелец использует как единственный источник «что и как запускать»,
поэтому проверяется не форматирование, а три содержательных правила:

1. **приор в команде не равен истине** — иначе система ищет кадр там, куда ей же
   и указали, и успех ничего не доказывает;
2. **σ берётся не «с запасом»** — шире прямо хуже, и приоритет источников σ
   именно такой, потому что у кадров без EXIF высота в манифесте это заглушка;
3. **непригодный снимок не получает команду** — с названной причиной.

Всё офлайн, снимки не нужны: :class:`ShotMeta` собирается руками.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from exif_index import PRIOR_OFFSET_FRACTION, Known, block_for, command_for  # noqa: E402

from aero_geoloc.drone import ShotMeta  # noqa: E402
from aero_geoloc.request import PRIOR_ENVELOPE, recommended_sigma_m  # noqa: E402
from aero_geoloc.types import Prior  # noqa: E402

ROOT = Path("test_images")


def meta(**kw) -> ShotMeta:
    base = dict(path=ROOT / "DRZ" / "DRZ_00755.JPG", width=4000, height=3000,
                model="FC220", fov_deg=72.0, lat=51.216215, lon=6.169252,
                altitude_m=90.0, yaw_deg=-32.0, pitch_from_nadir_deg=0.1)
    return ShotMeta(**{**base, **kw})


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6_371_008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# --- регламент «высота → σ» ---------------------------------------------------

def test_envelope_nodes_are_reproduced_exactly():
    """Узлы — из таблицы в JOURNAL.md, а не «как получилось»."""
    for altitude, sigma in PRIOR_ENVELOPE:
        assert recommended_sigma_m(altitude) == pytest.approx(sigma)


def test_envelope_interpolates_between_nodes():
    assert recommended_sigma_m(200.0) == pytest.approx(1250.0)   # середина 100↔300


def test_envelope_clamps_outside():
    """Ниже и выше таблицы — крайние значения, а не экстраполяция в бессмыслицу."""
    assert recommended_sigma_m(10.0) == pytest.approx(PRIOR_ENVELOPE[0][1])
    assert recommended_sigma_m(5000.0) == pytest.approx(PRIOR_ENVELOPE[-1][1])


def test_envelope_grows_with_altitude():
    values = [recommended_sigma_m(h) for h in (50, 100, 200, 300, 450, 600, 900)]
    assert values == sorted(values)


# --- приор в команде не равен истине -----------------------------------------

def test_prior_is_offset_from_truth_by_half_sigma():
    """Главное правило файла: приором нельзя подсовывать ответ."""
    known = Known(meta())
    cmd = command_for(known, 0, root=ROOT)
    lat = float(cmd.split("--lat ")[1].split()[0])
    lon = float(cmd.split("--lon ")[1].split()[0])
    offset = haversine_m(known.lat, known.lon, lat, lon)
    assert offset == pytest.approx(known.sigma_m * PRIOR_OFFSET_FRACTION, rel=0.02)


def test_offset_prior_still_holds_the_truth_well_inside_the_gate():
    """Полсигмы: истина обязана остаться глубоко внутри диска ±3σ."""
    known = Known(meta())
    cmd = command_for(known, 0, root=ROOT)
    lat = float(cmd.split("--lat ")[1].split()[0])
    lon = float(cmd.split("--lon ")[1].split()[0])
    assert haversine_m(known.lat, known.lon, lat, lon) < 3.0 * known.sigma_m


def test_command_path_works_when_pointed_at_a_subfolder():
    """Путь в команде обязан открываться из корня репозитория.

    Раньше он «укорачивался» относительно родителя корня, и при
    ``--images test_images/DRZ`` команда получала ``--image DRZ/DRZ_00755.JPG`` —
    файла по такому пути нет.
    """
    known = Known(meta())
    for root in (Path("test_images"), Path("test_images/DRZ")):
        cmd = command_for(known, 0, root=root)
        assert "--image test_images/DRZ/DRZ_00755.JPG " in cmd


def test_offset_direction_varies_between_shots():
    """Одинаковый сдвиг на весь набор дал бы ему систематический перекос."""
    directions = {command_for(Known(meta()), i, root=ROOT).split("--gsd")[0]
                  for i in range(4)}
    assert len(directions) == 4


# --- σ: измеренное важнее регламента -----------------------------------------

class _Case:
    """Заглушка кейса манифеста: важны только поля, которые читает Known."""

    def __init__(self, sigma_m, altitude_m, gsd_m=0.065):
        self.prior = Prior(lat=51.0, lon=6.0, sigma_m=sigma_m, altitude_m=altitude_m,
                           altitude_sigma_m=50.0, yaw_deg=0.0)
        self.gsd_m = gsd_m
        self.has_truth = True
        self.truth_lat, self.truth_lon, self.truth_source = 51.5, 46.06, "manual"


def test_measured_sigma_wins_over_the_envelope():
    """Кадр без EXIF: «высота 500 м» в манифесте — заглушка, регламент по ней врёт.

    Регламент дал бы 4 км, тогда как прогонами подтверждено 1.5 км. Совет вчетверо
    шире измеренного — ровно та ошибка, от которой этот файл предостерегает.
    """
    known = Known(meta(lat=None, lon=None, altitude_m=None, fov_deg=None),
                  _Case(sigma_m=1500.0, altitude_m=500.0))
    assert known.sigma_m == 1500.0 and "прогон" in known.sigma_source
    assert recommended_sigma_m(500.0) > 1500.0        # регламент действительно шире


def test_envelope_applies_when_altitude_is_real():
    """У кадра с EXIF высота настоящая — работает регламент."""
    known = Known(meta())
    assert known.sigma_m == pytest.approx(500.0) and "регламент" in known.sigma_source


def test_truth_falls_back_to_the_manifest_without_exif():
    known = Known(meta(lat=None, lon=None), _Case(sigma_m=1500.0, altitude_m=500.0))
    assert (known.lat, known.lon) == (51.5, 46.06)
    assert "владельца" in known.truth_source


# --- непригодный снимок команды не получает ----------------------------------

def test_oblique_shot_gets_no_command():
    """Съёмка в горизонт вне надирной модели — запускать её незачем."""
    text = block_for(Known(meta(pitch_from_nadir_deg=90.0)), 0, root=ROOT)
    assert "НЕ ЗАПУСКАТЬ" in text and "python scripts/locate.py" not in text


def test_excluded_case_shows_the_manifest_reason():
    known = Known(meta())
    known.excluded_reason = "Нет GPS в EXIF и нет приблизительной точки"
    assert "Нет GPS" in block_for(known, 0, root=ROOT)


def test_shot_without_position_is_not_runnable():
    known = Known(meta(lat=None, lon=None))
    assert not known.runnable and command_for(known, 0, root=ROOT) is None


def test_runnable_block_shows_truth_and_command_separately():
    """Истина и приор в файле рядом — но это разные строки, и путать их нельзя."""
    text = block_for(Known(meta()), 0, root=ROOT)
    assert "ИСТИНА" in text and "51.216215" in text
    assert "51.216215" not in text.split("ЗАПУСК")[1]      # в команде истины нет
