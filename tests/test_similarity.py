"""Меры согласия: проверяем ровно те свойства, ради которых они взяты.

Тесты аналитические. Главный из них — ``test_contrast_inversion``: он
воспроизводит суть сезонной проблемы в чистом виде (структура та же, связь
яркостей обратная) и требует, чтобы кандидаты её пережили, а база NCC — нет.
Если кандидат этот тест не проходит, брать его в трек B незачем.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from aero_geoloc.similarity import (
    SIGNALS,
    cfog,
    cfog_descriptor,
    edge_dice,
    grad_ncc,
    ncc,
    ngf,
    nmi,
)

#: Кандидаты, которые обязаны быть устойчивы к смене знака перепада.
STRUCTURAL = ("grad_ncc", "cfog", "ngf", "nmi", "edge_dice")


def scene(seed: int = 0, size: int = 192) -> np.ndarray:
    """Синтетическая «местность»: дороги, кварталы и текстура.

    Геометрия дорог тоже зависит от ``seed``: сцены с одинаковой сеткой улиц
    коррелируют по НАПРАВЛЕНИЯМ градиентов даже в разных местах, и «чужое место»
    получилось бы недостаточно чужим для мер, работающих с ориентациями.
    """
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 90, np.uint8)
    skew_x, skew_y = int(rng.integers(-40, 40)), int(rng.integers(-40, 40))
    for x in range(int(rng.integers(8, 30)), size, int(rng.integers(35, 60))):
        cv2.line(img, (x, 0), (x + skew_x, size), 200, 3)
    for y in range(int(rng.integers(8, 30)), size, int(rng.integers(30, 55))):
        cv2.line(img, (0, y), (size, y + skew_y), 170, 2)
    for _ in range(14):
        x, y = rng.integers(6, size - 34, 2)
        w, h = rng.integers(12, 30, 2)
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)),
                      int(rng.integers(40, 230)), -1)
    return cv2.GaussianBlur(img, (0, 0), 0.7)


IMG = scene()


@pytest.mark.parametrize("name", list(SIGNALS))
def test_identical_images_are_maximal(name):
    """Картинка сама с собой — верхняя точка шкалы каждой меры."""
    value = SIGNALS[name](IMG, IMG)
    expected = {"ncc": 1.0, "grad_ncc": 1.0, "cfog": 1.0, "ngf": 1.0,
                "nmi": 2.0, "edge_dice": 1.0}[name]
    assert value == pytest.approx(expected, abs=0.02)


@pytest.mark.parametrize("name", STRUCTURAL)
def test_contrast_inversion_survives(name):
    """Суть сезонной проблемы: структура та же, знак перепада обратный.

    Летом поле темнее дороги, весной светлее — край на месте, связь яркостей
    инвертирована. NCC при этом уходит в −1 (внизу проверяется отдельно), а
    структурные меры обязаны остаться на максимуме.
    """
    inverted = (255 - IMG.astype(np.int16)).astype(np.uint8)
    assert SIGNALS[name](IMG, inverted) > 0.9 * SIGNALS[name](IMG, IMG)


def test_ncc_is_destroyed_by_inversion():
    """База ломается именно там, где ломается на реальных данных."""
    inverted = (255 - IMG.astype(np.int16)).astype(np.uint8)
    assert ncc(IMG, inverted) == pytest.approx(-1.0, abs=0.01)


@pytest.mark.parametrize("name", list(SIGNALS))
def test_wrong_place_scores_lower_than_right_place(name):
    """Дискриминатор обязан отличать «то место» от «другого места».

    Это вторая работа NCC из обзора — та, ради которой мера вообще стоит в гейте.
    Мера, не проходящая этот тест, бесполезна независимо от сезонной стойкости.
    """
    other = scene(seed=7)
    assert SIGNALS[name](IMG, IMG) > SIGNALS[name](IMG, other)


@pytest.mark.parametrize("name", list(SIGNALS))
def test_monotone_brightness_change_is_tolerated(name):
    """Дымка/экспозиция: линейное преобразование яркости не должно ничего менять."""
    changed = np.clip(IMG.astype(np.float32) * 0.6 + 40, 0, 255).astype(np.uint8)
    assert SIGNALS[name](IMG, changed) > 0.85 * SIGNALS[name](IMG, IMG)


@pytest.mark.parametrize("name", list(SIGNALS))
def test_misalignment_lowers_score(name):
    """Мера должна падать со сдвигом — иначе она не годится и как метрика выравнивания."""
    shifted = np.roll(IMG, 12, axis=1)
    assert SIGNALS[name](IMG, shifted) < SIGNALS[name](IMG, IMG)


def test_cfog_floor_on_independent_images_is_near_zero():
    """Центрирование по каналам обязано вернуть мере рабочий диапазон.

    Проекции взяты по модулю, поэтому без вычитания среднего все дескрипторы
    лежат в положительном октанте и косинус любых двух ≈0.95: на реальных данных
    верная пара давала 0.958, а чужое место 0.950. Мера должна отделять ноль.
    """
    assert abs(cfog(IMG, scene(seed=13))) < 0.35
    assert cfog(IMG, IMG) > 0.95


def test_cfog_descriptor_is_unit_on_structure_and_zero_on_flat():
    """Плоский пиксель обязан остаться нулём, а не нормироваться в шум.

    Нормировка нулевого вектора породила бы «структуру» там, где её нет, и
    случайные направления в воде и полях начали бы совпадать сами собой.
    """
    desc = cfog_descriptor(IMG)
    assert desc.shape == (IMG.shape[0], IMG.shape[1], 8)
    norms = np.linalg.norm(desc, axis=-1)
    assert np.all((norms < 1e-6) | (np.abs(norms - 1.0) < 1e-3))
    assert (norms > 0.5).mean() > 0.5           # на сцене со структурой их большинство

    flat = np.full((64, 64), 100, np.uint8)
    assert np.all(np.linalg.norm(cfog_descriptor(flat), axis=-1) < 1e-6)


def test_ngf_floor_on_independent_images_is_one_half():
    """Пол NGF — 0.5, а не 0: средний cos² двух случайных направлений равен ½.

    Свойство неочевидное и прямо влияет на порог: «NGF 0.5» значит «связи нет»,
    и ставить границу принятия ниже неё бессмысленно.
    """
    flat = np.full_like(IMG, 128)
    noisy = (flat + np.random.default_rng(1).normal(0, 6, flat.shape)).astype(np.uint8)
    other = (flat + np.random.default_rng(2).normal(0, 6, flat.shape)).astype(np.uint8)
    assert ngf(noisy, other) == pytest.approx(0.5, abs=0.05)
    assert ngf(IMG, scene(seed=7)) < 0.75          # чужое место — заметно ближе к полу
    assert ngf(IMG, IMG) > 0.99                    # своё — на максимуме


def test_shape_mismatch_is_an_error():
    with pytest.raises(ValueError, match="размеры"):
        ncc(IMG, IMG[:100])


def test_colour_input_is_rejected():
    colour = cv2.cvtColor(IMG, cv2.COLOR_GRAY2BGR)
    with pytest.raises(ValueError, match="grayscale"):
        cfog(colour, colour)


@pytest.mark.parametrize("fn,lo,hi", [(cfog, -1.0, 1.0), (ngf, 0.0, 1.0),
                                      (nmi, 1.0, 2.0), (edge_dice, 0.0, 1.0),
                                      (ncc, -1.0, 1.0), (grad_ncc, -1.0, 1.0)])
def test_ranges_are_respected(fn, lo, hi):
    other = scene(seed=11)
    for value in (fn(IMG, IMG), fn(IMG, other)):
        assert lo - 1e-4 <= value <= hi + 1e-4
