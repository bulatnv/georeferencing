"""Загрузка внешних чекпоинтов (GIM/MINIMA) — трек A, фаза 2 из `docs/ROADMAP.md`.

Главное, что здесь проверяется, — **не «загрузилось», а «не загрузилось молча»**.
``load_state_dict(strict=False)`` стерпит любое несовпадение имён: сеть останется
на базовых весах, матчер отработает, A/B покажет «кандидат не помог», и вывод
будет сделан о кандидате, которого в вычислении не было. Тем же способом уже
дважды получался неверный вывод о LoFTR (`docs/JOURNAL.md`), поэтому загрузка
обязана падать, а не тихо откатываться.

Тесты работают на игрушечных модулях: ни сети, ни весов, ни torch-моделей.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from aero_geoloc.matcher import lightglue_state_dict  # noqa: E402
from aero_geoloc.weights import (  # noqa: E402
    CHECKPOINTS,
    CheckpointMismatch,
    apply_state_dict,
    checkpoint_path,
)


class Toy(torch.nn.Module):
    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.a = torch.nn.Linear(dim, dim)
        self.b = torch.nn.Linear(dim, dim)


# --- перекладка имён --------------------------------------------------------

def test_remap_merges_two_lists_into_one_block():
    """У GIM/MINIMA внимание разложено по двум спискам, у cvg — по одному блоку."""
    state = {
        "self_attn.0.Wqkv.weight": 1,
        "cross_attn.0.out_proj.bias": 2,
        "self_attn.11.ffn.0.weight": 3,
    }
    assert lightglue_state_dict(state) == {
        "transformers.0.self_attn.Wqkv.weight": 1,
        "transformers.0.cross_attn.out_proj.bias": 2,
        "transformers.11.self_attn.ffn.0.weight": 3,
    }


def test_remap_leaves_other_keys_alone():
    """Общие для обеих раскладок ветки трогать нельзя."""
    state = {"posenc.Wr.weight": 1, "log_assignment.0.matchability.weight": 2}
    assert lightglue_state_dict(state) == state


# --- проверка загрузки ------------------------------------------------------

def test_full_load_reports_the_number_of_tensors():
    model = Toy()
    donor = Toy()
    assert apply_state_dict(model, donor.state_dict(), label="донор") == 4
    assert torch.allclose(model.a.weight, donor.a.weight)


def test_partial_load_is_an_error_not_a_silent_fallback():
    """Половина имён не совпала — это другая архитектура, и молчать нельзя."""
    model = Toy()
    half = {k: v for k, v in Toy().state_dict().items() if k.startswith("a.")}
    with pytest.raises(CheckpointMismatch, match="покрыто 2 из 4"):
        apply_state_dict(model, half, label="половина")


def test_same_name_wrong_shape_does_not_count_as_matched():
    """Тензор того же имени, но другой формы — это не «частично подошло»."""
    model = Toy()
    other = Toy(dim=8).state_dict()
    with pytest.raises(CheckpointMismatch, match="несовпавшей формой"):
        apply_state_dict(model, other, label="другая размерность")


def test_error_names_what_to_do_next():
    """Сообщение должно называть причину, а не только факт."""
    with pytest.raises(CheckpointMismatch, match="код сети из репозитория"):
        apply_state_dict(Toy(), {}, label="пусто")


def test_computed_buffers_may_stay_uncovered():
    """LightGlue считает ``confidence_thresholds`` сам — 251/252 обязаны проходить."""
    model = Toy()
    donor = {k: v for k, v in Toy().state_dict().items() if k != "b.bias"}
    assert apply_state_dict(model, donor, label="почти всё", min_matched=0.7) == 3


# --- каталог чекпоинтов -----------------------------------------------------

@pytest.mark.parametrize("name", sorted(CHECKPOINTS))
def test_catalogue_records_licence_and_source(name):
    """Лицензия выясняется ДО интеграции, а не после победы кандидата."""
    spec = CHECKPOINTS[name]
    assert spec.licence and spec.source and spec.url.startswith("https://")
    assert spec.size_mb > 0


def test_missing_weights_without_network_name_the_url(tmp_path, monkeypatch):
    """Отказ качать — не «файл не найден», а «вот адрес, скачайте вручную»."""
    monkeypatch.setenv("AERO_WEIGHTS_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="https://"):
        checkpoint_path("minima_loftr", allow_download=False)


def test_dense_matcher_is_registered_like_any_other_core():
    """Смена ядра на плотное остаётся одной строкой конфигурации.

    Инвариант сменного ядра: всё выше матчера (pose, quality, георефа, стенд) не
    должно знать, разреженный он или плотный.
    """
    from aero_geoloc.matcher import RoMaMatcher, create_matcher

    m = create_matcher("minima_roma")
    assert isinstance(m, RoMaMatcher)
    assert m.checkpoint == "minima_roma"
    assert create_matcher("roma").checkpoint is None      # штатные веса RoMa


def test_roma_checkpoint_declares_what_is_and_is_not_inside():
    """Замороженный DINOv2 в чекпоинт не входит — это надо знать до отладки.

    Иначе «в файле всего 445 МБ» выглядит как повод заподозрить обрезанную
    загрузку, а не как штатное устройство RoMa.
    """
    spec = CHECKPOINTS["minima_roma"]
    assert "DINOv2" in spec.note
    assert spec.licence == "Apache-2.0"


def test_unknown_checkpoint_lists_the_known_ones():
    with pytest.raises(ValueError, match="доступны"):
        checkpoint_path("несуществующий")
