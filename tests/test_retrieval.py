"""Тесты retrieval-этажа (фаза 3, шаг 1): индекс, Recall@K, ротация, уникальность.

Проверяют обвязку на стенд-энкодере (``AveragePoolEncoder``) — как фаза 1
проверяла пайплайн на SIFT. Сильный энкодер (DINOv2) встанет за тем же
интерфейсом и поднимет Recall@K, не меняя ничего вокруг.
"""

from __future__ import annotations

import numpy as np
import pytest

import importlib.util

from aero_geoloc.retrieval import (
    AveragePoolEncoder,
    Cell,
    DinoV2Encoder,
    Encoder,
    MegaLocEncoder,
    RetrievalResult,
    TerrainIndex,
    calibrate_uniqueness_threshold,
    recall_at_k,
    should_localize,
)

_HAS_TORCH = importlib.util.find_spec("torch") is not None
from aero_geoloc.testbench import (
    SampleSpec,
    SceneBasemap,
    default_camera,
    generate_sample,
    make_homogeneous_scene,
    make_synthetic_scene,
)

CELL = 512


def _queries(scene, camera, offsets, *, seed=0, reference_size=1024):
    rng = np.random.default_rng(seed)
    out = []
    for ox in offsets:
        for oy in offsets:
            yaw = float(rng.uniform(0.0, 360.0))
            s = generate_sample(
                scene, camera, SampleSpec(yaw_deg=yaw, center_offset_px=(ox, oy)),
                reference_size=reference_size,
            )
            out.append((s.query, s.true_lat, s.true_lon, -yaw))  # доверяем yaw → предповорот
    return out


@pytest.fixture(scope="module")
def rich_scene():
    return make_synthetic_scene(3072, seed=0)


@pytest.fixture(scope="module")
def camera():
    return default_camera(512)


@pytest.fixture(scope="module")
def rich_index(rich_scene):
    bm = SceneBasemap(rich_scene)
    return TerrainIndex(AveragePoolEncoder(grid=24)).build(
        bm, rich_scene.georef, cell_size_px=CELL, overlap=0.5, rotations_deg=(0.0,)
    )


# --- энкодер ----------------------------------------------------------------


def test_encoder_output_is_normalized_and_deterministic():
    enc = AveragePoolEncoder(grid=16)
    img = make_synthetic_scene(256, seed=1).image
    v1 = enc.encode(img)
    v2 = enc.encode(img)
    assert v1.shape == (enc.dim,) == (256,)
    assert np.linalg.norm(v1) == pytest.approx(1.0, abs=1e-5)
    np.testing.assert_array_equal(v1, v2)


def test_encoder_rejects_bad_input():
    with pytest.raises(ValueError, match="grid"):
        AveragePoolEncoder(grid=1)
    with pytest.raises(ValueError, match="grayscale"):
        AveragePoolEncoder().encode(np.zeros((8, 8, 3), np.uint8))


# --- построение индекса -----------------------------------------------------


def test_index_build_produces_expected_cell_count(rich_scene):
    bm = SceneBasemap(rich_scene)
    idx = TerrainIndex(AveragePoolEncoder(24)).build(
        bm, rich_scene.georef, cell_size_px=CELL, overlap=0.5, rotations_deg=(0.0, 90.0)
    )
    # Сетка 512px с шагом 256 по стороне 3072: центры 256..2816 → 11×11 клеток,
    # ×2 ротации.
    assert len(idx) == 11 * 11 * 2
    assert all(isinstance(c, Cell) for c in idx._cells)


def test_index_build_rejects_bad_overlap(rich_scene):
    with pytest.raises(ValueError, match="overlap"):
        TerrainIndex(AveragePoolEncoder()).build(
            SceneBasemap(rich_scene), rich_scene.georef, cell_size_px=CELL, overlap=1.0
        )


# --- Recall@K (критерий приёмки шага) ---------------------------------------


def test_recall_at_k_high_for_correct_cell(rich_scene, camera, rich_index):
    queries = _queries(rich_scene, camera, offsets=[-600, -300, 0, 300, 600])
    # Верная клетка почти всегда в top-5, и часто уже в top-3 — retrieval не
    # теряет место (даже на слабом стенд-энкодере).
    assert recall_at_k(rich_index, queries, k=5, radius_m=130.0) >= 0.9
    assert recall_at_k(rich_index, queries, k=3, radius_m=130.0) >= 0.6
    assert recall_at_k(rich_index, queries, k=1, radius_m=130.0) >= 0.4


def test_rotation_augmentation_recovers_without_trusting_yaw(rich_scene, camera):
    """Аугментация индекса ротациями находит клетку без предповорота кадра."""
    bm = SceneBasemap(rich_scene)
    idx = TerrainIndex(AveragePoolEncoder(24)).build(
        bm, rich_scene.georef, cell_size_px=CELL, overlap=0.5,
        rotations_deg=(0.0, 90.0, 180.0, 270.0),
    )
    rng = np.random.default_rng(1)
    queries = []
    for ox in [-500, 0, 500]:
        for oy in [-500, 0, 500]:
            yaw = float(rng.choice([0.0, 90.0, 180.0, 270.0]))  # совпадает с аугментацией
            s = generate_sample(rich_scene, camera, SampleSpec(yaw_deg=yaw, center_offset_px=(ox, oy)),
                                reference_size=1024)
            queries.append((s.query, s.true_lat, s.true_lon, 0.0))  # БЕЗ предповорота
    assert recall_at_k(idx, queries, k=5, radius_m=130.0) >= 0.8


# --- сигнал уникальности ----------------------------------------------------


def test_uniqueness_higher_on_distinct_than_homogeneous(rich_scene, camera, rich_index):
    homo = make_homogeneous_scene(3072, seed=0)
    hidx = TerrainIndex(AveragePoolEncoder(24)).build(
        SceneBasemap(homo), homo.georef, cell_size_px=CELL, overlap=0.5
    )

    def median_uniqueness(scene, index):
        vals = []
        for q, _, _, prerot in _queries(scene, camera, offsets=[-500, -250, 0, 250, 500]):
            vals.append(index.query(q, k=5, prerotate_deg=prerot).uniqueness)
        return float(np.median(vals))

    rich_u = median_uniqueness(rich_scene, rich_index)
    homo_u = median_uniqueness(homo, hidx)
    # Самоподобная местность → далёкий двойник близок к top-1 → маленький зазор.
    # Сигнал слаб у стенд-энкодера, но направленно различает разрешимость;
    # сильный сигнал и калибровка порога → DINOv2 + шаг 2 фазы 3.
    assert rich_u > homo_u


def test_retrieval_result_carries_uniqueness_field():
    cells = [Cell(0, 0, 18, CELL), Cell(1, 1, 18, CELL)]
    r = RetrievalResult(cells, np.array([0.9, 0.6], np.float32), uniqueness=0.3)
    assert r.uniqueness == pytest.approx(0.3)
    assert r.best is cells[0]
    assert RetrievalResult([], np.empty((0,), np.float32)).best is None


def test_query_on_empty_index_returns_empty():
    idx = TerrainIndex(AveragePoolEncoder())
    r = idx.query(np.zeros((64, 64), np.uint8), k=5)
    assert r.cells == [] and r.similarities.size == 0
    assert r.best is None


# --- уникальность → флаг отказа (шаг 2) -------------------------------------


def test_uniqueness_threshold_separates_resolvable_terrain(rich_scene, camera, rich_index):
    """Калиброванный порог уникальности отделяет разрешимую местность от самоподобной."""
    homo = make_homogeneous_scene(3072, seed=0)
    hidx = TerrainIndex(AveragePoolEncoder(24)).build(
        SceneBasemap(homo), homo.georef, cell_size_px=CELL, overlap=0.5
    )
    offsets = [-500, -250, 0, 250, 500]
    rich_u = [rich_index.query(q, k=5, prerotate_deg=p).uniqueness
              for q, _, _, p in _queries(rich_scene, camera, offsets=offsets, seed=0)]
    homo_u = [hidx.query(q, k=5, prerotate_deg=p).uniqueness
              for q, _, _, p in _queries(homo, camera, offsets=offsets, seed=1)]

    u = np.array(rich_u + homo_u)
    y = np.array([True] * len(rich_u) + [False] * len(homo_u))
    threshold = calibrate_uniqueness_threshold(u, y)

    predicted = u >= threshold
    tpr = (predicted & y).sum() / y.sum()
    tnr = (~predicted & ~y).sum() / (~y).sum()
    assert (tpr + tnr) / 2.0 > 0.75  # балансная точность заметно выше случая
    assert np.median(homo_u) < threshold <= np.median(rich_u)


def test_should_localize_gate():
    cells = [Cell(0, 0, 18, CELL)]
    sims = np.array([0.9], np.float32)
    assert should_localize(RetrievalResult(cells, sims, uniqueness=0.3), min_uniqueness=0.1)
    assert not should_localize(RetrievalResult(cells, sims, uniqueness=0.05), min_uniqueness=0.1)
    assert not should_localize(RetrievalResult([], np.empty((0,), np.float32)), min_uniqueness=0.1)


def test_calibrate_uniqueness_threshold_degenerate_labels():
    # Все разрешимы (или все нет) — разделять нечего, порог 0 (ничего не отсекаем).
    assert calibrate_uniqueness_threshold([0.1, 0.2, 0.3], [True, True, True]) == 0.0
    assert calibrate_uniqueness_threshold([], []) == 0.0


# --- персистентность индекса (offline → runtime) ----------------------------


def test_index_save_load_roundtrip(rich_scene, camera, rich_index, tmp_path):
    path = tmp_path / "index.npz"
    rich_index.save(path)
    loaded = TerrainIndex.load(path, AveragePoolEncoder(24))

    assert len(loaded) == len(rich_index)
    # Загруженный индекс отвечает на запрос идентично исходному.
    q = _queries(rich_scene, camera, offsets=[0])[0]
    r0 = rich_index.query(q[0], k=5, prerotate_deg=q[3])
    r1 = loaded.query(q[0], k=5, prerotate_deg=q[3])
    assert [c.center_lat for c in r0.cells] == [c.center_lat for c in r1.cells]
    np.testing.assert_allclose(r0.similarities, r1.similarities, atol=1e-6)


def test_index_load_rejects_encoder_mismatch(rich_index, tmp_path):
    path = tmp_path / "index.npz"
    rich_index.save(path)
    with pytest.raises(ValueError, match="не совпадает"):
        TerrainIndex.load(path, AveragePoolEncoder(16))  # другая размерность


# --- DINOv2-энкодер (боевое ядро за интерфейсом Encoder) ---------------------


def test_dinov2_encoder_metadata_and_validation():
    enc = DinoV2Encoder()  # конструктор и dim не требуют torch
    assert enc.dim == 384
    assert isinstance(enc, Encoder)  # удовлетворяет протоколу
    with pytest.raises(ValueError, match="неизвестная модель"):
        DinoV2Encoder("resnet50")
    with pytest.raises(ValueError, match="кратен патчу 14"):
        DinoV2Encoder(image_size=100)


@pytest.mark.skipif(_HAS_TORCH, reason="torch установлен — боевой путь проверяется отдельно")
def test_dinov2_encode_without_torch_gives_clear_error():
    with pytest.raises(RuntimeError, match="torch"):
        DinoV2Encoder().encode(np.zeros((64, 64), np.uint8))


@pytest.mark.skipif(not _HAS_TORCH, reason="нужен torch (+ веса DINOv2 через torch.hub)")
def test_dinov2_encode_produces_normalized_vector():
    enc = DinoV2Encoder()
    v = enc.encode(make_synthetic_scene(256, seed=0).image)
    assert v.shape == (enc.dim,)
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-4)


# --- MegaLoc (VPR-энкодер Этажа 1) ------------------------------------------

_HAS_MEGALOC_DEPS = all(
    importlib.util.find_spec(m) is not None for m in ("torch", "huggingface_hub", "safetensors")
)


def test_megaloc_encoder_metadata_and_validation():
    enc = MegaLocEncoder()  # конструктор и dim не требуют torch
    assert enc.dim == 8448
    assert isinstance(enc, Encoder)  # удовлетворяет тому же протоколу, что DINOv2
    with pytest.raises(ValueError, match="кратен патчу 14"):
        MegaLocEncoder(image_size=100)


@pytest.mark.skipif(_HAS_TORCH, reason="torch установлен — боевой путь проверяется отдельно")
def test_megaloc_encode_without_torch_gives_clear_error():
    with pytest.raises(RuntimeError, match="torch"):
        MegaLocEncoder().encode(np.zeros((64, 64), np.uint8))


@pytest.mark.skipif(
    not _HAS_MEGALOC_DEPS, reason="нужны torch + huggingface_hub + safetensors (веса MegaLoc через torch.hub)"
)
def test_megaloc_encode_produces_normalized_vector():
    enc = MegaLocEncoder()
    v = enc.encode(make_synthetic_scene(256, seed=0).image)
    assert v.shape == (enc.dim,)
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-4)
