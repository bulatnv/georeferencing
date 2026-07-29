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
    PCAReducer,
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


# --- батчинг кодирования (O2) ------------------------------------------------


class _BatchingStubEncoder:
    """Детерминированный энкодер с ``encode_batch`` — для проверки обвязки без GPU.

    Считает, сколько раз звали пачку и сколько поштучно: так видно, что сборка
    действительно идёт батчами, а не по одной клетке.
    """

    def __init__(self, grid: int = 8) -> None:
        self.inner = AveragePoolEncoder(grid)
        self.batch_calls = 0
        self.single_calls = 0

    @property
    def dim(self) -> int:
        return self.inner.dim

    def encode(self, gray):
        self.single_calls += 1
        return self.inner.encode(gray)

    def encode_batch(self, grays):
        grays = list(grays)
        self.batch_calls += 1
        return np.stack([self.inner.encode(g) for g in grays]) if grays else np.empty(
            (0, self.dim), np.float32
        )


def test_build_batches_when_encoder_supports_it(rich_scene):
    """Сборка идёт пачками, а не по одной клетке, если ядро умеет encode_batch."""
    enc = _BatchingStubEncoder()
    idx = TerrainIndex(enc).build(
        SceneBasemap(rich_scene), rich_scene.georef, cell_size_px=CELL, overlap=0.5, batch_size=8
    )
    assert len(idx) == 11 * 11
    assert enc.batch_calls > 0 and enc.single_calls == 0  # поштучный путь не звался
    assert enc.batch_calls <= (len(idx) + 7) // 8 + 1  # именно пачками, а не по одному


def test_batched_build_matches_single_build_exactly(rich_scene):
    """Батч не меняет ни векторы, ни порядок клеток — только скорость.

    Энкодер детерминированный (без GPU), поэтому сравнение точное: расхождение
    означало бы ошибку обвязки, а не шум вычислений.
    """
    bm = SceneBasemap(rich_scene)
    single = TerrainIndex(_BatchingStubEncoder()).build(
        bm, rich_scene.georef, cell_size_px=CELL, overlap=0.5,
        rotations_deg=(0.0, 90.0), batch_size=1,
    )
    batched = TerrainIndex(_BatchingStubEncoder()).build(
        bm, rich_scene.georef, cell_size_px=CELL, overlap=0.5,
        rotations_deg=(0.0, 90.0), batch_size=8,
    )
    assert len(single) == len(batched)
    assert [(c.center_lat, c.rotation_deg) for c in single._cells] == [
        (c.center_lat, c.rotation_deg) for c in batched._cells
    ]
    np.testing.assert_allclose(single._stacked(), batched._stacked(), atol=1e-6)


def test_build_falls_back_for_encoder_without_batch(rich_scene):
    """Ядро без encode_batch обязано работать по-прежнему: батч — не новый контракт."""
    idx = TerrainIndex(AveragePoolEncoder(24)).build(
        SceneBasemap(rich_scene), rich_scene.georef, cell_size_px=CELL, overlap=0.5, batch_size=8
    )
    assert len(idx) == 11 * 11  # тот же результат, просто поштучным путём


def test_build_rejects_bad_batch_size(rich_scene):
    with pytest.raises(ValueError, match="batch_size"):
        TerrainIndex(AveragePoolEncoder()).build(
            SceneBasemap(rich_scene), rich_scene.georef, cell_size_px=CELL, batch_size=0
        )


# --- PCA-редукция (память/скорость Этажа 1, масштаб на большие территории) ---


def test_pca_reducer_shapes_normalized_deterministic():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 128)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    red = PCAReducer(32, whiten=True, seed=0).fit(x)
    assert red.dim == 32 and red.input_dim == 128
    reduced = red.transform(x)
    assert reduced.shape == (200, 32)
    np.testing.assert_allclose(np.linalg.norm(reduced, axis=1), 1.0, atol=1e-5)  # L2-норма
    # Одиночный вектор — тот же путь, что строка матрицы.
    np.testing.assert_allclose(red.transform(x[0]), reduced[0], atol=1e-6)
    # Детерминизм по seed (те же параметры, включая whiten).
    np.testing.assert_allclose(
        PCAReducer(32, whiten=True, seed=0).fit(x).transform(x[0]), reduced[0], atol=1e-6
    )


def test_pca_reducer_clamps_components_to_rank():
    red = PCAReducer(1000).fit(np.eye(20, 40, dtype=np.float32))  # ранг ≤ 20
    assert red.dim <= 20


def test_pca_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="не обучен"):
        PCAReducer(8).transform(np.zeros((4,), np.float32))


def test_compress_preserves_recall(rich_scene, camera, rich_index):
    """PCA-сжатие индекса не роняет Recall — обвязка сохраняет различительность.

    Проверяем именно интеграцию редукции (plain PCA сохраняет топ-дисперсию).
    Whitening здесь выключен: на слабом стенд-энкодере с малой выборкой (121
    клетка) деление на плохо оценённые малые сингулярные значения раздувает шум —
    его выигрыш дескриптор-зависим и меряется на реальном MegaLoc, а не тут.
    """
    queries = _queries(rich_scene, camera, offsets=[-600, -300, 0, 300, 600])
    base = recall_at_k(rich_index, queries, k=5, radius_m=130.0)
    compressed = TerrainIndex(AveragePoolEncoder(grid=24)).build(
        SceneBasemap(rich_scene), rich_scene.georef, cell_size_px=CELL, overlap=0.5
    ).compress(48, whiten=False)
    assert compressed._reducer is not None and compressed._reducer.dim == 48
    assert recall_at_k(compressed, queries, k=5, radius_m=130.0) >= base - 0.1


def test_compressed_index_save_load_roundtrip(rich_scene, camera, tmp_path):
    idx = TerrainIndex(AveragePoolEncoder(grid=24)).build(
        SceneBasemap(rich_scene), rich_scene.georef, cell_size_px=CELL, overlap=0.5
    ).compress(48)
    path = tmp_path / "index_pca.npz"
    idx.save(path)
    loaded = TerrainIndex.load(path, AveragePoolEncoder(24))
    assert loaded._reducer is not None and loaded._reducer.dim == 48
    q = _queries(rich_scene, camera, offsets=[0])[0]
    r0 = idx.query(q[0], k=5, prerotate_deg=q[3])
    r1 = loaded.query(q[0], k=5, prerotate_deg=q[3])
    assert [c.center_lat for c in r0.cells] == [c.center_lat for c in r1.cells]
    np.testing.assert_allclose(r0.similarities, r1.similarities, atol=1e-5)


def test_compressed_load_rejects_encoder_mismatch(rich_scene, tmp_path):
    idx = TerrainIndex(AveragePoolEncoder(grid=24)).build(
        SceneBasemap(rich_scene), rich_scene.georef, cell_size_px=CELL, overlap=0.5
    ).compress(48)
    path = tmp_path / "index_pca.npz"
    idx.save(path)
    with pytest.raises(ValueError, match="не совпадает"):
        TerrainIndex.load(path, AveragePoolEncoder(16))  # другой вход PCA


# --- FAISS-движок поиска (сублинейно, для больших территорий) ----------------

_HAS_FAISS = importlib.util.find_spec("faiss") is not None


@pytest.mark.skipif(not _HAS_FAISS, reason="нужен faiss (pip install faiss-cpu)")
def test_faiss_flat_matches_numpy_exact(rich_scene, camera, rich_index):
    """FAISS Flat (точный IP) даёт то же top-K, что numpy-kNN — эталонная сверка."""
    idx = TerrainIndex(AveragePoolEncoder(grid=24)).build(
        SceneBasemap(rich_scene), rich_scene.georef, cell_size_px=CELL, overlap=0.5
    ).use_faiss(kind="flat")
    for q in _queries(rich_scene, camera, offsets=[-300, 0, 300]):
        r_np = rich_index.query(q[0], k=5, prerotate_deg=q[3])
        r_fa = idx.query(q[0], k=5, prerotate_deg=q[3])
        assert [c.center_lat for c in r_fa.cells] == [c.center_lat for c in r_np.cells]
        np.testing.assert_allclose(r_fa.similarities, r_np.similarities, atol=1e-5)


@pytest.mark.skipif(not _HAS_FAISS, reason="нужен faiss (pip install faiss-cpu)")
def test_faiss_hnsw_keeps_recall_with_pca(rich_scene, camera, rich_index):
    """Полный масштаб-стек PCA+FAISS/HNSW не роняет Recall против точного numpy."""
    queries = _queries(rich_scene, camera, offsets=[-600, -300, 0, 300, 600])
    base = recall_at_k(rich_index, queries, k=5, radius_m=130.0)
    idx = TerrainIndex(AveragePoolEncoder(grid=24)).build(
        SceneBasemap(rich_scene), rich_scene.georef, cell_size_px=CELL, overlap=0.5
    ).compress(48, whiten=False).use_faiss(kind="hnsw", ef_search=128)
    assert recall_at_k(idx, queries, k=5, radius_m=130.0) >= base - 0.1


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


@pytest.mark.skipif(
    not _HAS_MEGALOC_DEPS, reason="нужны torch + huggingface_hub + safetensors"
)
def test_megaloc_batch_matches_single():
    """У боевого ядра пачка даёт те же векторы, что и поштучный путь.

    Допуск свободный: батчевые ядра cuDNN считают в другом порядке, и расхождение
    порядка 1e-4 нормально. Существенно, что косинус ≈ 1 — на ранжирование клеток
    такое отличие не влияет.
    """
    enc = MegaLocEncoder()
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 255, (256, 256), dtype=np.uint8) for _ in range(4)]
    single = np.stack([enc.encode(i) for i in images])
    batch = enc.encode_batch(images)
    assert batch.shape == (4, enc.dim)
    np.testing.assert_allclose(np.linalg.norm(batch, axis=1), 1.0, atol=1e-4)
    assert float((single * batch).sum(axis=1).min()) > 0.9999


@pytest.mark.skipif(
    not _HAS_MEGALOC_DEPS, reason="нужны torch + huggingface_hub + safetensors"
)
def test_fp16_matches_fp32_vectors():
    """FP16 ускоряет, не меняя векторы: косинус с FP32 ≈ 1.

    Смешанная точность здесь — оптимизация, а не размен качества: расхождение
    (~1e-4) на порядок меньше зазора между близкими клетками, поэтому ранжирование
    не меняется. Тест держит это свойство, чтобы FP16 нельзя было включить ценой
    точности незаметно.
    """
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 255, (256, 256), dtype=np.uint8) for _ in range(4)]
    exact = MegaLocEncoder(fp16=False).encode_batch(images)
    fast = MegaLocEncoder(fp16=True).encode_batch(images)
    np.testing.assert_allclose(np.linalg.norm(fast, axis=1), 1.0, atol=1e-4)
    assert float((exact * fast).sum(axis=1).min()) > 0.999
