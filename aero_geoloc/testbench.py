"""Синтетический стенд: генерация примеров с ground truth, метрики, прогон сетки.

Ядро валидации из ``docs/TESTING.md``. Идея: вырезаем query из подложки в
**известном** центре с известными поворотом и масштабом — и меряем, насколько
точно пайплайн восстановил истину. Ground truth есть без всякой разметки.

Принцип «стенд раньше кода фазы»: метрика точности существует до того, как мы
усложняем алгоритм, иначе «улучшения» неизмеримы.

Фаза 1 покрывает геометрические возмущения (поворот, масштаб, сдвиг центра).
Лестница возмущений внешнего вида L1–L6 — фаза 2, поэтому :class:`SampleSpec`
уже разведён на геометрию и внешний вид, чтобы её было куда доклеить.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator, Sequence

import cv2
import numpy as np

from .camera import Camera
from .geo import Georef, ground_mpp, haversine_m
from .localize import localize_against_reference
from .matcher import Matcher
from .types import LocalizationResult, Prior, Status

__all__ = [
    "Scene",
    "make_synthetic_scene",
    "make_homogeneous_scene",
    "SceneBasemap",
    "default_camera",
    "SampleSpec",
    "APPEARANCE_LEVELS",
    "apply_appearance",
    "iter_specs",
    "Sample",
    "generate_sample",
    "SampleMetrics",
    "evaluate",
    "GridSummary",
    "run_grid",
]


# --- сцена ------------------------------------------------------------------


@dataclass(frozen=True)
class Scene:
    """Большой растр подложки с известной привязкой — источник всех примеров."""

    image: np.ndarray
    georef: Georef

    def __post_init__(self) -> None:
        h, w = self.image.shape[:2]
        if (w, h) != (self.georef.width, self.georef.height):
            raise ValueError(
                f"растр {w}×{h} не совпадает с Georef "
                f"{self.georef.width}×{self.georef.height}"
            )

    @property
    def mpp(self) -> float:
        return self.georef.mpp

    def window(self, center_px: float, center_py: float, size: int) -> tuple[np.ndarray, Georef]:
        """Вырезать квадратное окно подложки — аналог «уже скачанного растра».

        В фазе 1 сети нет, поэтому окно вокруг приора играет роль того, что в
        фазе 2 будет тянуть ``basemap.py``.
        """
        x0 = int(round(center_px - (size - 1) / 2.0))
        y0 = int(round(center_py - (size - 1) / 2.0))
        h, w = self.image.shape[:2]
        if not (0 <= x0 and 0 <= y0 and x0 + size <= w and y0 + size <= h):
            raise ValueError(
                f"окно {size}×{size} в ({center_px:.1f}, {center_py:.1f}) "
                f"не помещается в растр {w}×{h}"
            )
        return self.image[y0 : y0 + size, x0 : x0 + size].copy(), self.georef.crop(
            x0, y0, size, size
        )


def make_synthetic_scene(
    size: int = 2048,
    *,
    seed: int = 0,
    center_lon: float = 37.6173,
    center_lat: float = 55.7558,
    zoom: int = 18,
) -> Scene:
    """Процедурная «подложка»: многооктавный шум плюс дороги и постройки.

    Настоящий растр подложки в фазе 1 не нужен и вреден — он требует сети и
    привязывает тесты к внешнему сервису. Процедурная сцена детерминирована по
    ``seed``, богата на структуру всех масштабов (SIFT находит на ней тысячи
    точек) и не самоподобна, то есть проверяет геометрию, а не устойчивость к
    неоднозначности — этим займётся фаза 2.
    """
    rng = np.random.default_rng(seed)

    # Многооктавный value noise: крупные пятна ландшафта + мелкая текстура.
    field_ = np.zeros((size, size), dtype=np.float32)
    amplitude = 1.0
    for octave in range(2, 8):
        cells = 2**octave
        coarse = rng.random((cells, cells), dtype=np.float32)
        field_ += amplitude * cv2.resize(coarse, (size, size), interpolation=cv2.INTER_CUBIC)
        amplitude *= 0.55

    # Дорожная сеть — длинные светлые линии, дающие устойчивые пересечения.
    roads = np.zeros((size, size), dtype=np.float32)
    for _ in range(18):
        p0 = rng.integers(0, size, size=2)
        angle = rng.uniform(0.0, math.pi)
        length = rng.integers(size // 3, size)
        p1 = p0 + (length * np.array([math.cos(angle), math.sin(angle)])).astype(int)
        cv2.line(roads, tuple(p0.tolist()), tuple(p1.tolist()), 1.0, int(rng.integers(2, 5)))

    # Постройки и поля — прямоугольные пятна с чёткими углами.
    blocks = np.zeros((size, size), dtype=np.float32)
    for _ in range(220):
        x, y = rng.integers(0, size, size=2)
        w, h = rng.integers(size // 90, size // 14, size=2)
        angle = rng.uniform(0.0, 180.0)
        box = cv2.boxPoints(((float(x), float(y)), (float(w), float(h)), float(angle)))
        cv2.fillPoly(blocks, [box.astype(np.int32)], float(rng.uniform(-0.6, 0.6)))

    combined = field_ + 0.35 * roads + 0.5 * blocks
    combined = cv2.normalize(combined, None, 0.0, 255.0, cv2.NORM_MINMAX)
    image = combined.astype(np.uint8)

    georef = Georef(
        center_lon=center_lon, center_lat=center_lat, zoom=zoom, width=size, height=size
    )
    return Scene(image=image, georef=georef)


def make_homogeneous_scene(
    size: int = 2048,
    *,
    seed: int = 0,
    center_lon: float = 37.6173,
    center_lat: float = 55.7558,
    zoom: int = 18,
) -> Scene:
    """Бедная текстурой, почти самоподобная «подложка» — поле/лес/вода.

    Гладкий низкоконтрастный шум без дорог и построек: устойчивых особых точек
    почти нет, локализация обязана честно отказать (``NOT_LOCALIZED``), а не
    выдать уверенно-неверную точку. Это регресс на однородной местности из
    ``docs/TESTING.md``, антипод богатой :func:`make_synthetic_scene`.
    """
    rng = np.random.default_rng(seed)
    coarse = rng.random((8, 8), dtype=np.float32)
    field_ = cv2.resize(coarse, (size, size), interpolation=cv2.INTER_CUBIC)
    field_ = cv2.GaussianBlur(field_, (0, 0), sigmaX=size / 64.0, sigmaY=size / 64.0)
    image = cv2.normalize(field_, None, 100.0, 155.0, cv2.NORM_MINMAX).astype(np.uint8)
    georef = Georef(
        center_lon=center_lon, center_lat=center_lat, zoom=zoom, width=size, height=size
    )
    return Scene(image=image, georef=georef)


@dataclass(frozen=True)
class SceneBasemap:
    """:class:`~aero_geoloc.basemap.BasemapSource` поверх процедурной сцены.

    Позволяет прогонять полноценный ``localize`` (с coarse-to-fine) **офлайн**:
    оркестрация запрашивает окно как у сети, а данные режутся из сцены.

    На **родном зуме сцены** — точный целочисленный кроп (пиксель-в-пиксель,
    чтобы ground truth точного уровня оставался честным). На **более грубом
    зуме** — кроп соответствующего геоучастка и даунсемпл до запрошенного
    размера (грубому уровню субпиксельная точность не нужна). Зум детальнее
    сцены недоступен — данных детальнее у процедурной сцены нет.
    """

    scene: Scene

    def _crop_bounds(self, center_px: float, size: int) -> int:
        """Левая/верхняя координата целочисленного кропа сцены по центру."""
        return int(round(center_px - (size - 1) / 2.0))

    def _check_inside(self, x0: int, y0: int, w: int, h: int) -> None:
        sh, sw = self.scene.image.shape[:2]
        if not (0 <= x0 and 0 <= y0 and x0 + w <= sw and y0 + h <= sh):
            raise ValueError(
                f"окно {w}×{h} в ({x0}, {y0}) не помещается в сцену {sw}×{sh}: "
                "увеличьте сцену либо сузьте приор/грубый уровень"
            )

    def __call__(
        self, center_lon: float, center_lat: float, zoom: int, width: int, height: int
    ) -> tuple[np.ndarray, Georef]:
        gr = self.scene.georef
        if zoom > gr.zoom:
            raise ValueError(
                f"SceneBasemap: запрошен зум {zoom} детальнее сцены {gr.zoom} — нет данных"
            )
        scx, scy = gr.lonlat_to_pixel(center_lon, center_lat)

        if zoom == gr.zoom:
            x0 = self._crop_bounds(scx, width)
            y0 = self._crop_bounds(scy, height)
            self._check_inside(x0, y0, width, height)
            image = self.scene.image[y0 : y0 + height, x0 : x0 + width].copy()
            return image, gr.crop(x0, y0, width, height)

        # Грубее сцены: берём в 2^Δ раз больший участок сцены и даунсемплим.
        factor = 2 ** (gr.zoom - zoom)
        w_src, h_src = width * factor, height * factor
        x0 = self._crop_bounds(scx, w_src)
        y0 = self._crop_bounds(scy, h_src)
        self._check_inside(x0, y0, w_src, h_src)
        src = self.scene.image[y0 : y0 + h_src, x0 : x0 + w_src]
        src_georef = gr.crop(x0, y0, w_src, h_src)
        coarse = cv2.resize(src, (width, height), interpolation=cv2.INTER_AREA)
        georef = Georef(
            center_lon=src_georef.center_lon,
            center_lat=src_georef.center_lat,
            zoom=zoom,
            width=width,
            height=height,
        )
        return coarse, georef


# --- генерация примера ------------------------------------------------------


@dataclass(frozen=True)
class SampleSpec:
    """Что именно возмущаем в одном примере.

    Attributes:
        yaw_deg: истинный курс кадра (он же поворот преобразования кадр→подложка).
        altitude_ratio: во сколько раз истинная высота отличается от приорной,
            то есть имитация ошибки высоты. 1.0 — приор точен.
        center_offset_px: сдвиг истинного центра от центра сцены, пиксели подложки.
        prior_offset_m: насколько промахивается приор позиции, метры.
        appearance_level: уровень лестницы возмущений внешнего вида
            (``docs/TESTING.md``): L0 нет, L1 яркость/контраст/гамма, L2 блюр/
            шум/JPEG, L3 спектральный сдвиг, L4 локальные изменения объектов.
        appearance_strength: сила возмущения выбранного уровня (0 = нет, 1 =
            номинал). Ею свипается «лестница» до точки перелома матчера.
    """

    yaw_deg: float = 0.0
    altitude_ratio: float = 1.0
    center_offset_px: tuple[float, float] = (0.0, 0.0)
    prior_offset_m: float = 0.0
    prior_offset_bearing_deg: float = 0.0
    appearance_level: int = 0
    appearance_strength: float = 1.0


APPEARANCE_LEVELS = (0, 1, 2, 3, 4)


def _appearance_seed(spec: SampleSpec) -> int:
    """Детерминированный seed возмущений из содержимого spec — воспроизводимо и разнообразно."""
    raw = (
        spec.yaw_deg * 1000.0
        + spec.altitude_ratio * 733.0
        + spec.prior_offset_m * 13.0
        + spec.appearance_level * 7919.0
        + spec.appearance_strength * 131.0
    )
    return int(abs(raw)) & 0xFFFFFFFF


def apply_appearance(
    image: np.ndarray, level: int, strength: float, rng: np.random.Generator
) -> np.ndarray:
    """Наложить возмущение внешнего вида уровня L``level`` на кадр (grayscale uint8).

    Возмущения имитируют разрыв «борт ↔ спутник» и накладываются ТОЛЬКО на query:
    подложка остаётся эталонной, отсюда и appearance gap. ``strength`` масштабирует
    силу (0 → тождество). Уровни — из таблицы ``docs/TESTING.md``.
    """
    if level == 0 or strength <= 0.0:
        return image
    out = image.astype(np.float32)

    if level == 1:  # экспозиция/время суток: гамма, контраст, яркость
        gamma = 2.0 ** (0.7 * strength * (2.0 * rng.random() - 1.0))
        out = 255.0 * np.power(np.clip(out / 255.0, 0.0, 1.0), gamma)
        alpha = 1.0 + 0.35 * strength * (2.0 * rng.random() - 1.0)
        beta = 30.0 * strength * (2.0 * rng.random() - 1.0)
        out = alpha * out + beta

    elif level == 2:  # оптика/сжатие/сенсор: блюр, шум, JPEG
        sigma = 1.8 * strength
        if sigma > 0.3:
            out = cv2.GaussianBlur(out, (0, 0), sigmaX=sigma, sigmaY=sigma)
        out = out + rng.normal(0.0, 14.0 * strength, size=out.shape).astype(np.float32)
        out = np.clip(out, 0.0, 255.0).astype(np.uint8)
        quality = int(np.clip(95.0 - 75.0 * strength, 12.0, 95.0))
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, quality])
        out = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    elif level == 3:  # сезон/другой сенсор: нелинейный спектральный ремап тона
        # Синусоидальный сдвиг тона (фиксированная частота, амплитуда ∝ strength)
        # локально разворачивает градиенты, как смена спектрального канала —
        # к чему градиентные дескрипторы заведомо чувствительны. Монотонен по силе.
        x = out / 255.0
        x = np.clip(x + strength * 0.35 * np.sin(2.0 * np.pi * 1.5 * x), 0.0, 1.0)
        out = 255.0 * x

    elif level == 4:  # возраст подложки: подрисованные/убранные объекты
        out = out.copy()
        h, w = out.shape
        for _ in range(int(round(6.0 * strength))):  # новые объекты
            x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
            bw, bh = int(rng.integers(w // 20, w // 6)), int(rng.integers(h // 20, h // 6))
            cv2.rectangle(out, (x, y), (min(x + bw, w - 1), min(y + bh, h - 1)),
                          float(rng.uniform(0.0, 255.0)), -1)
        for _ in range(int(round(4.0 * strength))):  # стёртые объекты → локальное среднее
            x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
            bw, bh = int(rng.integers(w // 20, w // 8)), int(rng.integers(h // 20, h // 8))
            patch = out[y : min(y + bh, h), x : min(x + bw, w)]
            if patch.size:
                out[y : min(y + bh, h), x : min(x + bw, w)] = float(patch.mean())
    else:
        raise ValueError(f"неизвестный уровень возмущения L{level}, доступны {APPEARANCE_LEVELS}")

    return np.clip(out, 0.0, 255.0).astype(np.uint8)


@dataclass(frozen=True)
class Sample:
    """Сгенерированный пример вместе с истиной и вырезанным окном подложки."""

    query: np.ndarray
    camera: Camera
    prior: Prior
    reference: np.ndarray
    reference_georef: Georef
    spec: SampleSpec

    # Ground truth.
    true_lat: float
    true_lon: float
    true_heading_deg: float
    true_altitude_m: float
    true_scale: float
    true_center_ref_px: tuple[float, float]
    reference_mpp: float


#: Разрешение подложки, под которое настроена камера стенда (z18, широта Москвы).
REFERENCE_MPP = ground_mpp(55.7558, 18)


def default_camera(size: int = 512, *, altitude_m: float = 600.0) -> Camera:
    """Камера стенда: на высоте ``altitude_m`` её GSD равен mpp подложки.

    Масштаб преобразования получается ≈ 1, то есть кадр и подложка сопоставимы
    по детальности — режим, ради которого в фазе 2 будет подбираться зум.

    Фокус в пикселях **не зависит от размера кадра**: меньший кадр — это более
    узкое поле зрения той же камеры, а не другая оптика. Иначе уменьшение
    ``size`` втихую меняло бы масштаб и ломало геометрию примеров.
    """
    focal_px = altitude_m / REFERENCE_MPP
    focal_mm = 28.0
    return Camera(
        image_width=size,
        image_height=size,
        focal_mm=focal_mm,
        sensor_width_mm=focal_mm * size / focal_px,
    )


def generate_sample(
    scene: Scene,
    camera: Camera,
    spec: SampleSpec,
    *,
    prior_altitude_m: float = 600.0,
    prior_sigma_m: float = 200.0,
    altitude_sigma_m: float = 100.0,
    reference_size: int = 1280,
) -> Sample:
    """Вырезать query в известном центре с известными поворотом и масштабом.

    Истинное преобразование «кадр → подложка» строится явно:
    ``M(u) = s·R(θ)·(u − c_q) + c_ref``. Именно его и обязан восстановить
    пайплайн, поэтому ошибка стенда сводится только к интерполяции.

    Args:
        spec: возмущения примера.
        prior_altitude_m: высота, которую «знает» борт; истинная равна
            ``prior_altitude_m · spec.altitude_ratio``.
        prior_sigma_m: σ приора позиции (в фазе 1 приор узкий).
        reference_size: размер вырезаемого окна подложки.
    """
    if spec.appearance_level not in APPEARANCE_LEVELS:
        raise ValueError(
            f"неизвестный уровень L{spec.appearance_level}, доступны {APPEARANCE_LEVELS}"
        )

    scene_cx, scene_cy = scene.georef.center_pixel
    true_cx = scene_cx + spec.center_offset_px[0]
    true_cy = scene_cy + spec.center_offset_px[1]

    true_altitude_m = prior_altitude_m * spec.altitude_ratio
    # Масштаб: 1 пиксель кадра = GSD метров = GSD/mpp пикселей подложки.
    true_scale = camera.gsd(true_altitude_m) / scene.mpp

    # Окно подложки вокруг ПРИОРНОГО центра — борт не знает истинного.
    # Азимут отсчитывается от севера по часовой; в растре север — это −y.
    offset_px = spec.prior_offset_m / scene.mpp
    bearing = math.radians(spec.prior_offset_bearing_deg)
    prior_cx = true_cx + offset_px * math.sin(bearing)
    prior_cy = true_cy - offset_px * math.cos(bearing)
    reference, ref_georef = scene.window(prior_cx, prior_cy, reference_size)

    # Кадр обязан целиком лежать внутри окна: иначе часть «истины» окажется
    # выдуманной интерполяцией краёв, и метрика перестанет что-либо значить.
    half_diag = 0.5 * true_scale * math.hypot(camera.image_width, camera.image_height)
    margin = reference_size / 2.0 - (half_diag + offset_px)
    if margin < 0.0:
        raise ValueError(
            f"кадр не помещается в окно подложки: не хватает {-margin:.0f} px. "
            f"Увеличьте reference_size (сейчас {reference_size}) либо уменьшите "
            f"altitude_ratio / prior_offset_m."
        )

    # Истинное преобразование кадр → окно подложки.
    theta = math.radians(spec.yaw_deg)
    lin = true_scale * np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )
    c_q = np.array([(camera.image_width - 1) / 2.0, (camera.image_height - 1) / 2.0])
    true_ref_px = np.asarray(ref_georef.lonlat_to_pixel(*scene.georef.pixel_to_lonlat(true_cx, true_cy)))
    m = np.hstack([lin, (true_ref_px - lin @ c_q).reshape(2, 1)])

    # Даунсемплинг с true_scale > 1 даёт алиасинг — гасим его лёгким размытием,
    # как это делает оптика реальной камеры.
    source = reference
    if true_scale > 1.05:
        sigma = 0.5 * math.sqrt(true_scale**2 - 1.0)
        source = cv2.GaussianBlur(reference, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # WARP_INVERSE_MAP: dst(u) = src(M·u), то есть query собирается ровно по M.
    query = cv2.warpAffine(
        source,
        m,
        (camera.image_width, camera.image_height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # Возмущение внешнего вида — только на query (подложка эталонна → appearance gap).
    query = apply_appearance(
        query, spec.appearance_level, spec.appearance_strength,
        np.random.default_rng(_appearance_seed(spec)),
    )

    true_lon, true_lat = scene.georef.pixel_to_lonlat(true_cx, true_cy)
    # Координаты приора берём из той же Georef, что и позицию окна, — так
    # заявленный промах приора и фактический сдвиг окна не могут разъехаться.
    prior_lon, prior_lat = scene.georef.pixel_to_lonlat(prior_cx, prior_cy)

    prior = Prior(
        lat=float(prior_lat),
        lon=float(prior_lon),
        sigma_m=prior_sigma_m,
        altitude_m=prior_altitude_m,
        altitude_sigma_m=altitude_sigma_m,
        yaw_deg=spec.yaw_deg,
    )

    return Sample(
        query=query,
        camera=camera,
        prior=prior,
        reference=reference,
        reference_georef=ref_georef,
        spec=spec,
        true_lat=float(true_lat),
        true_lon=float(true_lon),
        true_heading_deg=spec.yaw_deg % 360.0,
        true_altitude_m=true_altitude_m,
        true_scale=true_scale,
        true_center_ref_px=(float(true_ref_px[0]), float(true_ref_px[1])),
        reference_mpp=ref_georef.mpp,
    )


# --- метрики ----------------------------------------------------------------


@dataclass(frozen=True)
class SampleMetrics:
    """Ошибки одного примера. Главная метрика — ``center_error_m``."""

    localized: bool
    center_error_m: float
    center_error_px: float
    heading_error_deg: float
    scale_error_pct: float
    altitude_error_pct: float
    n_inliers: int
    inlier_ratio: float
    reprojection_rmse_px: float
    status: Status
    reason: str | None = None


def evaluate(result: LocalizationResult, sample: Sample) -> SampleMetrics:
    """Сравнить результат локализации с истиной примера."""
    if not result.is_localized:
        nan = float("nan")
        return SampleMetrics(
            localized=False,
            center_error_m=nan,
            center_error_px=nan,
            heading_error_deg=nan,
            scale_error_pct=nan,
            altitude_error_pct=nan,
            n_inliers=int(result.diagnostics.get("n_inliers", 0)),
            inlier_ratio=float(result.diagnostics.get("inlier_ratio", 0.0)),
            reprojection_rmse_px=float(result.diagnostics.get("reprojection_rmse_px", nan)),
            status=result.status,
            reason=result.diagnostics.get("reason"),
        )

    center_error_m = float(
        haversine_m(sample.true_lat, sample.true_lon, result.center_lat, result.center_lon)
    )
    heading_error = abs((result.heading_deg - sample.true_heading_deg + 180.0) % 360.0 - 180.0)
    est_scale = float(result.diagnostics["scale"])

    return SampleMetrics(
        localized=True,
        center_error_m=center_error_m,
        center_error_px=center_error_m / sample.reference_mpp,
        heading_error_deg=heading_error,
        scale_error_pct=100.0 * (est_scale / sample.true_scale - 1.0),
        altitude_error_pct=100.0 * (result.altitude_est_m / sample.true_altitude_m - 1.0),
        n_inliers=int(result.diagnostics.get("n_inliers", 0)),
        inlier_ratio=float(result.diagnostics.get("inlier_ratio", 0.0)),
        reprojection_rmse_px=float(result.diagnostics.get("reprojection_rmse_px", float("nan"))),
        status=result.status,
    )


# --- прогон сетки -----------------------------------------------------------


@dataclass(frozen=True)
class GridSummary:
    """Агрегат по прогону. Медиана, а не среднее: провалы не должны её тянуть."""

    n_samples: int
    n_localized: int
    success_rate: float
    median_center_error_m: float
    p90_center_error_m: float
    max_center_error_m: float
    median_center_error_px: float
    median_heading_error_deg: float
    max_heading_error_deg: float
    median_scale_error_pct: float
    max_abs_scale_error_pct: float
    median_altitude_error_pct: float
    metrics: list[SampleMetrics] = field(repr=False, default_factory=list)

    def format_report(self) -> str:
        """Человекочитаемая сводка для CLI и логов."""
        return "\n".join(
            [
                f"примеров:               {self.n_samples}",
                f"локализовано:           {self.n_localized} ({100 * self.success_rate:.1f}%)",
                f"ошибка центра, медиана: {self.median_center_error_m:.3f} м "
                f"({self.median_center_error_px:.3f} px подложки)",
                f"ошибка центра, p90:     {self.p90_center_error_m:.3f} м",
                f"ошибка центра, макс:    {self.max_center_error_m:.3f} м",
                f"ошибка курса, медиана:  {self.median_heading_error_deg:.4f}°",
                f"ошибка курса, макс:     {self.max_heading_error_deg:.4f}°",
                f"ошибка масштаба, медиана: {self.median_scale_error_pct:+.4f}%",
                f"ошибка масштаба, макс|·|: {self.max_abs_scale_error_pct:.4f}%",
                f"ошибка высоты, медиана: {self.median_altitude_error_pct:+.4f}%",
            ]
        )


def _summarize(metrics: Sequence[SampleMetrics]) -> GridSummary:
    ok = [m for m in metrics if m.localized]
    nan = float("nan")
    if not ok:
        return GridSummary(
            n_samples=len(metrics),
            n_localized=0,
            success_rate=0.0,
            median_center_error_m=nan,
            p90_center_error_m=nan,
            max_center_error_m=nan,
            median_center_error_px=nan,
            median_heading_error_deg=nan,
            max_heading_error_deg=nan,
            median_scale_error_pct=nan,
            max_abs_scale_error_pct=nan,
            median_altitude_error_pct=nan,
            metrics=list(metrics),
        )

    center_m = np.array([m.center_error_m for m in ok])
    heading = np.array([m.heading_error_deg for m in ok])
    scale = np.array([m.scale_error_pct for m in ok])
    return GridSummary(
        n_samples=len(metrics),
        n_localized=len(ok),
        success_rate=len(ok) / len(metrics),
        median_center_error_m=float(np.median(center_m)),
        p90_center_error_m=float(np.percentile(center_m, 90)),
        max_center_error_m=float(np.max(center_m)),
        median_center_error_px=float(np.median([m.center_error_px for m in ok])),
        median_heading_error_deg=float(np.median(heading)),
        max_heading_error_deg=float(np.max(heading)),
        median_scale_error_pct=float(np.median(scale)),
        max_abs_scale_error_pct=float(np.max(np.abs(scale))),
        median_altitude_error_pct=float(np.median([m.altitude_error_pct for m in ok])),
        metrics=list(metrics),
    )


def iter_specs(
    yaw_deg: Sequence[float] = (0.0,),
    altitude_ratio: Sequence[float] = (1.0,),
    center_offset_px: Sequence[tuple[float, float]] = ((0.0, 0.0),),
    prior_offset_m: Sequence[float] = (0.0,),
) -> Iterator[SampleSpec]:
    """Декартова сетка возмущений — прогон «по сетке» из ``docs/TESTING.md``."""
    for yaw in yaw_deg:
        for ratio in altitude_ratio:
            for offset in center_offset_px:
                for prior_offset in prior_offset_m:
                    yield SampleSpec(
                        yaw_deg=yaw,
                        altitude_ratio=ratio,
                        center_offset_px=offset,
                        prior_offset_m=prior_offset,
                        # Азимут промаха приора привязан к yaw, чтобы сетка не
                        # вырождалась в один и тот же перекос по всем примерам.
                        prior_offset_bearing_deg=(yaw * 2.0) % 360.0,
                    )


def run_grid(
    scene: Scene,
    camera: Camera,
    specs: Sequence[SampleSpec],
    *,
    matcher: Matcher | None = None,
    prior_altitude_m: float = 600.0,
    prior_sigma_m: float = 200.0,
    reference_size: int = 1280,
    trust_yaw: bool = True,
    refine: bool = False,
) -> GridSummary:
    """Сгенерировать примеры по сетке, прогнать пайплайн и собрать метрики.

    Args:
        refine: включить субпиксельный ECC-refinement в пайплайне — то самое
            A/B «с refinement и без» для проверки решения №1 из ``docs/STATUS.md``.
    """
    metrics = []
    for spec in specs:
        sample = generate_sample(
            scene,
            camera,
            spec,
            prior_altitude_m=prior_altitude_m,
            prior_sigma_m=prior_sigma_m,
            reference_size=reference_size,
        )
        result = localize_against_reference(
            sample.query,
            sample.camera,
            sample.prior,
            sample.reference,
            sample.reference_georef,
            matcher=matcher,
            trust_yaw=trust_yaw,
            refine=refine,
        )
        metrics.append(evaluate(result, sample))
    return _summarize(metrics)
