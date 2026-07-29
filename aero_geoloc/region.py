"""Карта района: какая геометрия нужна кадру и где она лежит на диске.

Зачем модуль ([TOOL_PLAN.md](../docs/TOOL_PLAN.md), этап T1). Знание «какой зум
взять под этот кадр, какого размера клетка, какое перекрытие и где кэш» нужно и
харнессу оценки, и инструменту владельца. Пока оно жило в приватных функциях
одного скрипта, второй потребитель мог только скопировать — а копия неизбежно
разъезжается. В этом проекте так уже было: запас окна точного уровня и
перекрытие сетки считались в двух местах по разным формулам, и это стоило
кросс-сезонных кадров (веха про передачу Этаж 1 → Этаж 2 в ``docs/JOURNAL.md``).

Три вещи, которые модуль удерживает вместе:

**Клетка ≈ отпечатку кадра.** Иначе эмбеддинги несопоставимы по масштабу.
Отпечатки в наборе гуляют от ~90 м (52 м AGL) до ~520 м, поэтому зум выбирается
**на кадр**, а не фиксируется: фиксированный либо раздул бы число клеток на
больших отпечатках, либо потребовал бы качать тайлы гигабайтами.

**Перекрытие связано с запасом окна** (:func:`localize.required_cell_overlap`),
но связь не сводится к формуле — она тянет за собой ещё и ``top_k``. Здесь
берётся измеренная политика, а не расчёт; обоснование — в docstring той функции.

**Ключ кэша — вся геометрия, а не имя кадра.** Снимки одной серии делят приор и
нарезку, и по имени карта пересобиралась бы на каждый кадр. При этом в ключ
входит **всё**, что меняет содержимое: перекрытие однажды забыли, и прогон с
другой плотностью сетки молча взял старую карту, «доказав», что изменение не
помогло. Тихое переиспользование хуже лишней пересборки.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .camera import Camera
from .geo import Georef, ground_mpp, zoom_for_mpp
from .localize import MAX_FINE_WINDOW_PX, required_cell_overlap
from .retrieval import TerrainIndex
from .types import Prior

__all__ = [
    "RegionPlan", "plan_region", "build_or_load",
    "estimated_build_seconds", "human_time",
    "DEFAULT_CELL_PX", "DEFAULT_PCA_DIM",
]

#: Целевой размер клетки индекса в пикселях. Зум подбирается так, чтобы отпечаток
#: кадра занял примерно столько — компромисс между детальностью эмбеддинга и
#: числом тайлов, которые придётся скачать.
DEFAULT_CELL_PX = 350

#: Размерность после PCA-редукции дескрипторов (`docs/JOURNAL.md`, веха PCA+FAISS).
DEFAULT_PCA_DIM = 1024


@dataclass(frozen=True)
class RegionPlan:
    """Геометрия карты района под конкретный кадр плюс путь её кэша.

    Attributes:
        region: привязка региона индексации (центр — приор, размер — 2·радиус).
        cell_px: сторона клетки в пикселях уровня индекса.
        overlap: перекрытие соседних клеток.
        rotations_deg: углы ротационной аугментации. Один угол — курс известен;
            восемь — неизвестен, и это в 8 раз дороже по сборке.
        mpp_index: разрешение уровня индекса, м/пиксель.
        footprint_m: отпечаток кадра на земле (наибольшая сторона), метры.
        radius_m: радиус района.
        pca_dim: размерность после редукции.
        path: файл кэша.
    """

    region: Georef
    cell_px: int
    overlap: float
    rotations_deg: tuple[float, ...]
    mpp_index: float
    footprint_m: float
    radius_m: float
    pca_dim: int
    path: Path

    @property
    def cells(self) -> int:
        """Оценка числа клеток — по ней видно цену сборки ДО её начала."""
        step_px = max(1, round(self.cell_px * (1.0 - self.overlap)))
        per_side = max(1, int(self.region.width // step_px))
        return per_side * per_side * len(self.rotations_deg)

    @property
    def cached(self) -> bool:
        return self.path.exists()

    def describe(self) -> str:
        """Человекочитаемая строка для прогресса: цена и её причина."""
        state = "есть в кэше" if self.cached else "нужно строить"
        rot = ("курс известен" if len(self.rotations_deg) == 1
               else f"курс неизвестен, ×{len(self.rotations_deg)} поворотов")
        return (f"район ±{self.radius_m / 1000:.1f} км, zoom {self.region.zoom}, "
                f"клетка {self.cell_px} px ≈ {self.footprint_m:.0f} м, "
                f"перекрытие {self.overlap:.2f}, ~{self.cells} клеток ({rot}); {state}")


def plan_region(
    camera: Camera,
    prior: Prior,
    *,
    radius_m: float,
    max_zoom: int,
    fine_zoom: int,
    trust_yaw: bool,
    rotation_step_deg: int = 45,
    cell_px_target: int = DEFAULT_CELL_PX,
    overlap: float | None = None,
    pca_dim: int = DEFAULT_PCA_DIM,
    max_fine_window_px: int = MAX_FINE_WINDOW_PX,
    cache_dir: str | Path = "maps",
    prefix: str = "map",
) -> RegionPlan:
    """Подобрать геометрию карты района под кадр.

    Args:
        camera: камера кадра (уже в том разрешении, в котором пойдёт в пайплайн).
        prior: приор — его центр становится центром района.
        radius_m: радиус района. Должен покрывать неопределённость приора.
        max_zoom: предел зума провайдера подложки.
        fine_zoom: зум точного уровня — от него зависит требуемое перекрытие.
        trust_yaw: известен ли курс. Если нет, индекс аугментируется поворотами,
            и сборка дорожает пропорционально их числу.
        rotation_step_deg: шаг аугментации при неизвестном курсе.
        overlap: перекрытие явно; ``None`` — измеренная политика.
    """
    if radius_m <= 0:
        raise ValueError("radius_m должен быть > 0")
    footprint_m = max(camera.footprint_m(prior.altitude_m))
    z_index = zoom_for_mpp(footprint_m / cell_px_target, prior.lat,
                           mode="coarser", max_zoom=max_zoom)
    mpp_index = ground_mpp(prior.lat, z_index)
    cell_px = max(32, round(footprint_m / mpp_index))
    region_px = int(2 * radius_m / mpp_index)
    region = Georef(prior.lon, prior.lat, z_index, region_px, region_px)

    if overlap is None:
        overlap = required_cell_overlap(
            footprint_m, ground_mpp(prior.lat, fine_zoom),
            max_window_px=max_fine_window_px,
        )
    rotations = ((0.0,) if trust_yaw
                 else tuple(float(d) for d in range(0, 360, rotation_step_deg)))

    tag = (f"{region.center_lat:.4f}_{region.center_lon:.4f}_z{region.zoom}"
           f"_r{int(radius_m)}_c{cell_px}_o{int(overlap * 100)}"
           f"_rot{len(rotations)}_pca{pca_dim}")
    return RegionPlan(
        region=region, cell_px=cell_px, overlap=float(overlap), rotations_deg=rotations,
        mpp_index=mpp_index, footprint_m=footprint_m, radius_m=float(radius_m),
        pca_dim=pca_dim, path=Path(cache_dir) / f"{prefix}_{tag}.npz",
    )


def build_or_load(
    plan: RegionPlan,
    basemap,
    encoder,
    *,
    rebuild: bool = False,
    ef_search: int = 128,
) -> tuple[TerrainIndex, float]:
    """Карта района: с диска, если есть, иначе построить и сохранить.

    Returns:
        ``(индекс, секунды на сборку)``. Ноль секунд означает попадание в кэш.
    """
    if plan.path.exists() and not rebuild:
        index = TerrainIndex.load(plan.path, encoder)
        index.use_faiss(kind="hnsw", ef_search=ef_search)
        return index, 0.0

    started = time.perf_counter()
    index = TerrainIndex(encoder).build(
        basemap, plan.region, cell_size_px=plan.cell_px,
        overlap=plan.overlap, rotations_deg=plan.rotations_deg,
    )
    index.compress(plan.pca_dim, whiten=False)
    elapsed = time.perf_counter() - started
    plan.path.parent.mkdir(parents=True, exist_ok=True)
    index.save(plan.path)
    index.use_faiss(kind="hnsw", ef_search=ef_search)
    return index, elapsed


def estimated_build_seconds(plan: RegionPlan, *, seconds_per_cell: float = 0.055) -> float:
    """Грубая оценка времени сборки, чтобы предупредить ДО её начала.

    Коэффициент замерен на этом окружении (GPU, батч 8, FP16) и нужен не для
    точности, а чтобы владелец знал: это минуты, а не секунды. Ошибка в разы
    здесь безобидна, а отсутствие оценки — нет.
    """
    return plan.cells * seconds_per_cell


def human_time(seconds: float) -> str:
    """Секунды словами: «12 с», «4 мин», «1 ч 20 мин»."""
    if seconds < 90:
        return f"{seconds:.0f} с"
    minutes = seconds / 60.0
    if minutes < 90:
        return f"{minutes:.0f} мин"
    return f"{int(minutes // 60)} ч {int(minutes % 60)} мин"
