"""Обзор папки со снимками: что в метаданных и как это запускать.

Инструмент владельца. Проходит по снимкам, читает EXIF/XMP **без декодирования
пикселей** и пишет текстовый файл: координаты, высота, курс, вычисленный GSD и
**готовая команда** на каждый снимок.

    .venv/Scripts/python.exe scripts/exif_index.py --images test_images
    .venv/Scripts/python.exe scripts/exif_index.py --images test_images/DRZ --out DRZ.txt

Зачем именно так, а не «список координат»
-----------------------------------------
1. **Координаты в EXIF лежат тройками «градусы, минуты, секунды»**, и взять из
   них последнее число — ошибка, которая выглядит как правдоподобная координата.
   Ровно так был потерян первый запуск: `58.37 / 9.306` вместо
   `51°12'58.37" / 6°10'9.31"` = `51.21621 / 6.16925`, и поиск ушёл в море.
   Здесь пересчёт сделан один раз и правильно.

2. **Приор в команде намеренно сдвинут** от истины на половину σ. Если задать
   приором точные координаты снимка, система будет искать кадр там, куда ей же
   и указали, и успех не докажет ничего. Истина в файле есть — отдельной
   строкой, для сверки результата, а не для подстановки на вход.

3. **σ приора не берётся «с запасом»**: запас здесь вредит, чем шире круг, тем
   больше похожих мест в него попадает. Источник σ — по убыванию доверия:
   значение, **проверенное прогонами** (из манифеста), затем регламент «высота →
   σ» (:func:`~aero_geoloc.request.recommended_sigma_m`). Порядок именно такой,
   потому что у кадров без EXIF высота в манифесте — заглушка, и регламент по
   ней советует вчетверо более широкий приор, чем измеренный.

4. **Истина берётся и из манифеста**, а не только из EXIF: у половины набора
   метаданные срезаны, а разметка владельца лежит в `datasets/test_images.yaml`.

Снимки без GPS, без высоты или снятые не в надир не пропускаются, а попадают в
файл с причиной — «нечего запускать» тоже полезное знание о наборе.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aero_geoloc.dataset import load_dataset  # noqa: E402
from aero_geoloc.drone import ShotMeta, read_metadata  # noqa: E402
from aero_geoloc.request import recommended_sigma_m  # noqa: E402

SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

#: Сдвиг приора от истины — доля от σ, а не фиксированные метры.
#:
#: Фиксированные сотни метров бессмысленны на обоих концах: при σ = 0.5 км это
#: уже больше половины диска, при σ = 4 км — почти точное указание, и успех
#: ничего не докажет. Половина σ держит истину глубоко внутри ±3σ и при этом
#: остаётся честной задачей на любом масштабе.
PRIOR_OFFSET_FRACTION = 0.5


def _offset_prior(lat: float, lon: float, sigma_m: float, index: int) -> tuple[float, float]:
    """Сдвинуть приор от истины на половину σ в одну из четырёх сторон.

    Сторона зависит от номера снимка — так соседние кадры не смещаются одинаково,
    и набор не приобретает систематического перекоса, который можно было бы
    случайно «выучить».
    """
    import math

    offset_m = sigma_m * PRIOR_OFFSET_FRACTION
    dlat = offset_m / 111_320.0
    dlon = offset_m / (111_320.0 * max(math.cos(math.radians(lat)), 0.01))
    sign_lat, sign_lon = ((1, 0), (0, 1), (-1, 0), (0, -1))[index % 4]
    return round(lat + sign_lat * dlat, 5), round(lon + sign_lon * dlon, 5)


class Known:
    """Что известно о снимке после слияния EXIF и манифеста.

    Манифест нужен потому, что у половины набора EXIF срезан начисто (Саратов,
    Волгоград, кадры Уфы из видео), а истина по ним размечена владельцем и лежит
    в ``datasets/test_images.yaml``. Без слияния файл был бы полезен только для
    снимков DJI, то есть ровно наполовину.
    """

    def __init__(self, meta: ShotMeta, case=None) -> None:
        self.meta, self.case = meta, case
        # ExcludedCase несёт только причину исключения — полей истины у него нет,
        # поэтому обращаемся через getattr, а не по типу.
        self.excluded_reason = getattr(case, "reason", None)
        self.lat, self.lon, self.truth_source = meta.lat, meta.lon, "EXIF"
        if not meta.has_position and getattr(case, "has_truth", False):
            self.lat, self.lon = case.truth_lat, case.truth_lon
            self.truth_source = f"разметка владельца ({case.truth_source})"
        self.gsd = meta.gsd_m() or getattr(case, "gsd_m", None)
        prior = getattr(case, "prior", None)
        self.altitude_m = meta.altitude_m or (prior.altitude_m if prior else None)

        # σ: измеренное важнее регламента. У кадров без EXIF «высота 500 м» в
        # манифесте — заглушка при известном GSD, и регламент по ней выдаёт 4 км,
        # тогда как прогонами подтверждено 1.5 км. Советовать вчетверо более
        # широкий приор значило бы советовать ровно ту ошибку, от которой этот
        # файл предостерегает.
        if meta.altitude_m is None and prior is not None and prior.sigma_m:
            self.sigma_m, self.sigma_source = prior.sigma_m, "проверено прогонами"
        else:
            self.sigma_m = recommended_sigma_m(self.altitude_m or 500.0)
            self.sigma_source = "регламент «высота → σ»"

    @property
    def runnable(self) -> bool:
        return (self.lat is not None and self.gsd is not None
                and self.meta.is_nadir and self.excluded_reason is None)


def command_for(known: Known, index: int, *, root: Path) -> str | None:
    """Готовая строка запуска либо ``None``, если запускать нечем."""
    if known.lat is None or known.gsd is None:
        return None
    lat, lon = _offset_prior(known.lat, known.lon, known.sigma_m, index)
    sigma_km = known.sigma_m / 1000.0
    path = known.meta.path
    path = path.relative_to(root.parent) if root.parent in path.parents else path
    cmd = (f"python scripts/locate.py --image {path.as_posix()} "
           f"--lat {lat} --lon {lon} --sigma-km {sigma_km:g} --gsd {known.gsd:.3f}")
    if known.meta.yaw_deg is not None:
        cmd += f" --yaw {known.meta.yaw_deg:.0f}"
    return cmd


def block_for(known: Known, index: int, *, root: Path, duplicate_of: str | None = None) -> str:
    """Абзац про один снимок: факты, затем команда либо причина её отсутствия."""
    meta = known.meta
    lines = [f"### {meta.path.name}   [{meta.path.parent.name}]"]
    lines.append(f"    камера      {meta.model or 'метаданных нет'}, {meta.width}×{meta.height}"
                 + (f", снят {meta.datetime}" if meta.datetime else ""))
    if duplicate_of:
        lines.append(f"    ДУБЛИКАТ    того же кадра, что {duplicate_of} — запускать один из двух")
    if known.lat is not None:
        lines.append(f"    ИСТИНА      {known.lat:.6f}, {known.lon:.6f}   ({known.truth_source})"
                     f"\n                ← ею проверяют ОТВЕТ, а не задают вопрос")
    if known.altitude_m is not None and known.gsd is not None:
        lines.append(f"    высота      {known.altitude_m:.0f} м над землёй   "
                     f"⇒ GSD {known.gsd:.3f} м/пкс, кадр покрывает "
                     f"{known.gsd * meta.width:.0f}×{known.gsd * meta.height:.0f} м")
    elif known.gsd is not None:
        lines.append(f"    GSD         {known.gsd:.3f} м/пкс (из манифеста), кадр покрывает "
                     f"{known.gsd * meta.width:.0f}×{known.gsd * meta.height:.0f} м")
    if meta.yaw_deg is not None:
        lines.append(f"    курс        {meta.yaw_deg:.0f}°"
                     + (f", наклон от надира {meta.pitch_from_nadir_deg:.0f}°"
                        if meta.pitch_from_nadir_deg is not None else ""))
    if known.truth_source == "EXIF":
        for problem in meta.problems:
            lines.append(f"    ! {problem}")

    cmd = command_for(known, index, root=root)
    if known.excluded_reason:
        lines.append(f"    НЕ ЗАПУСКАТЬ: {known.excluded_reason}")
    elif not meta.is_nadir:
        lines.append("    НЕ ЗАПУСКАТЬ: кадр вне надирной модели, результата не будет")
    elif cmd is None:
        lines.append("    запускать нечем: не хватает координат или масштаба")
    else:
        lines.append(f"    приор       ±{known.sigma_m / 1000:g} км ({known.sigma_source}), "
                     f"центр сдвинут от истины на {known.sigma_m * PRIOR_OFFSET_FRACTION:.0f} м")
        lines.append(f"    ЗАПУСК\n        {cmd}")
    return "\n".join(lines)


HEADER = """\
ОБЗОР СНИМКОВ: координаты, масштаб и готовые команды
====================================================
Собрано автоматически: scripts/exif_index.py. Пересобрать после добавления
снимков — той же командой.

КАК ЧИТАТЬ
  ИСТИНА   точные координаты — из EXIF либо из разметки владельца. Ими проверяют
           ОТВЕТ, а не задают вопрос.
  ЗАПУСК   готовая команда. Приор в ней намеренно сдвинут от истины на половину
           σ: если искать кадр по его же координатам, успех ничего не доказывает.

ПРО КООРДИНАТЫ В EXIF
  Они лежат тройками «градусы, минуты, секунды»: (51, 12, 58.3725) — это
  51°12'58.37" = 51.21621, а НЕ 58.37. Здесь пересчёт уже сделан.

ПРО ПОГРЕШНОСТЬ ПРИОРА
  --sigma-km взята по регламенту «высота → σ» (docs/JOURNAL.md), а не с запасом.
  Шире — прямо хуже: чем больше круг, тем больше похожих мест в нём, и верное
  тонет. Замерено на DRZ_00755 (90 м): ±2 км — отказ, ±0.5 км — 1.7 м.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", default="test_images", help="папка со снимками")
    parser.add_argument("--out", default="", help="куда писать; по умолчанию <папка>_exif.txt")
    parser.add_argument("--manifest", default="datasets/test_images.yaml",
                        help="манифест с ручной разметкой истины; '' — не использовать")
    args = parser.parse_args()

    root = Path(args.images)
    if not root.is_dir():
        print(f"{root} — не папка")
        return 1
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in SUFFIXES)
    if not files:
        print(f"в {root} нет изображений")
        return 1

    # Истина по снимкам без EXIF живёт в манифесте — без него файл был бы полезен
    # только для кадров DJI, то есть ровно наполовину.
    by_path: dict[Path, object] = {}
    if args.manifest and Path(args.manifest).is_file():
        ds = load_dataset(args.manifest)
        for case in [*ds.cases, *ds.excluded]:
            by_path[Path(case.path).resolve()] = case

    blocks, runnable, seen = [], 0, {}
    for i, path in enumerate(files):
        try:
            meta = read_metadata(path)
        except Exception as exc:  # noqa: BLE001 — один битый файл не рушит обзор
            blocks.append(f"### {path.name}\n    ! не прочитать: {type(exc).__name__}: {exc}")
            continue
        known = Known(meta, by_path.get(path.resolve()))
        # Один и тот же кадр лежит в наборе дважды (00049 в корне и в DRZ/).
        # Молча выкидывать нельзя — обзор должен отражать папку как есть.
        key = (known.lat, known.lon, meta.datetime, meta.width, meta.height)
        duplicate_of = seen.get(key) if known.lat is not None else None
        seen.setdefault(key, path.name)
        if known.runnable and duplicate_of is None:
            runnable += 1
        blocks.append(block_for(known, i, root=root, duplicate_of=duplicate_of))

    out = Path(args.out) if args.out else Path(f"{root.name}_exif.txt")
    text = (HEADER
            + f"\nСнимков: {len(files)}, из них с готовой командой: {runnable}\n"
            + "\n" + "=" * 68 + "\n\n"
            + "\n\n".join(blocks) + "\n")
    out.write_text(text, encoding="utf-8")
    print(f"{len(files)} снимков, с готовой командой {runnable} → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
