"""Внешние чекпоинты: откуда, под какой лицензией и **действительно ли легли**.

Зачем модуль ([ROADMAP.md](../docs/ROADMAP.md), фаза 2). Дешёвая часть трека A —
это не новые архитектуры, а **другие веса в те же сети**: GIM обучен на
интернет-видео ради обобщения, MINIMA — целево на инвариантность вида. Обе
подмены ложатся в уже установленные `lightglue` и `kornia`, и потому стоят почти
ничего.

Главная опасность здесь не техническая, а методологическая. ``load_state_dict``
с ``strict=False`` молча стерпит **любое** несовпадение имён: сеть останется на
базовых весах, матчер отработает, A/B покажет «GIM не помог», и вывод будет
сделан о GIM, хотя GIM в вычислении не участвовал. Ровно тем же способом уже
дважды получался неверный вывод о LoFTR (`docs/JOURNAL.md`). Поэтому
:func:`apply_state_dict` **считает, сколько тензоров реально совпало**, и падает,
если совпало мало.

Проверено на этих чекпоинтах:

======================  ========================================================
``gim_lightglue``       251/252 в LightGlue + 24/24 в SuperPoint — ложится
                        полностью (единственный «пропуск» —
                        ``confidence_thresholds``, буфер, который модель считает
                        сама)
``minima_lightglue``    251/252 в LightGlue; SuperPoint остаётся штатным
``minima_loftr``        211/211 в ``kornia.feature.LoFTR`` — точь-в-точь
``gim_loftr``           **НЕ ложится**: 114/211, у GIM свой backbone
                        (``backbone.encode.*`` против ``backbone.*``) и 258
                        лишних тензоров. Нужен код сети из репозитория GIM;
                        подсунуть его в kornia нельзя
======================  ========================================================

Кэш весов — каталог из ``AERO_WEIGHTS_DIR`` либо ``weights/`` в корне проекта;
он в ``.gitignore``, потому что это чужие данные на сотни мегабайт.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "Checkpoint", "CHECKPOINTS", "CheckpointMismatch",
    "weights_dir", "checkpoint_path", "read_state_dict", "apply_state_dict",
]


class CheckpointMismatch(RuntimeError):
    """Чекпоинт не лёг в модель — вместо тихого отката поднимаем ошибку."""


@dataclass(frozen=True)
class Checkpoint:
    """Внешние веса: адрес, размер, лицензия и чем они интересны.

    Лицензия и источник хранятся рядом с URL намеренно: вопрос «а можно ли это в
    продукте» возникает после того, как кандидат победил, и к тому моменту
    выяснять происхождение весов уже поздно (``docs/ROADMAP.md``, риски).
    """

    name: str
    url: str
    filename: str
    size_mb: float
    licence: str
    source: str
    note: str = ""


CHECKPOINTS: dict[str, Checkpoint] = {
    "gim_lightglue": Checkpoint(
        name="gim_lightglue",
        url="https://raw.githubusercontent.com/xuelunshen/gim/main/weights/gim_lightglue_100h.ckpt",
        filename="gim_lightglue_100h.ckpt",
        size_mb=52.7,
        licence="MIT",
        source="github.com/xuelunshen/gim",
        note="SuperPoint + LightGlue, обученные на 100 ч интернет-видео. Чекпоинт "
             "несёт и свой SuperPoint — он и берётся, иначе матчер собрался бы из "
             "половин двух разных обучений.",
    ),
    "minima_lightglue": Checkpoint(
        name="minima_lightglue",
        url="https://huggingface.co/lsxi77777/MINIMA/resolve/main/minima_lightglue.pth",
        filename="minima_lightglue.pth",
        size_mb=47.5,
        licence="Apache-2.0",
        source="huggingface.co/lsxi77777/MINIMA",
        note="Только LightGlue, дообученный на синтетических кросс-модальных парах; "
             "SuperPoint остаётся штатным.",
    ),
    "minima_loftr": Checkpoint(
        name="minima_loftr",
        url="https://huggingface.co/lsxi77777/MINIMA/resolve/main/minima_loftr.ckpt",
        filename="minima_loftr.ckpt",
        size_mb=46.4,
        licence="Apache-2.0",
        source="huggingface.co/lsxi77777/MINIMA",
        note="LoFTR той же архитектуры, что в kornia: 211 тензоров совпадают точь-в-точь.",
    ),
}


def weights_dir() -> Path:
    """Каталог кэша весов: ``AERO_WEIGHTS_DIR`` либо ``weights/`` в корне проекта."""
    env = os.environ.get("AERO_WEIGHTS_DIR")
    return Path(env) if env else Path(__file__).resolve().parents[1] / "weights"


def checkpoint_path(name: str, *, allow_download: bool = True) -> Path:
    """Путь к файлу весов; при отсутствии — скачать один раз.

    Скачивание идёт во временный файл и переименовывается только целиком: иначе
    прерванная загрузка оставила бы обрезанный файл, который в следующий раз
    молча приняли бы за готовый.
    """
    if name not in CHECKPOINTS:
        raise ValueError(f"неизвестный чекпоинт {name!r}, доступны: {sorted(CHECKPOINTS)}")
    spec = CHECKPOINTS[name]
    path = weights_dir() / spec.filename
    if path.exists():
        return path
    if not allow_download:
        raise FileNotFoundError(
            f"нет весов {spec.filename} в {weights_dir()}; скачать вручную: {spec.url}"
        )
    import urllib.request

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(spec.url, tmp)
    tmp.replace(path)
    return path


def read_state_dict(path) -> dict:
    """Словарь тензоров из чекпоинта, развёрнутый из обёртки Lightning."""
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise CheckpointMismatch(f"{path}: ожидался словарь тензоров, получено {type(obj).__name__}")
    return obj


def apply_state_dict(module, state: dict, *, label: str, min_matched: float = 0.9) -> int:
    """Загрузить веса в модуль и **проверить**, что они действительно легли.

    Совпадение считается по именам И формам: тензор с тем же именем, но другой
    формой, — это не «частично подошло», а другая архитектура.

    Args:
        module: целевая сеть.
        state: словарь тензоров, уже приведённый к её именованию.
        label: имя чекпоинта для сообщения об ошибке.
        min_matched: минимальная доля покрытых тензоров модели. Дефолт 0.9, а не
            1.0, потому что часть буферов модели вычисляет сама
            (``confidence_thresholds`` в LightGlue).

    Returns:
        Сколько тензоров модели покрыто чекпоинтом.

    Raises:
        CheckpointMismatch: если покрыто меньше ``min_matched``. Молчаливый откат
            на базовые веса недопустим: матчер отработает, A/B покажет «кандидат
            не помог», и вывод будет сделан о кандидате, которого в вычислении не
            было.
    """
    target = module.state_dict()
    matched = {k: v for k, v in state.items()
               if k in target and tuple(target[k].shape) == tuple(v.shape)}
    fraction = len(matched) / max(len(target), 1)
    if fraction < min_matched:
        wrong_shape = sum(1 for k, v in state.items()
                          if k in target and tuple(target[k].shape) != tuple(v.shape))
        raise CheckpointMismatch(
            f"чекпоинт {label!r} не подходит модели: покрыто {len(matched)} из "
            f"{len(target)} тензоров ({fraction:.0%}, нужно ≥{min_matched:.0%}), "
            f"с несовпавшей формой {wrong_shape}, лишних в чекпоинте "
            f"{len(state) - len(matched)}. Это другая архитектура, а не другие веса — "
            f"нужен код сети из репозитория автора."
        )
    module.load_state_dict(matched, strict=False)
    return len(matched)
