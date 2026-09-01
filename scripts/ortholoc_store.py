"""Компактное хранение сэмплов OrthoLoC: запись, чтение и разбор объёма.

Исходный сэмпл OrthoLoC весит около 16 МБ, а весь датасет — 269 ГБ, и почти
весь этот объём уходит на данные, которые либо избыточны, либо хранятся с
точностью, далеко превосходящей измеримую нашим протоколом:

=====================  ==========  ============================================
массив                 в файле     что с ним не так
=====================  ==========  ============================================
``point_map``            7.2 МБ    float32 мировые XYZ, но меряем мы пиксели
                                   DOP, а ошибка лучшего ядра — 2.6 px
``dsm``                  4.2 МБ    три канала XYZ, из которых X и Y —
                                   регулярная сетка: две полные карты хранят
                                   то, что описывается двумя векторами
``image_dop``            2.5 МБ    сырые пиксели вместо JPEG
``image_query``          2.0 МБ    то же
``vertices``/``faces``   0.3 МБ    меш сцены; из него читалось одно число
=====================  ==========  ============================================

Компактный формат хранит то же содержание иначе:

- изображения — сжатыми: на тестовых сплитах **без потерь** (WebP lossless,
  вдвое меньше сырых пикселей), на обучающих — JPEG q95;
- вместо мировых координат — GT-карта «пиксель кадра → пиксель DOP» с шагом
  1/256 px на тестовых сплитах и 1/16 px на обучающих: это ровно та величина,
  против которой считается EPE;
- высоты (третий канал ``point_map`` и ``dsm``) — целыми с шагом 5 мм;
- ``dsm`` XY — двумя векторами осей сетки;
- вместо меша — медиана Z вершин.

Отдельно от квантования работает **дельта-кодирование**: и GT, и высоты —
почти линейные поля, поэтому хранятся построчные разности, а не значения.
Замерено на реальном сэмпле: GT в этом виде занимает 0.45 МБ против 1.98 МБ
прямым хранением, при той же точности. Дыры в разметке перед разностями
заполняются протяжкой соседа — иначе на каждой границе маски возникал бы
скачок на весь диапазон.

Точность: GT искажается не более чем на половину шага (0.002 px в профиле
``eval``, 0.031 px в ``train``), высоты — на 2.5 мм. Против EPE 2.6 px у
лучшего ядра и самого строгого порога инлайера 1 px это пренебрежимо; границы
проверяются в ``tests/test_ortholoc_store.py``.

Почему профиля два: и пиксели, и GT — это то, из чего считается метрика, а не
только то, что лежит на диске. Матчеры недетерминированы (два прогона на одних
и тех же данных совпадают точно лишь на 6 сэмплах из 40), поэтому форматы
сравнивались **парно по сэмплам** — медиана ``|Δ inl1|`` ванильной RoMa
относительно исходных данных:

======================  ========  =============================================
вариант                 медиана   что это значит
======================  ========  =============================================
повторный прогон         0.0057   собственный шум матчера
профиль ``eval``         0.0040   неотличимо от шума
GT 1/16 px без потерь    0.0059   на границе шума
JPEG q95                 0.0103   вдвое выше шума
======================  ========  =============================================

Поэтому тестовые сплиты хранятся без потерь и с мелким шагом GT, а сжатие с
потерями остаётся там, где изображения идут в обучение вместе с аугментациями.

Чтение единообразно для обоих форматов: :func:`open_sample` отдаёт объект с
интерфейсом ``np.load`` — ``d["image_query"]``, ``d["point_map"]``,
``d["dsm"]``, — поэтому вызывающий код не знает, что лежит на диске.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import cv2
import numpy as np

#: Метка формата: по ней :func:`open_sample` отличает компактный сэмпл от
#: исходного.
FORMAT = "ortholoc-slim/1"

#: Шаг хранения GT-координат по умолчанию, px. Половина шага (0.031 px) на
#: порядок меньше самого строгого порога инлайера (1 px) и на два — типичной
#: ошибки ядра. На тестовых сплитах шаг мельче: см. :data:`PROFILES`.
GT_STEP = 1.0 / 16.0
#: Смещение GT перед упаковкой в uint16, px: координаты DOP бывают
#: отрицательными (кадр смотрит за край ортофото). Диапазон получается
#: [−1024, +3095] px при стороне DOP 1024.
GT_BIAS = 1024.0
#: Шаг хранения высот, м. 5 мм высоты дают около 0.02 px в проекции кадра —
#: заведомо ниже всего, что мы измеряем.
Z_STEP = 0.005
#: Качество JPEG для обеих сторон.
JPEG_QUALITY = 95

#: Профили хранения. Тестовые сплиты — то, на чём снимаются числа бенчмарка,
#: поэтому изображения там кодируются **без потерь** (WebP lossless сжимает
#: вдвое, оставляя пиксели байт-в-байт), а GT хранится вчетверо точнее.
#: Замерено на 40 сэмплах парным сравнением с исходными данными: JPEG q95
#: меняет ``inl1`` ванильной RoMa на 0.0103 по медиане сэмпла, тогда как
#: повторный прогон на тех же данных — на 0.0057. Профиль ``eval`` укладывается
#: в 0.0040, то есть в собственный шум матчера.
PROFILES = {
    "eval":  dict(codec="webp", quality=101, gt_step=1.0 / 256.0),
    "train": dict(codec="jpeg", quality=JPEG_QUALITY, gt_step=GT_STEP),
}


# ————————————————————— кодирование плоских полей —————————————————————

def _ffill_rows(a: np.ndarray, m: np.ndarray) -> np.ndarray:  # noqa: D401
    """Заполнить дыры значением ближайшего валидного соседа по строке.

    Нужно перед взятием разностей: под маской в массиве лежит произвольное
    значение, и на каждой границе дыры разность подскакивала бы на весь
    диапазон, разрушая сжатие (и переполняя int16).
    """
    w = a.shape[1]
    cols = np.arange(w)[None, :]
    idx = np.where(m, cols, 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    fwd = np.take_along_axis(a, idx, axis=1)
    idx_b = np.where(m[:, ::-1], cols, 0)
    np.maximum.accumulate(idx_b, axis=1, out=idx_b)
    bwd = np.take_along_axis(a[:, ::-1], idx_b, axis=1)[:, ::-1]
    filled = np.where(m.cumsum(axis=1) > 0, fwd, bwd)
    return np.where(m.any(axis=1)[:, None], filled, 0)


def encode_plane(values: np.ndarray, mask: np.ndarray, name: str) -> dict:
    """Поле ``uint16`` + маска → массивы для записи (построчные разности).

    Если разности не влезают в ``int16`` (поле не гладкое), кодирование
    честно откатывается к прямому хранению — с пометкой в файле, а не молча.
    """
    filled = _ffill_rows(values, mask.astype(bool))
    diff = np.diff(filled.astype(np.int64), axis=1)
    if diff.size and (diff.min() < -32768 or diff.max() > 32767):
        return {f"{name}_mode": np.asarray("raw"), f"{name}_val": filled}
    return {f"{name}_mode": np.asarray("delta"),
            f"{name}_head": filled[:, :1],
            f"{name}_diff": diff.astype(np.int16)}


def decode_plane(z, name: str) -> np.ndarray:
    """Обратная операция к :func:`encode_plane`."""
    if str(z[f"{name}_mode"]) == "raw":
        return z[f"{name}_val"]
    head = z[f"{name}_head"]
    diff = z[f"{name}_diff"].astype(np.int64)
    out = np.concatenate([head.astype(np.int64),
                          head.astype(np.int64) + np.cumsum(diff, axis=1)], axis=1)
    return out.astype(head.dtype)


def _keys(d) -> tuple:
    return tuple(getattr(d, "files", ()) or ())


# ——————————————————————————— GT и высоты ———————————————————————————

def pack_gt(gt_x: np.ndarray, gt_y: np.ndarray, step: float = GT_STEP):
    """GT-координаты → ``uint16`` фиксированной точки и маска валидности.

    Вне зоны видимости в исходных данных стоит ``NaN``; в целое он не
    переносится, поэтому валидность выносится в отдельную маску. Шаг задаётся
    профилем и пишется в файл: на тестовых сплитах он вчетверо мельче.
    """
    ok = np.isfinite(gt_x) & np.isfinite(gt_y)
    # мелкий шаг не влезает в 16 бит: при 1/256 px диапазон uint16 — всего
    # 256 px, вчетверо меньше стороны DOP. Разрядность выбирается по шагу,
    # а не назначается: дельты всё равно остаются короткими
    span = (2.0 * GT_BIAS + 4096.0) / step
    dtype = np.uint16 if span <= 65535 else np.uint32
    packed = np.zeros(gt_x.shape + (2,), dtype)
    for i, g in enumerate((gt_x, gt_y)):
        v = np.where(ok, (np.nan_to_num(g) + GT_BIAS) / step, 0.0)
        if v.max() > np.iinfo(dtype).max:
            raise ValueError(f"GT не влезает в {dtype.__name__} при шаге {step}")
        packed[..., i] = np.clip(np.rint(v), 0, np.iinfo(dtype).max).astype(dtype)
    return packed, ok.astype(np.uint8)


def unpack_gt(packed: np.ndarray, mask: np.ndarray, step: float = GT_STEP):
    """Обратная операция к :func:`pack_gt`; вне маски — ``NaN``."""
    ok = mask.astype(bool)
    out = []
    for i in range(2):
        g = packed[..., i].astype(np.float64) * step - GT_BIAS
        out.append(np.where(ok, g, np.nan))
    return out[0], out[1]


def pack_heights(z: np.ndarray):
    """Карта высот → ``uint16`` с шагом :data:`Z_STEP`, маска и начало отсчёта.

    Отсчёт свой у каждого сэмпла: мировые Z здесь локальные (десятки метров),
    и общего начала у сцен нет.
    """
    ok = np.isfinite(z)
    z0 = float(np.nanmin(z)) if ok.any() else 0.0
    q = np.where(ok, np.rint((np.nan_to_num(z) - z0) / Z_STEP), 0.0)
    if q.max() > 65535:                    # перепад высот больше 327 м
        raise ValueError("диапазон высот не влезает в шаг хранения")
    return q.astype(np.uint16), ok.astype(np.uint8), np.float32(z0)


def unpack_heights(q: np.ndarray, mask: np.ndarray, z0) -> np.ndarray:
    ok = mask.astype(bool)
    z = q.astype(np.float32) * Z_STEP + np.float32(z0)
    return np.where(ok, z, np.nan).astype(np.float32)


def dop_scale(d) -> np.ndarray:
    """Масштаб DOP (м/пкс по x и y), с восстановлением там, где его нет.

    У сцены L06 (440 сэмплов из 16.4 тыс.) массив ``scale`` в файле
    отсутствует. Он выводится из ``dsm``: каналы X/Y — мировые координаты
    сетки DOP, и медианный шаг по столбцу и строке есть масштаб. Проверено на
    сэмплах, где ``scale`` есть: расхождение меньше 1e-3.
    """
    if "scale" in _keys(d):
        return np.asarray(d["scale"], dtype=np.float64)
    g = np.asarray(d["dsm"], dtype=np.float64)
    with np.errstate(invalid="ignore"):
        return np.array([float(np.nanmedian(np.diff(g[..., 0], axis=1))),
                         float(np.nanmedian(np.diff(g[..., 1], axis=0)))])


def gt_from_pointmap(point_map: np.ndarray, scale, dop_w: int, dop_h: int):
    """Мировые XY кадра → пиксели DOP. Формула та же, что в бенчмарке: DOP
    ортографичен, центр мировых координат — центр растра."""
    sx, sy = float(np.asarray(scale)[0]), float(np.asarray(scale)[1])
    return (point_map[..., 0].astype(np.float64) / sx + (dop_w - 1) / 2.0,
            point_map[..., 1].astype(np.float64) / sy + (dop_h - 1) / 2.0)


def pointmap_from_gt(gt_x: np.ndarray, gt_y: np.ndarray, pm_z: np.ndarray,
                     scale, dop_w: int, dop_h: int) -> np.ndarray:
    """Обратный ход: пиксели DOP → мировые XYZ кадра (как в ``point_map``)."""
    sx, sy = float(np.asarray(scale)[0]), float(np.asarray(scale)[1])
    pm = np.empty(gt_x.shape + (3,), np.float32)
    pm[..., 0] = (gt_x - (dop_w - 1) / 2.0) * sx
    pm[..., 1] = (gt_y - (dop_h - 1) / 2.0) * sy
    pm[..., 2] = pm_z
    return pm


# ——————————————————————————————— запись ———————————————————————————————

def encode_image(rgb: np.ndarray, codec: str, quality: int) -> np.ndarray:
    """RGB → сжатый поток байт. ``webp`` с качеством 101 — режим без потерь."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if codec == "webp":
        ok, buf = cv2.imencode(".webp", bgr, [cv2.IMWRITE_WEBP_QUALITY, quality])
    elif codec == "jpeg":
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        raise ValueError(f"неизвестный кодек изображения: {codec}")
    if not ok:
        raise RuntimeError(f"не удалось закодировать {codec}")
    return buf.reshape(-1)


def _first_valid(a: np.ndarray, axis: int) -> np.ndarray:
    """Первое конечное значение вдоль оси — заготовка оси сетки.

    Брать просто нулевую строку нельзя: у части сэмплов нижние строки сетки
    целиком ``NaN`` (область вне съёмки), и ось получилась бы пустой.
    """
    ok = np.isfinite(a)
    idx = np.argmax(ok, axis=axis)
    return np.take_along_axis(a, np.expand_dims(idx, axis), axis=axis).squeeze(axis)


def grid_axes(dsm: np.ndarray):
    """Оси сетки DOP (X по столбцам, Y по строкам) или ``None``, если сетка не осевая.

    Проверяется у каждого сэмпла, а не принимается на веру: повёрнутая сетка
    при хранении двумя векторами исказилась бы молча. Сравнение идёт **по
    валидным элементам** — дыры в XY встречаются (14 % площади у одного
    сэмпла), и они допустимы ровно при одном условии: там, где нет XY, нет и
    высоты. Тогда восстановление XY полной сеткой ни на что не влияет — такие
    точки всё равно отбраковываются по ``NaN`` в Z.
    """
    x, y, z = dsm[..., 0], dsm[..., 1], dsm[..., 2]
    ax, ay = _first_valid(x, axis=0), _first_valid(y, axis=1)
    okx, oky = np.isfinite(x), np.isfinite(y)
    same_x = np.allclose(x[okx], np.broadcast_to(ax[None, :], x.shape)[okx],
                         rtol=0, atol=1e-4)
    same_y = np.allclose(y[oky], np.broadcast_to(ay[:, None], y.shape)[oky],
                         rtol=0, atol=1e-4)
    holes_covered = not ((~okx | ~oky) & np.isfinite(z)).any()
    return (ax, ay) if (same_x and same_y and holes_covered) else None


def build(d, *, profile: str = "train", codec=None, quality=None,
          gt_step=None) -> dict:
    """Содержимое исходного сэмпла → словарь массивов компактного формата.

    Параметры берутся из профиля (:data:`PROFILES`), любой можно переопределить
    поштучно. Выбранные значения пишутся в файл, поэтому чтение не зависит от
    того, какими константами пользовался писавший.
    """
    prof = PROFILES[profile]
    codec = codec or prof["codec"]
    quality = prof["quality"] if quality is None else quality
    gt_step = prof["gt_step"] if gt_step is None else gt_step

    dop = d["image_dop"]
    hd, wd = dop.shape[:2]
    pm = np.asarray(d["point_map"])
    scale = dop_scale(d)
    gt_x, gt_y = gt_from_pointmap(pm, scale, wd, hd)
    packed, gt_mask = pack_gt(gt_x, gt_y, gt_step)

    dsm = np.asarray(d["dsm"], dtype=np.float32)
    axes = grid_axes(dsm)
    if axes is None:
        raise ValueError("сетка dsm не осевая — компактное хранение неприменимо")

    pm_q, pm_mask, pm_z0 = pack_heights(pm[..., 2])
    dz_q, dz_mask, dz_z0 = pack_heights(dsm[..., 2])

    out = {
        "fmt": np.asarray(FORMAT),
        "sample_id": np.asarray(d["sample_id"]) if "sample_id" in _keys(d) else np.asarray(""),
        "image_codec": np.asarray(codec),
        "image_query_enc": encode_image(d["image_query"], codec, quality),
        "image_dop_enc": encode_image(dop, codec, quality),
        "gt_step": np.float64(gt_step),
        "gt_mask": gt_mask,
        "pm_z_mask": pm_mask, "pm_z0": pm_z0,
        "dsm_z_mask": dz_mask, "dsm_z0": dz_z0,
        "dsm_x": axes[0].astype(np.float32),
        "dsm_y": axes[1].astype(np.float32),
        "scale": scale.astype(np.float32),
        "extrinsics": np.asarray(d["extrinsics"], dtype=np.float32),
        "intrinsics": np.asarray(d["intrinsics"], dtype=np.float32),
    }
    out.update(encode_plane(packed[..., 0], gt_mask, "gt_x"))
    out.update(encode_plane(packed[..., 1], gt_mask, "gt_y"))
    out.update(encode_plane(pm_q, pm_mask, "pm_z"))
    out.update(encode_plane(dz_q, dz_mask, "dsm_z"))

    keys = _keys(d)
    if "keypoints" in keys:                      # мелочь, дешевле сохранить
        out["keypoints"] = np.asarray(d["keypoints"], dtype=np.float32)
    if "vertices" in keys:
        v = np.asarray(d["vertices"])
        # из меша читалась только медиана Z (высота съёмки в бенчмарке);
        # сама геометрия сцены уже есть в dsm
        out["median_vertex_z"] = np.float32(np.median(v[:, 2]))
        out["n_vertices"] = np.int32(len(v))
    if "faces" in keys:
        out["n_faces"] = np.int32(len(d["faces"]))
    return out


def write(path: Path, d, **kw) -> int:
    """Записать компактный сэмпл; возвращает размер получившегося файла."""
    path = Path(path)
    np.savez_compressed(path, **build(d, **kw))
    return path.stat().st_size


# ——————————————————————————————— чтение ———————————————————————————————

class SlimSample:
    """Компактный сэмпл с интерфейсом ``np.load``.

    Тяжёлые массивы (изображения, ``point_map``, ``dsm``) собираются по
    требованию и кэшируются: вызывающему коду не нужно знать, что на диске
    лежит не то же самое, что он просит.
    """

    FILES = ("sample_id", "image_query", "image_dop", "point_map", "dsm",
             "scale", "extrinsics", "intrinsics", "keypoints")

    def __init__(self, path: Path):
        self._z = np.load(path, allow_pickle=False)
        self._cache: dict = {}

    @property
    def files(self):
        return [k for k in self.FILES
                if k != "keypoints" or "keypoints" in self._z.files]

    @property
    def raw(self):
        """Массивы как они лежат в файле — для разбора объёма и отладки."""
        return self._z

    def __contains__(self, key):
        return key in self.files

    def __getitem__(self, key):
        if key not in self._cache:
            self._cache[key] = self._build(key)
        return self._cache[key]

    def _build(self, key):
        z = self._z
        if key in ("scale", "extrinsics", "intrinsics", "keypoints", "sample_id"):
            return z[key]
        if key in ("image_query", "image_dop"):
            enc = z["image_query_enc" if key == "image_query" else "image_dop_enc"]
            return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if key == "gt":
            packed = np.stack([decode_plane(z, "gt_x"), decode_plane(z, "gt_y")], -1)
            return unpack_gt(packed, z["gt_mask"], float(z["gt_step"]))
        if key == "point_map":
            gt_x, gt_y = self["gt"]
            pm_z = unpack_heights(decode_plane(z, "pm_z"), z["pm_z_mask"], z["pm_z0"])
            hd, wd = len(z["dsm_y"]), len(z["dsm_x"])
            return pointmap_from_gt(gt_x, gt_y, pm_z, z["scale"], wd, hd)
        if key == "dsm":
            zz = unpack_heights(decode_plane(z, "dsm_z"), z["dsm_z_mask"], z["dsm_z0"])
            h, w = zz.shape
            dsm = np.empty((h, w, 3), np.float32)
            dsm[..., 0] = z["dsm_x"][None, :]
            dsm[..., 1] = z["dsm_y"][:, None]
            dsm[..., 2] = zz
            return dsm
        raise KeyError(key)

    @property
    def median_vertex_z(self) -> float:
        """Медиана Z вершин меша — то единственное, что из меша читалось."""
        if "median_vertex_z" not in self._z.files:
            return float("nan")
        return float(self._z["median_vertex_z"])

    @property
    def profile(self) -> dict:
        """Чем записан файл: кодек изображений и шаг GT."""
        return dict(codec=str(self._z["image_codec"]),
                    gt_step=float(self._z["gt_step"]))

    def close(self):
        self._z.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class RawSample:
    """Исходный сэмпл OrthoLoC под тем же интерфейсом, что и компактный."""

    def __init__(self, path: Path):
        self._path = Path(path)
        self._z = np.load(path, allow_pickle=False)

    @property
    def files(self):
        got = list(self._z.files)
        return got if "scale" in got else got + ["scale"]

    @property
    def raw(self):
        return self._z

    def __contains__(self, key):
        return key in self._z.files

    def __getitem__(self, key):
        if key == "gt":
            dop = self._z["image_dop"]
            return gt_from_pointmap(self._z["point_map"], dop_scale(self._z),
                                    dop.shape[1], dop.shape[0])
        if key == "scale":
            # у сцены L06 массива нет — масштаб выводится из сетки dsm.
            # dop_scale смотрит именно ключи архива, а не наш интерфейс:
            # иначе files, где scale объявлен всегда, замкнул бы вызов на себя
            return dop_scale(self._z)
        return self._z[key]

    @property
    def median_vertex_z(self) -> float:
        with zipfile.ZipFile(self._path) as zf:
            if "vertices.npy" not in zf.namelist():
                return float("nan")
        return float(np.median(self._z["vertices"][:, 2]))

    def close(self):
        self._z.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def is_slim(path: Path) -> bool:
    """Компактный ли это сэмпл — по метке формата в архиве."""
    with zipfile.ZipFile(path) as zf:
        return "fmt.npy" in zf.namelist()


def open_sample(path: Path):
    """Открыть сэмпл любого из двух форматов единым интерфейсом."""
    path = Path(path)
    return SlimSample(path) if is_slim(path) else RawSample(path)
