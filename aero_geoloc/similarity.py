"""Кандидаты в меру согласия кадра и подложки — то, чем можно заменить NCC.

Зачем модуль ([ROADMAP.md](../docs/ROADMAP.md), фаза 1; обзор — `RESEARCH_B_VERIFICATION.md`).
Наблюдение обзора: NCC выполняет **две разные работы** — меряет качество
выравнивания и служит дискриминатором «то место или не то». Смена сезона ломает
первую наверняка (яркости не связаны линейно), а вторую — только если сдвиг у
верных и ложных пар различается. Измерено на наборе: у летних кадров NCC
0.53–0.69, у апрельских — 0.04–0.11 при **верной** позе, то есть шкала уезжает на
порядок и общий порог 0.12 режет верное.

Все меры здесь построены на структуре, а не на яркости, и потому переживают
монотонное и даже инвертирующее преобразование интенсивности:

===================  =========================================================
``ncc``              база для сравнения: линейная связь яркостей
``grad_ncc``         NCC по модулю градиента — «сколько даёт просто переход к
                     градиентам», контроль к CFOG
``cfog``             дескрипторы ориентированных градиентов; фаворит обзора —
                     «NCC в сезонно-устойчивом домене»
``ngf``              нормированные поля градиентов: совпадение НАПРАВЛЕНИЙ
``nmi``              нормированная взаимная информация: любая статистическая
                     связь яркостей, не только линейная
``edge_dice``        доля совпавших краёв с допуском в пикселях
===================  =========================================================

Контракт: обе картинки — **grayscale одного размера и уже выровненные**. Маску
валидности модуль не знает намеренно; вызывающий обязан подать область, где обе
картинки определены (после поворота это вписанный прямоугольник). Иначе чёрные
углы попадут в статистику и подтянут любую меру.

Больше — лучше у всех мер. Диапазоны разные (``ncc`` в ``[-1, 1]``, ``nmi`` в
``[1, 2]``), поэтому сравнивать их между собой можно только по разделяющей
способности, а не по абсолютной величине.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = [
    "SIGNALS", "ncc", "grad_ncc", "cfog", "ngf", "nmi", "edge_dice",
    "cfog_descriptor", "DenseDinoSimilarity", "dense_dino",
]


def _prepare(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"меры согласия ждут grayscale, получено {a.ndim}D и {b.ndim}D")
    if a.shape != b.shape:
        raise ValueError(f"размеры не совпадают: {a.shape} против {b.shape}")
    return a.astype(np.float32), b.astype(np.float32)


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Нормированная кросс-корреляция яркостей — та самая база, что ломается сезоном."""
    x, y = _prepare(a, b)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return 0.0 if denom == 0.0 else float((x * y).sum() / denom)


def _gradients(img: np.ndarray, *, blur: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    src = cv2.GaussianBlur(img, (0, 0), blur) if blur > 0 else img
    gx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy


def grad_ncc(a: np.ndarray, b: np.ndarray, *, blur: float = 1.0) -> float:
    """NCC по модулю градиента — контрольная мера.

    Нужна, чтобы не приписать CFOG заслугу, которую даёт сам по себе переход от
    яркостей к градиентам: если ``grad_ncc`` разделяет так же, городить
    дескрипторы незачем.
    """
    x, y = _prepare(a, b)
    gxa, gya = _gradients(x, blur=blur)
    gxb, gyb = _gradients(y, blur=blur)
    return ncc(np.hypot(gxa, gya), np.hypot(gxb, gyb))


def cfog_descriptor(img: np.ndarray, *, bins: int = 8, sigma: float = 2.0,
                    blur: float = 1.0) -> np.ndarray:
    """Плотный дескриптор CFOG: ``(H, W, bins)``, каждый пиксель — единичный вектор.

    Схема (Ye et al., «Channel Features of Orientated Gradients»): проекции
    градиента на ``bins`` направлений → сглаживание каждого канала по
    пространству → сглаживание **по ориентациям** ядром ``[1, 2, 1]`` →
    вычитание среднего по каналам → L2 на пиксель.

    Модуль проекции (``abs``) взят намеренно: он делает дескриптор нечувствительным
    к **смене знака градиента**. Летом поле темнее дороги, весной — светлее; край
    тот же, знак перепада обратный. Именно на этом рассыпается NCC.
    """
    src = img.astype(np.float32)
    gx, gy = _gradients(src, blur=blur)
    angles = np.arange(bins, dtype=np.float32) * (np.pi / bins)
    channels = [np.abs(gx * float(np.cos(t)) + gy * float(np.sin(t))) for t in angles]
    if sigma > 0:
        channels = [cv2.GaussianBlur(c, (0, 0), sigma) for c in channels]
    stack = np.stack(channels, axis=-1)
    # Сглаживание по кругу ориентаций: соседние направления не независимы.
    smoothed = (np.roll(stack, 1, axis=-1) + 2.0 * stack + np.roll(stack, -1, axis=-1)) / 4.0
    # Вычитание среднего по каналам — не косметика, а необходимость. Проекции
    # взяты по модулю, поэтому все дескрипторы лежат в положительном октанте, и
    # косинус двух ПРОИЗВОЛЬНЫХ таких векторов ≈0.95. Измерено: без центрирования
    # верная пара давала 0.958, а чужое место 0.950 — мера формально работала, а
    # практически была неотличима от константы. После центрирования пол меры —
    # ноль, и весь диапазон снова рабочий.
    smoothed = smoothed - smoothed.mean(axis=-1, keepdims=True)
    norm = np.linalg.norm(smoothed, axis=-1, keepdims=True)
    # Плоский пиксель остаётся НУЛЁМ, а не нормируется в случайное направление:
    # нормировка шума породила бы структуру там, где её нет.
    return np.where(norm > 1e-6, smoothed / np.maximum(norm, 1e-6), 0.0).astype(np.float32)


def cfog(a: np.ndarray, b: np.ndarray, *, bins: int = 8, sigma: float = 2.0,
         blur: float = 1.0) -> float:
    """Согласие по CFOG: среднее скалярное произведение единичных дескрипторов.

    Значение в ``[−1, 1]``: 1 — структура совпала пиксель в пиксель, около нуля —
    связи нет. Это и есть «NCC, перенесённая в домен, где сезон не меняет
    описание».

    Усреднение идёт по пикселям, структурным **хотя бы на одной** картинке. Так
    гладкая вода и поля не разбавляют меру (иначе снимок с большой однородной
    зоной получал бы низкий балл при идеальном совпадении), но и не прощают
    случай «здесь структура есть, а там её нет»: у такой пары произведение равно
    нулю и в среднее попадает.
    """
    x, y = _prepare(a, b)
    fa = cfog_descriptor(x, bins=bins, sigma=sigma, blur=blur)
    fb = cfog_descriptor(y, bins=bins, sigma=sigma, blur=blur)
    structured = (np.linalg.norm(fa, axis=-1) > 0.5) | (np.linalg.norm(fb, axis=-1) > 0.5)
    if int(np.count_nonzero(structured)) < 16:
        return 0.0
    return float(np.mean(np.sum(fa * fb, axis=-1)[structured]))


def ngf(a: np.ndarray, b: np.ndarray, *, eta: float | None = None,
        blur: float = 1.0) -> float:
    """Нормированные поля градиентов: средний ``cos²`` угла между перепадами.

    Считается **только по структурным пикселям** — тем, где градиент в обеих
    картинках больше ``eta``. Квадрат снимает знак: противоположно направленные
    градиенты на одном и том же крае считаются согласными — то же соображение,
    что и ``abs`` в CFOG.

    Важное свойство, которое надо знать при выставлении порога: **пол этой меры
    не ноль, а 0.5**. У независимых картинок направления градиентов случайны, а
    среднее ``cos²`` двух случайных направлений на плоскости равно ½. Значение
    0.5 означает «связи нет», и порог имеет смысл только выше него.

    ``eta`` по умолчанию — десятая доля средней величины градиента: она отсекает
    плоские участки, где направление задаёт шум.
    """
    x, y = _prepare(a, b)
    gxa, gya = _gradients(x, blur=blur)
    gxb, gyb = _gradients(y, blur=blur)
    mag_a, mag_b = np.hypot(gxa, gya), np.hypot(gxb, gyb)
    if eta is None:
        eta = 0.1 * float(mag_a.mean() + mag_b.mean()) / 2.0
    eta = float(max(eta, 1e-6))
    mask = (mag_a > eta) & (mag_b > eta)
    if int(np.count_nonzero(mask)) < 16:
        return 0.5      # сравнивать нечего — сообщаем «связи нет», а не «совпало»
    dot = (gxa * gxb + gya * gyb)[mask]
    denom = (mag_a[mask] * mag_b[mask]) ** 2
    return float(np.mean(dot * dot / np.maximum(denom, 1e-12)))


def nmi(a: np.ndarray, b: np.ndarray, *, bins: int = 32) -> float:
    """Нормированная взаимная информация ``(H(a)+H(b))/H(a,b)``.

    1.0 — яркости независимы, 2.0 — одна однозначно определяет другую. В отличие
    от NCC не требует **линейной** связи: годится, когда сезон переставил уровни
    серого произвольным, но устойчивым образом.
    """
    x, y = _prepare(a, b)
    joint, _, _ = np.histogram2d(x.ravel(), y.ravel(), bins=bins)
    joint = joint / max(joint.sum(), 1.0)
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    def _entropy(p: np.ndarray) -> float:
        nz = p[p > 0]
        return float(-(nz * np.log(nz)).sum())

    h_joint = _entropy(joint)
    if h_joint <= 0.0:
        return 2.0
    return (_entropy(px) + _entropy(py)) / h_joint


def edge_dice(a: np.ndarray, b: np.ndarray, *, tol_px: int = 2,
              low: int = 50, high: int = 150) -> float:
    """Доля краёв, нашедших пару в пределах ``tol_px`` (симметрично).

    Самая грубая из мер и самая понятная глазами: дороги, здания и берега дают
    края в обоих сезонах, даже когда яркости не имеют ничего общего. Допуск в
    пикселях нужен, потому что идеального совпадения краёв не бывает и без сезона.
    """
    x, y = _prepare(a, b)
    ea = cv2.Canny(cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), low, high)
    eb = cv2.Canny(cv2.normalize(y, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), low, high)
    total = int(np.count_nonzero(ea)) + int(np.count_nonzero(eb))
    if total == 0:
        return 0.0
    kernel = np.ones((2 * tol_px + 1, 2 * tol_px + 1), np.uint8)
    da = cv2.dilate(ea, kernel)
    db = cv2.dilate(eb, kernel)
    matched = int(np.count_nonzero(ea & db)) + int(np.count_nonzero(eb & da))
    return matched / total


#: Меры, считающиеся без torch. Имя → функция; порядок — как в таблицах отчётов.
SIGNALS = {
    "ncc": ncc,
    "grad_ncc": grad_ncc,
    "cfog": cfog,
    "ngf": ngf,
    "nmi": nmi,
    "edge_dice": edge_dice,
}


class DenseDinoSimilarity:
    """Косинус плотных признаков DINOv2 — самая сильная априорная ставка обзора.

    Основание не теоретическое, а измеренное: **на этом же наборе** глобальный
    дескриптор того же семейства (MegaLoc, Этаж 1) держит апрельские кадры на
    ранге 1–2 против летней подложки. То есть представление сезон переживает —
    вопрос лишь в том, сохраняется ли эта устойчивость на уровне патчей, где
    решается «то место или не то».

    Считается как среднее косинусов между патч-токенами в одинаковых позициях:
    картинки уже выровнены, поэтому патч ↔ патч сопоставляются напрямую.

    torch грузится лениво, как у :class:`aero_geoloc.retrieval.DinoV2Encoder`:
    без него импорт пакета не должен падать.
    """

    def __init__(self, model_name: str = "dinov2_vitb14", *, image_size: int = 518,
                 device: str | None = None, fp16: bool = True) -> None:
        if image_size % 14 != 0:
            raise ValueError(f"image_size должен быть кратен патчу 14, получено {image_size}")
        self.model_name = model_name
        self.image_size = image_size
        self.fp16 = fp16
        self._device = device
        self._model = None
        self._torch = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
        except ImportError as exc:  # pragma: no cover — зависит от окружения
            raise RuntimeError(
                "DenseDinoSimilarity требует torch: pip install -r requirements-real.txt"
            ) from exc
        self._torch = torch
        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = torch.hub.load("facebookresearch/dinov2", self.model_name)
        self._model = self._model.to(self._device).eval()

    def _tokens(self, img: np.ndarray):
        torch = self._torch
        resized = cv2.resize(img.astype(np.float32), (self.image_size, self.image_size),
                             interpolation=cv2.INTER_AREA)
        arr = (resized / 255.0 - 0.449) / 0.226          # серый канал по нормировке ImageNet
        tensor = torch.from_numpy(arr)[None, None].to(self._device).expand(1, 3, -1, -1)
        with torch.no_grad():
            if self.fp16 and str(self._device).startswith("cuda"):
                with torch.autocast("cuda", dtype=torch.float16):
                    out = self._model.forward_features(tensor)["x_norm_patchtokens"]
            else:
                out = self._model.forward_features(tensor)["x_norm_patchtokens"]
        return torch.nn.functional.normalize(out.float()[0], dim=-1)

    def __call__(self, a: np.ndarray, b: np.ndarray) -> float:
        a, b = _prepare(a, b)
        self._ensure_model()
        ta, tb = self._tokens(a), self._tokens(b)
        return float((ta * tb).sum(dim=-1).mean().item())


#: Разделяемый экземпляр: модель весит сотни мегабайт, и создавать её на каждый
#: кандидат веера нельзя. Кэш на процесс, а не глобальная переменная в коде
#: пайплайна, — чтобы владение оставалось у модуля меры.
_SHARED: dict[str, DenseDinoSimilarity] = {}


def dense_dino(model_name: str = "dinov2_vitb14", **kwargs) -> DenseDinoSimilarity:
    """Общий на процесс :class:`DenseDinoSimilarity` — по одной модели на имя."""
    key = model_name + repr(sorted(kwargs.items()))
    if key not in _SHARED:
        _SHARED[key] = DenseDinoSimilarity(model_name, **kwargs)
    return _SHARED[key]
