# aero-geoloc — аэро-геолокализация надирных снимков

Определение географического положения кадра, снятого примерно в надир (с борта
БПЛА/самолёта), по картографической подложке. На вход — снимок и грубое
приближение (координаты ±5 км, высота, ориентация, параметры камеры); на выходе —
уточнённые координаты центра, курс, оценка высоты, отпечаток на карте и **честная
оценка качества** (эллипс ошибки + статус, вплоть до отказа).

> **Статус:** все фазы 0–5 реализованы; пайплайн **валидирован на реальных
> бортовых кадрах DJI** против Esri через appearance gap (LightGlue: 9 м vs GPS,
> где SIFT даёт 0 инлайеров). 331 тест зелёный (+6 gated по torch/сети/снимкам).
> Осталась калибровка доверия на объёме данных. Подробно — [docs/STATUS.md](docs/STATUS.md).

По сути это **итеративный georeferencing через image registration**: спутниковая
подложка вокруг приора → сопоставление кадра с ней → преобразование → координаты.

## Как это устроено

Система **двухэтажная** — это принципиальное разделение:

| | Этаж 1 — Retrieval | Этаж 2 — Pose |
|---|---|---|
| Вопрос | «ГДЕ примерно?» | «ГДЕ точно и под каким поворотом?» |
| Механизм | эмбеддинг → ANN-поиск | матчинг точек → подобие |
| Точность | размер клетки | пиксель / субпиксель |

Эмбеддинг физически не даёт метровую точность, поэтому не заменяет матчер, а
снимает с него дорогой грубый поиск в диске ±5 км. Центральное инженерное
решение — **сменные ядра**: матчер (`Matcher`) и энкодер (`Encoder`) спрятаны за
интерфейсами, и всё вокруг (георефа, pose, refinement, quality, retrieval, стенд)
от их внутренностей не зависит. Классика (SIFT/AKAZE) и обучаемые
(LightGlue/LoFTR/DINOv2) подключаются одной строкой.

Рантайм-конвейер: нормализация → retrieval (top-K клеток + сигнал уникальности) →
матчинг → RANSAC-подобие (4 DoF, приор как ограничение) → субпиксельный
ECC-refinement + цикл переуточнения масштаба → считывание координат через
`Georef` → ковариация центра + статус. Для низких высот — последовательностный
режим (визуальная одометрия + EKF).

## Установка

Python 3.10+. Наборы зависимостей разложены по задачам:

```bash
pip install -r requirements-dev.txt
```

| Файл | Для чего | Что тянет |
|---|---|---|
| `requirements.txt` | только ядро | NumPy, OpenCV (`opencv-contrib-python`) |
| `requirements-dev.txt` | тесты и синтетический стенд | + pytest |
| `requirements-real.txt` | боевой конвейер на реальных кадрах | + torch, LightGlue, FAISS, Pillow |
| `requirements-lock.txt` | воспроизведение измерений | точные версии оффлайн-окружения |

- Работает одинаково в **bash** и **PowerShell** (в последнем `pip install -e '.[dev]'`
  ломается на квадратных скобках, а `-r` — нет).
- Загрузка тайлов подложки — через stdlib, без доп. зависимостей.
- Для GPU ставьте `torch`/`torchvision` с индекса PyTorch **до** установки
  `requirements-real.txt`; **LightGlue** тянется из GitHub (на PyPI его нет),
  поэтому нужен `git`. Подробности и подводные камни — в шапке того файла.
- Веса моделей (MegaLoc, DINOv2, LightGlue) скачиваются при первом запуске.

## Быстрый старт

```python
from aero_geoloc import localize, Camera, Prior, TileCache
from aero_geoloc.basemap import TileBasemap

camera = Camera(image_width=512, image_height=512, focal_mm=28.0, sensor_width_mm=8.03)
prior = Prior(lat=55.7558, lon=37.6173, sigma_m=200,
              altitude_m=600, altitude_sigma_m=100, yaw_deg=137)

result = localize(image, camera, prior, TileBasemap(cache=TileCache("tiles")))
print(result.status)                       # LOCALIZED / LOW_CONFIDENCE / NOT_LOCALIZED
print(result.center_lat, result.center_lon)
print(result.heading_deg, result.altitude_est_m)
print(result.error_ellipse_m)              # (semi-major, semi-minor, angle), метры
```

Для офлайн-работы (тесты, стенд) источником служит `SceneBasemap(scene)` поверх
процедурной сцены — тот же интерфейс `BasemapSource`, но без сети. Смена ядра
матчинга — `localize(..., matcher=create_matcher("lightglue"))`; retrieval-этаж —
`localize(..., index=TerrainIndex.load("index.npz", encoder))`.

## Измеренные результаты (синтетика)

- **Субпиксельный ECC-refinement** сдвинул медиану ошибки центра
  0.505 px → **0.011 px** (0.004 м), 120/120 без ложняков; после ECC точность не
  зависит от матчера.
- **Coarse-to-fine на реальных тайлах Esri** (широкий приор): ошибка 0.1 см.
- **Цикл масштаба** (ошибка высоты ×2): 4.48 см → 0.30 см, высота восстановлена.
- **Ковариация центра откалибрована**: покрытие эллипса совпадает с номиналом
  (Монте-Карло 0.384/0.857 против 0.393/0.865).
- **Retrieval**: Recall@5 = 1.0; порог уникальности отделяет разрешимую местность
  от однородной (балансная точность 0.88).
- **Лестница возмущений L1–L4**: точка перелома измерена — AKAZE устойчивее на
  блюре/шуме (L2), SIFT — на спектральном сдвиге (L3); провалы комплементарны.
- **Реальные бортовые кадры DJI** (LightGlue против Esri, appearance gap
  лето↔весна): ошибка **9 м** vs GPS, где SIFT/LoFTR дают 0 инлайеров; на
  экстремальном кросс-сезонном gap — честный отказ. См. `scripts/localize_real.py`.

## Тесты и стенд

```bash
.venv/Scripts/python.exe -m pytest
```

```bash
.venv/Scripts/python.exe scripts/run_benchmark.py --matcher sift --yaw-step 30 --refine
```

```bash
.venv/Scripts/python.exe scripts/run_ladder.py --matcher sift
```

Вывод русскоязычный: при запуске через bash нужен префикс `PYTHONIOENCODING=utf-8`.

## Структура

```
aero_geoloc/
├── geo.py         Web Mercator, Georef — единственный мост пиксель↔координата
├── camera.py      модель камеры: GSD, footprint, ректификация наклона
├── types.py       Prior, LocalizationRequest, LocalizationResult, Status
├── matcher.py     ★ сменное ядро: SIFT/AKAZE + LightGlue/LoFTR (gated)
├── pose.py        similarity/RANSAC, приоры-ограничения, ECC-refinement
├── basemap.py     загрузка/сшивка XYZ-тайлов Esri, кэш, BasemapSource
├── quality.py     ковариация центра, эллипс, статус/отказ
├── retrieval.py   ★ Этаж 1: индекс местности, Recall@K, уникальность, DINOv2
├── localize.py    оркестрация: coarse-to-fine, retrieval, цикл масштаба
├── sequence.py    визуальная одометрия + EKF (низкие высоты)
├── drone.py       загрузка бортового снимка (EXIF/XMP → камера, GPS, курс)
└── testbench.py   синтетика, лестница, траектории, метрики
scripts/           run_benchmark · run_ladder · build_index · localize_real
tests/             331 тест
```

## Документация

| Документ | О чём |
|---|---|
| [docs/README.md](docs/README.md) | Полный обзор: возможности, ограничения, дорожная карта |
| [docs/STATUS.md](docs/STATUS.md) | Текущее состояние, метрики, принятые решения |
| [docs/PLAN.md](docs/PLAN.md) | Поэтапный план и стратегия тестирования |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Двухэтажная схема, границы модулей, форматы |
| [docs/PIPELINE.md](docs/PIPELINE.md) | Стадии обработки 0–7: формулы, модель, refinement |
| [docs/RETRIEVAL.md](docs/RETRIEVAL.md) | «Хеширование местности»: индекс, ротация, калибровка |
| [docs/TESTING.md](docs/TESTING.md) | Стенд, метрики, калибровка доверия, точка перелома |

## Принципы

- **Стенд раньше кода**: метрика существует до усложнения алгоритма.
- **Приоры — ограничения, а не подсказки**: решение вне диска ±3σ отбрасывается.
- **Честный отказ дороже красивой точки**: `NOT_LOCALIZED` — легитимный исход.
- **Сменные ядра**: смена матчера/энкодера не трогает обвязку.
