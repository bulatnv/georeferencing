# Сравнительное исследование VPR-энкодеров для Этажа 1 (retrieval) геолокализации БПЛА по спутниковой подложке

## TL;DR
- Приоритет для zero-shot прогона на стенде: **(1) MegaLoc → (2) SALAD → (3) AnyLoc-VLAD-DINOv2 → (4) BoQ-DINOv2 → (5) NetVLAD (нижняя планка)**. MegaLoc и SALAD — это «сильный DINOv2-бэкбон + обученная агрегация», ровно та гипотеза, которая должна поднять DRZ_19206 из ранга 506; AnyLoc — сильнейший zero-shot универсал с прямыми доказательствами на аэро/надирных датасетах.
- Замена «сырого DINOv2 mean-token (dim=384)» на обученную VLAD/SALAD/BoQ-агрегацию почти наверняка поднимет верную клетку в top-K: на надир↔ортоспутник (Nardo-Air) в работе FoundLoc AnyLoc-DINO даёт **R@1=94.37, R@5=100, Top-3@5=100% zero-shot** (против NetVLAD-finetuned Top-3@5=52.11), а сама AnyLoc-статья показала, что доменные VLAD-словари дают **до 4× выше Recall@1**, тогда как сырой mean/CLS «сжимается в малую область признакового пространства, теряя различительную способность». Ваш gap — appearance (дата/сезон/тени), а не ракурс, что легче классических cross-view бенчмарков.
- Лицензионное предупреждение критично для закрытого/коммерческого проекта: **SALAD — GPL-3.0 (копилефт, блокер для проприетарного продукта)**; **MegaLoc — веса MIT, но код репозитория без LICENSE-файла и заимствует GPL-код SALAD** (нужна чистая реализация инференса); **BoQ — MIT (чистый)**; **AnyLoc/NetVLAD — permissive**; базовый **DINOv2-бэкбон сегодня Apache-2.0** (коммерчески разрешён; исходный релиз 2023 был CC-BY-NC — тяните актуальные веса).

## Key Findings

1. **Гипотеза верна для нашего домена.** Все кандидаты используют сильный бэкбон (DINOv2 или обученный CNN) + агрегацию, обученную отделять места. AnyLoc прямо показал, что сырой CLS/mean-токен DINOv2 сильно проигрывает VLAD-агрегации на аэро- и надирных данных: доменные VLAD-словари дают до 4× выше Recall@1, а признаки специализированного MixVPR, наоборот, «концентрируются в малой области, теряя различительность».
2. **Надир↔надир проще, чем кажется.** Ваш случай (надирный кадр против ортоспутника, gap в основном appearance) ближе к «aerial-to-aerial» бенчмаркам (VPAir, Nardo-Air, ALTO), где универсальные VPR (AnyLoc, MegaLoc, SALAD) работают хорошо zero-shot, чем к косым cross-view (University-1652/SUES-200), под которые заточены Sample4Geo/FSRA/DAC.
3. **Rotation (yaw) — главный риск.** ViT/DINOv2 НЕ инвариантны к повороту (фиксированные позиционные эмбеддинги). Показательно: FoundLoc для надирной локализации держал «yaw ... всегда направленным на север» — то есть prerotate/фиксацию ориентации, а не аугментацию. Для нас это решается либо prerotate запроса на −yaw (если yaw известен из IMU/компаса), либо ротационной аугментацией индекса ×8–12.
4. **Специализированные drone→satellite модели (Sample4Geo, FSRA, DAC) не дают zero-shot преимущества** в нашем надир↔надир сценарии: они обучены на конкретных cross-view датасетах (CVUSA/CVACT/University-1652), их дескрипторы и веса заточены под их протокол, и они, как правило, сиамские (отдельные ветки drone/satellite). Интеграция дороже, а выигрыш неочевиден.
5. **MegaLoc — актуальный SOTA-универсал (CVPR 2025 Workshops)** и лучший первый кандидат: DINOv2-base + SALAD-агрегация, обучен на пяти разнородных датасетах, веса MIT.

## Details

### Базовая линия: сырой DINOv2 (mean/CLS-токен)
- **Суть:** ViT-S/14 (dim=384) → усреднённый патч-токен или CLS. Никакой VPR-агрегации, никакого дообучения.
- **Поведение:** AnyLoc эмпирически показал, что «per-image» CLS/mean-дескрипторы фундаментально слабее агрегации локальных патч-фич для VPR, особенно в неструктурированных/аэро-доменах. Это ровно ваш DRZ_19206 на ранге 506 из 2116.
- **Размерность:** 384 (ViT-S), 768 (ViT-B), 1024 (ViT-L), 1536 (ViT-g). Лицензия Apache-2.0.

### 1. NetVLAD (Arandjelović et al.) — нижняя планка
- **Статья:** «NetVLAD: CNN architecture for weakly supervised place recognition», CVPR 2016.
- **Агрегация/бэкбон:** обучаемый VLAD-слой (soft-assignment к кластерам) поверх VGG-16/ResNet.
- **Размерность:** типично 4096 (после PCA-whitening); нативно 32×512=16384 → редуцируют.
- **Аэро-поведение:** на Nardo-Air (FoundLoc, arXiv:2310.16299, Table I) NetVLAD zero-shot очень слаб (R@1=42.25, R@5=76.06), после дообучения на домене — R@1=78.87, R@5=100, но Top-3@5 всего 52.11 — «тревожное число ложноположительных в top-5 даже после дообучения». Требует дообучения на аэро-домен.
- **Веса/лицензия:** веса берутся из hloc; код permissive. Yaw: не инвариантен; VLAD-суммирование даёт частичную устойчивость к перестановке, но не к повороту патчей.
- **Роль:** контрольная нижняя планка «настоящего VPR».

### 2. AnyLoc (Keetha et al., RA-L 2023) — zero-shot универсал
- **Статья:** «AnyLoc: Towards Universal Visual Place Recognition», IEEE RA-L 2023 (arXiv:2308.00688).
- **Агрегация/бэкбон:** VLAD (обычно 32 кластера, hard-assignment) поверх **замороженного** DINOv2 ViT-G/14, layer 31, value-facet. Никакого дообучения — только словарь кластеров (можно доменный, «aerial»).
- **Размерность:** 32×1536 = **49152** (очень большая). PCA до ~1024–4096 почти без потерь (AnyLoc-VLAD-DINOv2-PCA сопоставим с полным).
- **Аэро/надир-доказательства (сильнейшие среди кандидатов):**
  - Nardo-Air (drone→ortho-satellite): в AnyLoc-статье R@1≈99.5%; на усреднении по «aerial» R@1=86.5.
  - FoundLoc (Table I): AnyLoc-DINO на Nardo-Air **R@1=94.37, R@5=100, Top-3@5=100% zero-shot** — «значительно выше Recall@N и 100% Top-3@5 без всякого дообучения», что «гарантирует надёжные георефы во время полёта».
  - VP-Air (aerial-to-aerial, 300 м, downward): AnyLoc — сильнейший zero-shot глобальный дескриптор (R@1≈0.62 в независимом proxy-тесте MDPI).
  - AnyLoc-статья: доменные VLAD-словари дают **до 4× выше Recall@1** + ещё ~6% от семантической характеризации фич.
- **Rotation:** Nardo-Air-R (ротированный) показал деградацию при экстремальных ортогональных ракурсах, но для надир↔надир с известным yaw prerotate снимает проблему; VLAD частично устойчив.
- **Стоимость:** ViT-G14 тяжёлый; в aerial-VPR survey (aero-vloc) именно AnyLoc заметно медленнее прочих из-за ViT-G и dim=49152, и «существенно удлиняет» вычисление дескрипторов и поиск. VRAM высокий. Есть облегчённые варианты (ViT-B/-S).
- **Веса/лицензия:** torch.hub `AnyLoc/DINO`, permissive; бэкбон DINOv2 Apache-2.0. **Zero-shot — да**, дообучение не требуется.

### 3. SALAD (Izquierdo & Civera, CVPR 2024)
- **Статья:** «Optimal Transport Aggregation for Visual Place Recognition», CVPR 2024 (arXiv:2311.15937).
- **Агрегация/бэкбон:** «Sinkhorn Algorithm for Locally Aggregated Descriptors» — переформулировка soft-assignment NetVLAD как задачи оптимального транспорта, с учётом feature-to-cluster и cluster-to-feature отношений + «dustbin»-кластер для отбрасывания неинформативных фич; поверх **дообученного** DINOv2 ViT-B (последние 4 блока размораживаются). 64 кластера, редукция 768→128 (патчи) + 256 (CLS).
- **Размерность:** **8448** (64×128 + 256). Поддерживает разные размеры дескриптора (варианты 512×32, 2048×64).
- **Производительность:** SOTA на MSLS/Nordland/Pitts; обучение всего ~30 мин на RTX 3090. Инференс DINOv2-B ~2.4 мс/изобр (таблица DINOv2-конфигураций в статье).
- **Аэро/надир:** в aerial-VPR survey (aero-vloc) SALAD входит в топ-методы для глобальной локализации на VPAir/ALTO; ViT-бэкбон устойчивее CNN на аэро.
- **Rotation:** ViT — не инвариантен; нужен prerotate/аугментация.
- **Веса/лицензия:** веса `dino_salad.ckpt` (Google Drive / torch.hub `serizba/salad`). **Код репозитория — GPL-3.0** (подтверждено LICENSE-файлом); отдельной лицензии на веса нет → де-факто GPL-3.0. **Копилефт-блокер: GPL-3.0 не позволяет включать программу в проприетарные продукты.** Бэкбон DINOv2 Apache-2.0.
- **Zero-shot:** веса уже обучены на GSV-Cities (городской домен) — для нас это zero-shot прогон (без нашего дообучения), но домен обучения городской.

### 4. MegaLoc (Berton & Masone, CVPR 2025 Workshops) — топ-кандидат
- **Статья:** «MegaLoc: One Retrieval to Place Them All», CVPR 2025 Workshops (arXiv:2502.17237).
- **Агрегация/бэкбон:** DINOv2-base ViT + **SALAD-агрегация** (~228M параметров). Мульти-датасетное обучение на **пяти датасетах: SF-XL, GSV-Cities, MSLS, MegaScenes, ScanNet** — покрытие indoor/outdoor/day/night/seasonal и landmark/visual-localization. Обучен существенно шире, чем SALAD. Memory-efficient обучение снижает VRAM с ~300 ГБ до 60 ГБ.
- **Размерность:** финальная проекция маппит выход с 16640 до **8448** измерений, затем L2-норма; на практике PCA до 1024 для индекса (как в Netryx Astra V2).
- **Аэро/надир-доказательства:** на UAV-VisLoc и DenseUAV MegaLoc — один из сильнейших retrieval (в CAEVL-статье MegaLoc и DAC дают высший recall на синтетическом DenseUAV); в AirZoo дообучение MegaLoc на аэро даёт существенный прирост (значит, база уже сильная и дообучаемая — на UAV-VisLoc +11.42 R@1 в лучшей сцене). В SEALOC MegaLoc и AnyLoc — топ по Recall@1/@10 среди ViT-методов.
- **Rotation:** ViT — не инвариантен; prerotate/аугментация обязательны.
- **Стоимость:** DINOv2-base «~5× тяжелее CosPlace-бэкбона»; индексация 1 км радиуса ~20–30 мин на потребительском GPU (оценка автора Netryx). На RTX 4090 инференс ~2–4 мс/изобр (по DINOv2-B).
- **Веса/лицензия:** torch.hub `gmberton/MegaLoc` и HF `gberton/MegaLoc`; **веса MIT** (карточка HF). **НО код репозитория без LICENSE-файла (по умолчанию all-rights-reserved) и `megaloc_model.py` явно заимствует код из SALAD (GPL-3.0)** → для закрытого продукта нужна чистая реализация инференса поверх MIT-весов. Бэкбон DINOv2 Apache-2.0.
- **Zero-shot:** да, из коробки; самый широкий домен обучения из всех.

### 5. BoQ (Ali-bey et al., CVPR 2024)
- **Статья:** «BoQ: A Place is Worth a Bag of Learnable Queries», CVPR 2024, pp. 17794–17803 (arXiv:2405.07364).
- **Агрегация/бэкбон:** набор обучаемых глобальных запросов (Bag-of-Queries), которые через cross-attention «зондируют» локальные фичи; конкатенация выходов блоков + линейная проекция + L2. Работает с CNN (ResNet50) и ViT (DINOv2).
- **Размерность:** ResNet50-BoQ = **16384**; DINOv2-BoQ = **12288**.
- **Производительность:** «новый SOTA на 14 крупномасштабных бенчмарках»; первый глобальный метод, достигший **95% R@1 на Pitts250k-test** (официально 96.6), MSLS-val R@1 93.8, SVOX-night 97.7; «>30× быстрее» two-stage R²Former.
- **Аэро/надир:** прямых надир↔спутник замеров меньше, чем у AnyLoc/MegaLoc; заточен под городской ground-VPR (GSV-Cities). Это его главный минус для нашего домена.
- **Rotation:** ViT/CNN — не инвариантен.
- **Веса/лицензия:** torch.hub `amaralibey/bag-of-queries`, вход 320×320. **Код MIT (чистый)**; веса без отдельной лицензии → де-факто MIT. **Самая чистая лицензия для коммерции** (особенно ResNet50-вариант без DINOv2).
- **Zero-shot:** да, но домен обучения городской ground.

### Опциональные модели (кратко)
- **MixVPR (WACV 2023):** all-MLP mixing поверх ResNet-50, dim 4096. Быстрый, но AnyLoc показал, что на аэро/надире DINOv2-CLS обгоняет MixVPR (на Nardo-Air на 41%, на VP-Air на 35% при сильных ракурсах). Не приоритет.
- **CosPlace (CVPR 2022) / EigenPlaces (ICCV 2023):** классификационное обучение, ResNet, dim 512/2048; viewpoint-robust (EigenPlaces). Дешёвый индекс, но на аэро слабее ViT-методов. Полезны как быстрые бэйзлайны.
- **CricaVPR (CVPR 2024):** cross-image correlation поверх DINOv2, dim 10752. Хорош на кросс-условиях, но большая размерность.
- Все они доступны через `gmberton/VPR-methods-evaluation` (единый враппер — идеально для вашего стенда).

### Свежие 2025–2026 модели
- **Pair-VPR (RA-L 2025):** two-stage (глобальный дескриптор + pair-classifier реранкинг), ViT, siamese masked image modelling. У вас реранкинг уже делает LightGlue → брать только глобальную ветку. Не обязателен в шорт-лист.
- **EffoVPR (ICLR 2025):** self-attention фичи из замороженного DINOv2 как zero-shot реранкер; интересен, но это реранкинг, а не глобальный индекс.
- **SelaVPR / SelaVPR++:** адаптеры поверх DINOv2, global+local. В aero-vloc SelaVPR был среди топ-методов на VPAir.
- **DINOv3-based агрегации:** DINOv3 (Aug 2025, Gram anchoring, 1.7B изобр, включая аэро) — лучшие dense-фичи и коммерчески-дружелюбная (кастомная, но разрешает commercial) лицензия. Пока нет зрелого готового VPR-энкодера на DINOv3 с публичными весами; кандидат «на потом».
- **Вывод:** MegaLoc остаётся самым практичным SOTA-универсалом с готовыми весами; новее — либо реранкеры (не наш этаж), либо пока без зрелых весов.

### Rotation augmentation vs prerotate на −yaw
- **Prerotate на −yaw (предпочтительно, если yaw известен):** поворачиваем запрос (или кроп подложки) на −yaw перед энкодингом, индекс держим в одной («северной») ориентации. Экономит память индекса в 8–12×, снимает главную слабость ViT. Так делает FoundLoc (yaw фиксирован на север), AeroBEV/BEVLoc (fine-network выравнивает кроп по текущему yaw робота) и ViT-VS (перебор {0°,90°,180°,−90°}, выбор по max cosine similarity).
- **Ротационная аугментация индекса ×8–12:** надёжнее, если yaw неизвестен/ненадёжен, но раздувает индекс и стоимость поиска. Ваш план ×8–12 корректен как fallback.
- **Рекомендация:** если есть надёжный компас/IMU — prerotate; иначе аугментация ×8 (шаг 45°) как компромисс, ×12 (30°) для повышенной точности.

## Сравнительная таблица

| Модель | Кросс-домен дрон↔спутник | Кросс-дата устойчивость | Инвариантность к yaw | Масштаб (footprint 100–600 м) | Dim / память индекса | Стоимость инференса (RTX 4090) | Веса + лицензия | Zero-shot / дообучение |
|---|---|---|---|---|---|---|---|---|
| **Raw DINOv2 (mean/CLS)** | Слабая (ваш случай, ранг 506) | Средняя | Нет (ViT) | Средний | 384 → 1.5 КБ/вектор (fp32) | ~1.3 мс (ViT-S) | Apache-2.0 | **Zero-shot** |
| **NetVLAD** | Слабая zero-shot; нужна дообучка | Низкая-средняя | Нет; VLAD частично | Средний | 4096 → 16 КБ | ~5–10 мс (VGG) | Permissive (hloc) | **Нужно дообучение** |
| **AnyLoc-VLAD-DINOv2** | **Сильная (Nardo-Air Top-3@5=100% zero-shot)** | Высокая | Нет; prerotate/aug | Хороший | 49152 → 192 КБ (PCA→4–16 КБ) | Высокая (ViT-G, тяжёлый) | torch.hub, permissive; DINOv2 Apache-2.0 | **Zero-shot** |
| **SALAD** | Хорошая (топ на VPAir/ALTO) | Высокая | Нет; prerotate/aug | Хороший | 8448 → 33 КБ | ~2.4 мс (ViT-B) | **GPL-3.0 (блокер для проприетарного)** | **Zero-shot** (домен город) |
| **MegaLoc** | **Сильная (UAV-VisLoc/DenseUAV/SEALOC топ)** | **Очень высокая (5 датасетов)** | Нет; prerotate/aug | Хороший | 8448 → 33 КБ (PCA→1024=4 КБ) | ~2–4 мс (ViT-B) | **Веса MIT**; код без лицензии+GPL-фрагменты | **Zero-shot** |
| **BoQ-DINOv2** | Средняя (мало аэро-замеров) | Высокая | Нет; prerotate/aug | Средний-хороший | 12288 → 48 КБ | ~3–4 мс | **MIT (чистый)** | **Zero-shot** (домен город) |
| **MixVPR** | Слабее ViT на аэро | Средняя | Нет | Средний | 4096 → 16 КБ | быстрый | permissive | **Zero-shot** |
| **Sample4Geo** | Заточен под cross-view (косой) | Средняя | Аугментация satellite rotation при обучении | Под свой протокол | ~1024 (ConvNeXt) | ~3.6 мс | Код доступен (ICCV23) | **Нужно дообучение на домен** |

**Память индекса при ±5 км, 2000–20000 клеток, ротационная аугментация ×8–12:** например MegaLoc PCA-1024 (fp32=4 КБ/вектор): 20000×12×4 КБ ≈ 960 МБ; AnyLoc raw 49152 (192 КБ): 20000×12×192 КБ ≈ 44 ГБ (нереально без PCA). **Вывод: для любого ViT-метода с большим dim обязателен PCA до 512–1024 перед индексом ±5 км с аугментацией.** При int8/PQ-квантовании FAISS цифры падают ещё в 4–32×.

## Recommendations

**Этап 0 (инфраструктура).** Возьмите `gmberton/VPR-methods-evaluation` — единый враппер для NetVLAD, CosPlace, EigenPlaces, MixVPR, AnyLoc, SALAD, CricaVPR, MegaLoc. Это даёт идентичную нарезку/prerotate/метрики при смене только Encoder — ровно ваш протокол. Метрика: ранг ближайшей к истине клетки + Recall@K (K≤5–10). Критерий приёмки: DRZ_19206 в top-5–10 без деградации 00049.

**Приоритетный порядок zero-shot прогона:**
1. **MegaLoc (первый).** Самый широкий домен обучения (5 датасетов), SOTA-универсал, dim 8448 (PCA→1024), веса MIT. Наивысший шанс поднять DRZ_19206 в top-5. Единственная забота — чистая реализация инференса (см. лицензию).
2. **SALAD (второй).** Та же SALAD-агрегация, что в ядре MegaLoc, но обучен только на GSV-Cities. Хороший диагностический контраст к MegaLoc: покажет, даёт ли мульти-датасетность MegaLoc реальный прирост на вашем домене. **Только для оффлайн-эксперимента** — GPL-3.0 не тащить в продукт.
3. **AnyLoc-VLAD-DINOv2 (третий).** Сильнейшие прямые доказательства на надир↔ортоспутник (Nardo-Air Top-3@5=100% zero-shot, FoundLoc). Обязательно PCA (dim 49152 неприемлем для ±5 км). Если MegaLoc/SALAD не добьют — AnyLoc с доменным «aerial»-словарём кластеров почти наверняка добьёт (до 4× Recall@1).
4. **BoQ-DINOv2 (четвёртый).** Чистая MIT-лицензия; если по качеству близок к MegaLoc — предпочтительный продакшн-вариант по лицензии.
5. **NetVLAD (нижняя планка).** Только как контроль «что даёт настоящий VPR над сырым DINOv2».

**Пороги смены решения:**
- Если MegaLoc zero-shot даёт DRZ_19206 в top-5 и не роняет 00049 → фиксируем MegaLoc (реализация инференса поверх MIT-весов), PCA-1024, prerotate по yaw. Готово.
- Если ни один zero-shot не добивает DRZ_19206 в top-10 → это сигнал, что gap не appearance-only (возможно, масштаб/footprint или качество подложки). Тогда: (а) доменный словарь AnyLoc на ваших клетках; (б) лёгкое дообучение SALAD/BoQ на паре надир↔подложка (GSV-стиль); (в) пересмотр нарезки клеток (zoom/overlap — aero-vloc показал, что это критично для аэро).
- Если лицензия MegaLoc-кода блокирует → BoQ-DINOv2 (MIT) как продакшн-энкодер, MegaLoc-веса только если реализуете чистый инференс.

**Rotation:** внедрите prerotate запроса на −yaw если есть компас/IMU (экономит индекс ×8–12); иначе аугментация индекса ×8 (шаг 45°).

## Каждая рекомендованная модель — шаги интеграции (протокол Encoder)

Общий протокол: класс `Encoder` с методом `encode(gray_or_rgb_tile) -> np.ndarray` (L2-нормированный вектор), ленивая загрузка весов при первом вызове.

**Вход и нормализация (общее):**
- Все три ViT-метода требуют **3-канальный RGB**, НЕ grayscale. Ваш серый кадр надо реплицировать в 3 канала (`gray → stack ×3`) перед подачей. (Отдельное исследование показало, что grayscale-обучение MixVPR почти не теряет качество — 82.4 vs 81.2 R@1, — но готовые веса ждут RGB-нормализацию ImageNet.)
- Нормализация: ImageNet mean/std (`[0.485,0.456,0.406]`/`[0.229,0.224,0.225]`).
- Разрешение: кратно 14 (DINOv2 patch). Стандарт **322×322** для SALAD/MegaLoc; 224×224 или 322×322 для AnyLoc; **320×320** для BoQ.

**MegaLoc:**
- Веса: HF `gberton/MegaLoc` (MIT) или torch.hub `gmberton/MegaLoc` (`get_trained_model`). Инференс: `model(image[B,3,322,322]) → [B,8448]` L2-норм («any size works», но кратно 14).
- Для закрытого продукта: реализовать инференс независимо (DINOv2-B backbone Apache-2.0 + SALAD-агрегатор своей чистой реализацией), т.к. официальный код без лицензии и с GPL-фрагментами.
- PCA: обучить на database-клетках, редуцировать 8448→1024 перед FAISS-индексом.

**SALAD:**
- Веса: torch.hub `serizba/salad` (`dinov2_salad`) или `dino_salad.ckpt`. Вход 322×322 RGB.
- **Только оффлайн-бенчмарк** (GPL-3.0). Не включать в закрытый билд.

**AnyLoc-VLAD-DINOv2:**
- Веса: torch.hub `AnyLoc/DINO` (`get_vlad_model`, domain=…); вход 224×224 RGB → [1,49152].
- Обязательно PCA (fit на database) до 1024–4096. Рассмотреть доменный словарь кластеров на «aerial» (K-means по патч-фичам ваших клеток, K=32; центры сохраняются как тензор (32,1536)).
- Для экономии — заменить ViT-G на ViT-B backbone.

**BoQ-DINOv2:**
- Веса: `torch.hub.load("amaralibey/bag-of-queries","get_trained_boq", backbone_name="dinov2", output_dim=12288)`; вход 320×320 RGB (использовать transform из репо). MIT — чистый продакшн. Вариант `backbone_name="resnet50", output_dim=16384` полностью избегает DINOv2.

## Библиография
- **NetVLAD** — R. Arandjelović, P. Gronat, A. Torii, T. Pajdla, J. Sivic. «NetVLAD: CNN architecture for weakly supervised place recognition». CVPR 2016. Веса — через hloc.
- **AnyLoc** — N. Keetha et al. «AnyLoc: Towards Universal Visual Place Recognition». IEEE RA-L 2023. arXiv:2308.00688. Репо: github.com/AnyLoc/AnyLoc; torch.hub: github.com/AnyLoc/DINO.
- **SALAD** — S. Izquierdo, J. Civera. «Optimal Transport Aggregation for Visual Place Recognition». CVPR 2024. arXiv:2311.15937. Репо: github.com/serizba/salad (**GPL-3.0**).
- **MegaLoc** — G. Berton, C. Masone. «MegaLoc: One Retrieval to Place Them All». CVPR 2025 Workshops. arXiv:2502.17237. Репо: github.com/gmberton/MegaLoc; веса: huggingface.co/gberton/MegaLoc (**MIT**).
- **BoQ** — A. Ali-bey, B. Chaib-draa, P. Giguère. «BoQ: A Place is Worth a Bag of Learnable Queries». CVPR 2024, pp. 17794–17803. arXiv:2405.07364. Репо: github.com/amaralibey/Bag-of-Queries (**MIT**).
- **MixVPR** — A. Ali-bey et al. «MixVPR: Feature Mixing for Visual Place Recognition». WACV 2023.
- **CosPlace** — G. Berton, C. Masone, B. Caputo. «Rethinking Visual Geo-localization for Large-Scale Applications». CVPR 2022.
- **EigenPlaces** — G. Berton et al. «EigenPlaces: Training Viewpoint Robust Models for VPR». ICCV 2023.
- **CricaVPR** — Lu et al. CVPR 2024.
- **Pair-VPR** — S. Hausler, P. Moghadam. IEEE RA-L 2025. Репо: github.com/csiro-robotics/Pair-VPR.
- **EffoVPR** — Tzachor et al. ICLR 2025.
- **DINOv2** — M. Oquab et al. «DINOv2: Learning Robust Visual Features without Supervision». TMLR 2024. github.com/facebookresearch/dinov2 (**Apache-2.0** для стандартных ViT-бэкбонов).
- **DINOv3** — O. Siméoni et al. arXiv:2508.10104 (Aug 2025). github.com/facebookresearch/dinov3 (кастомная commercial-friendly лицензия).
- **Sample4Geo** — F. Deuser, K. Habel, N. Oswald. «Sample4Geo: Hard Negative Sampling For Cross-View Geo-Localisation». ICCV 2023. arXiv:2303.11851. Репо: github.com/Skyy93/Sample4Geo.
- **University-1652** — Z. Zheng, Y. Wei, Y. Yang. ACM MM 2020. arXiv:2002.12186.
- **FoundLoc** — «FoundLoc: Vision-based Onboard Aerial Localization in the Wild». arXiv:2310.16299 (источник Nardo-Air замеров).
- **Aerial VPR survey / aero-vloc** — I. Moskalenko, A. Kornilova, G. Ferrer. «Visual place recognition for aerial imagery: A survey». arXiv:2406.00885. Репо: github.com/prime-slam/aero-vloc.
- **VPR-methods-evaluation** — G. Berton. Единый враппер 10+ VPR моделей: github.com/gmberton/VPR-methods-evaluation.

## Caveats
- **Оценки латентности** MegaLoc/SALAD/BoQ на RTX 4090 — это экстраполяция из таблицы DINOv2-конфигураций (ViT-B ~2.4 мс на batch на железе авторов) и оценок автора Netryx; точные цифры на 4090 надо мерить на стенде.
- **Память индекса** посчитана для fp32; при int8/PQ-квантовании FAISS цифры падают ещё в 4–32×.
- **Аэро-доказательства для BoQ** тоньше, чем для AnyLoc/MegaLoc — прямых надир↔спутник замеров в литературе мало; риск, что городской ground-домен обучения хуже переносится.
- **Лицензии весов** SALAD/BoQ/MegaLoc отдельно не прописаны разработчиками (кроме MegaLoc-весов MIT на HF) — юридически это «наследование лицензии репозитория»; для коммерции нужна проверка юристом, особенно GPL-фрагменты в коде MegaLoc.
- **DINOv2-лицензия**: сегодня стандартные ViT-бэкбоны Apache-2.0 (код + веса), но исходный релиз 2023 был CC-BY-NC; убедитесь, что тянете актуальные Apache-веса. Специализированные derivative-модели в репо DINOv2 (Cell-DINO, XRay-DINO) остаются non-commercial — это не наш случай.
- **Число датасетов MegaLoc**: в статье названы пять (SF-XL, GSV-Cities, MSLS, MegaScenes, ScanNet); встречающаяся формулировка «6 датасетов» (карточка Netryx) — вторичный источник, менее надёжный, чем сама статья.
- **Nardo-Air R (rotated)** показал деградацию AnyLoc при ортогональных ракурсах — надир↔надир с prerotate это не затрагивает, но при неизвестном yaw без аугментации риск реален.
- **Специализированные drone→satellite** (Sample4Geo/FSRA/DAC) не отброшены полностью: если после zero-shot прогона потребуется дообучение и появятся пары надир↔подложка, Sample4Geo (contrastive + hard-negative) — хорошая база для fine-tune, но это уже не zero-shot путь.