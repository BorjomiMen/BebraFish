# SimCalc + diffusion addon · zebrafish highSNR

Генерация статических изображений 400×485 и instance-масок. Только highSNR.

## Архитектура

1. **SimCalc:** подготовленный растровый шаблон ткани, глаз и тёмных разделений из одной записи Zenodo 6293696 слегка деформируется (3 px); новые нейроны создаются ядрами SimCalc. Детектор добавляет photon/read noise. Маски относятся к новым нейронам.
2. **Diffusion addon:** epsilon-prediction U-Net обучается только на реальных highSNR кадрах. DDIM/SDEdit обрабатывает синтетику; ограниченная знаковая поправка ослабляется около масок. Маски копируются без изменений.

## Вход → выход

| Шаг | Вход | Выход |
|---|---|---|
| SimCalc | `configs/hybrid_highSNR_template_v1.npz`, параметры и seed | `reports/hybrid_highSNR_dataset_1000_mild/hybrid_highSNR_1000.npz`: 1000 `uint16` изображений, `uint16` instance-маски, тканевые маски, seed и метаданные |
| Обучение | NPZ SimCalc + полный реальный highSNR TIFF Zenodo (скачивается автоматически, ~3,9 ГБ) | `reports/additive_simcalc_training/run_*/last.pt`, метрики и примеры |
| Добавка | NPZ SimCalc + обученный checkpoint | `reports/additive_simcalc_refined_1000/`: NPZ-порции по 25 кадров с исходником, `refined_images`, точной `additive_residual_adu` (`int32`), исходными масками; `qa.json`, `manifest.json`, `first_comparison.png` |

## Запуск

Python ≥3.10. Установить `requirements.txt` в интерпретатор PyCharm. Затем открыть и нажать **Run** по порядку:

1. `generate_hybrid_high_dataset.py` — создаёт 1000 синтетических кадров; если NPZ уже есть, проверяет его.
2. `train_additive_simcalc.py` — скачивает реальный TIFF и обучает prior. Настройки в начале файла; повторный Run продолжает `last.pt`.
3. `refine_additive_simcalc.py` — применяет добавку ко всем кадрам (`LIMIT=0`). Для пробы поставить `LIMIT=25` и отдельную пустую `OUTPUT_DIR`.

Весов в репозитории нет: шаги 2–3 требуют собственного обучения. Полное обучение и обработка 1000 кадров требуют значительного времени; GPU предпочтителен. Исходник и refined-варианты нужно держать в одной части train/validation/test. Флаг `review_required` требует визуальной проверки: сохранность масок и улучшение реализма не гарантированы. Все головы происходят из шаблона одной реальной записи.

Код добавки основан на [AdditiveZebraFish](https://github.com/BorjomiMen/AdditiveZebraFish). Происхождение и лицензии: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
