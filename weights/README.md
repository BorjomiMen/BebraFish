# Checkpoints

Локальный комплект:

- `unetpp/best.pt`: U-Net++, format `naomi-unetpp-fullframe-v1`, epoch 145.
- `diffusion/step_003000.pt`: diffusion, format `neural-naomi-static-vdiffusion-v1`, step 3000.

Дополнительный checkpoint пользователя:

- `diffusion/step_007500.pt`: скопировать сюда свой `last.pt`, сохранённый на шаге 7500.

Основной ноутбук выбирает 7500 при наличии, иначе 3000. Через CLI путь задаётся
параметром `--diffusion`. Нормализации, архитектура и порог сегментации читаются
из checkpoint. `manifest.json` содержит размеры и SHA-256 включённых файлов.

Для импорта с проверкой формата использовать `tools/import_weights.py`.
Крупные checkpoint исключены из Git. Для публичного репозитория разместить их
отдельными файлами релиза и добавить ссылки на скачивание в этот документ.
