# Diffusion Zebrafish

Фреймворк генерации статических флуоресцентных изображений и бинарных масок.
Диффузионная модель создаёт изображение из случайного шума. Обученный U-Net++
сегментирует полученное изображение. Модели обучаются независимо.

## 1. Состав проекта

```text
diffusion zebrafish/
  README.md
  pipeline.py                     единая командная строка
  pyproject.toml                   установка пакета и console entry point
  requirements.txt
  notebooks/
    Zebrafish_Framework.ipynb      основной ноутбук, локальный Python и Colab
  src/zebrafish_diffusion/
    api.py                        загрузка моделей, generate(seed), segment_file
    cli.py                        подготовка данных, обучение, генерация
    data.py                       адаптеры PNG/TIFF/NPY/NPZ/NAOMi/SimCalc
    unetpp.py                     U-Net++, обучение и применение сегментатора
    diffusion.py                  обучение v-prediction diffusion, DDIM, EMA
    diffusion_unet.py             архитектура диффузионной U-Net
    diffusion_sampling.py         cosine schedule и исходные sampling utilities
    zenodo.py                     загрузка демонстрационного highSNR TIFF
  configs/
    generation.json
    unet_training.json
    diffusion_training.json
  weights/
    manifest.json                 происхождение и SHA-256 включённых весов
    unetpp/best.pt                 готовый сегментатор
    diffusion/step_007500.pt       готовая диффузия, шаг 7500
  data/examples/                  примеры пар NAOMi для просмотра формата
  outputs/                        результаты запусков
  tools/
    import_weights.py             импорт собственных checkpoint
    build_colab_zip.py            сборка ZIP для Colab
  tests/test_data.py               проверки адаптеров данных
  dist/                           собранные архивы
```

В ноутбуке и CLI по умолчанию выбирается файл `step_*.pt` с наибольшим номером
из `weights/diffusion`. В текущем локальном комплекте это `step_007500.pt`.
Имена файлов не меняют сохранённый шаг: он читается из поля `step` внутри checkpoint.

## 2. Установка

Python 3.10 или новее. Изображения — двумерные, один канал, размер не менее 16x16.
В пределах одного обучающего набора все изображения имеют одинаковые H и W.
Обученный профиль из комплекта имеет размер 400x485 (H,W).

### Windows / PowerShell

Открыть терминал в папке проекта:

```powershell
cd "$env:USERPROFILE\Desktop\diffusion zebrafish"
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
```

Если нужная CUDA-сборка PyTorch уже установлена, использовать её среду Python:

```powershell
python -m pip install numpy pillow tifffile
python -m pip install -e . --no-deps
```

### Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Все следующие команды выполняются из корня проекта. В активированной среде
`python pipeline.py ...` и `zebrafish-diffusion ...` эквивалентны.

## 3. Запуск из готовых весов

В комплекте находятся:

| Модель | Файл | Формат checkpoint |
|---|---|---|
| U-Net++ | `weights/unetpp/best.pt` | `naomi-unetpp-fullframe-v1` |
| Диффузия, 7500 шагов | `weights/diffusion/step_007500.pt` | `neural-naomi-static-vdiffusion-v1` |

Генерация одного изображения и маски на CPU:

```bash
python pipeline.py generate --unet weights/unetpp/best.pt --seed 20331027 --steps 100 --device cpu --output outputs/demo
```

С явным выбором checkpoint:

```bash
python pipeline.py generate --diffusion weights/diffusion/step_007500.pt --unet weights/unetpp/best.pt --seed 20331027 --steps 100 --device auto --output outputs/demo7500
```

Для CUDA явно указать `--device cuda`. `auto` выбирает CUDA при наличии, иначе CPU.
На CPU вычисления выполняются в float32. На CUDA по умолчанию включён FP16 AMP;
`--no-amp` отключает его. `--threads 4` задаёт число CPU-потоков.

Генерация нескольких пар:

```bash
python pipeline.py generate --count 10 --seed 42 --steps 100 --device auto
```

Seed последовательно принимает значения 42, 43, ..., 51. По умолчанию эта команда
выбирает checkpoint с наибольшим номером в `weights/diffusion`. Для явного выбора
указать `--diffusion`.

С JSON-конфигурацией:

```bash
python pipeline.py generate --config configs/generation.json --seed 123 --device cpu
```

Аргументы командной строки переопределяют значения JSON. Значение `diffusion_checkpoint: null`
включает автоматический выбор; вместо null можно указать конкретный путь.

### Сегментация существующего изображения

```bash
python pipeline.py segment --input data/examples/naomi/images/naomi_0000_seed20263000.tif --unet weights/unetpp/best.pt --device cpu --output outputs/segmentation
```

Имя примера уточняется в `data/examples/README.md`. Вместо примера можно указать
свой одиночный PNG/TIFF. Диффузионная модель для этой команды не загружается.

## 4. Основной ноутбук

Файл: `notebooks/Zebrafish_Framework.ipynb`.

В ноутбуке отдельные переключатели для следующих операций:

1. Подключение проекта и установка зависимостей.
2. Выбор локальных весов или импорт из Google Drive.
3. Генерация по seed с отображением изображения, маски, вероятностей и наложения.
4. Сегментация пользовательского изображения.
5. Подготовка пар и обучение U-Net++.
6. Подготовка изображений и обучение диффузии.

Генерация включена по умолчанию. Секции обучения выключены, пока пользователь
не установит соответствующий переключатель `True`.

### Colab

Сборка архива с кодом:

```bash
python tools/build_colab_zip.py
```

Сборка архива с кодом и находящимися в папке весами:

```bash
python tools/build_colab_zip.py --include-weights
```

Полученный ZIP находится в `dist`. Загрузить его в корень «Моего диска».
Открыть ноутбук в Colab. В первой ячейке указать `COLAB_ZIP` и выполнить ячейку.
Архив распаковывается в `/content/diffusion_zebrafish`. Обучение удобно выполнять
на GPU; для применения готовых моделей доступна обычная CPU-среда.

Во второй ячейке можно импортировать существующие веса:

```text
DRIVE_UNET = /content/drive/MyDrive/best.pt
DRIVE_DIFFUSION = /content/drive/MyDrive/neural_naomi_runs/run_.../last.pt
```

При пустом `DRIVE_DIFFUSION` выбирается самый новый `run_*/last.pt` в
`MyDrive/neural_naomi_runs`. Сохранённый шаг печатается и используется в имени
импортируемого файла: например, `step_007500.pt`.

В Colab результаты сохраняются в `MyDrive/diffusion_zebrafish_outputs`,
временные подготовленные данные — в `/content`. Локально результаты сохраняются
в `outputs`. Для повторного обучения после сброса VM входной датасет подключается
снова, затем указывается путь к сохранённому checkpoint.

## 5. Данные для U-Net++

U-Net++ обучается на парах «изображение — бинарная маска». Маска содержит 0 для фона
и 1 либо 255 для foreground. При импорте instance labels используются как
`foreground = labels > 0`: отдельные instance ID объединяются в бинарную маску.

### Свой набор изображений и масок

```text
my_segmentation/
  images/
    scene_001.tif
    scene_002.png
  masks/
    scene_001.png
    scene_002.png
```

Изображение и маска должны иметь одинаковое имя без расширения и одинаковый размер.
Маска каждого обучающего примера содержит foreground и фон.

Подготовка:

```bash
python pipeline.py prepare-unet --source my_segmentation --output data/prepared_masks
```

Результат:

```text
data/prepared_masks/
  images/*.tif
  masks/*.png
  pairs.csv
  source.json
```

`pairs.csv` имеет колонки `name,image,mask,group`. Пути задаются относительно
подготовленной папки. `group` объединяет зависимые примеры: один seed, исходную
сцену, исходное изображение или другой выбранный идентификатор. Одна группа целиком
попадает в train, validation или test. Требуется не менее пяти групп.
У подготовленных обычных пар исходный `group` равен имени файла; его можно изменить
до первого обучения. Для уже подготовленного набора `pairs.csv` сохраняется при импорте.

### Данные NAOMi

Исходная NAOMi: https://bitbucket.org/adamshch/naomi_sim/src/master/

NAOMi формирует сцену и её флуоресцентное изображение. Маски экспортируются из
пространственных профилей сгенерированных клеток. Для обучения требуются
согласованные пары: одно изображение и маска в той же системе пиксельных координат.

Существующий формат нашего экспорта поддерживается напрямую:

```text
ZEBRAFISH/
  images/*.png
  masks_png_400x485/
    correspondence.csv
    semantic/*.png
```

```bash
python pipeline.py prepare-unet --source ZEBRAFISH --output data/prepared_naomi
```

В `correspondence.csv` используются поля `image,semantic_mask,seed`.
Изображения и маски связываются по этому CSV. В подготовленный набор включаются
существующие пары.

Для другого экспортера NAOMi достаточно сохранить результаты в стандартных
`images/` и `masks/`, затем вызвать `prepare-unet`.

Исходный набор обученного сегментатора — экспорт `ZEBRAFISH`: 348 полных изображений
400x485 с бинарными масками. Формирование обучающих/validation/test частей выполнялось
по seed; исходное разбиение имело размеры 244/52/52. В комплекте находятся примеры
пар из этого экспорта. Сам симулятор не является зависимостью запуска моделей.

### Данные SimCalc

Поддерживается NPZ следующего состава:

| Поле | Размер | Назначение |
|---|---|---|
| `synthetic_images` | N,H,W | изображения |
| `synthetic_labels` | N,H,W | instance labels, 0 — фон |
| `synthetic_seeds` | N | seed для группировки пар, необязательно |

```bash
python pipeline.py prepare-unet --source hybrid_highSNR_1000.npz --output data/prepared_simcalc
```

Альтернативные названия полей:

```bash
python pipeline.py prepare-unet --source own.npz --image-key frames --mask-key annotation --output data/prepared_masks
```

Для генерации новых обучающих данных в NAOMi/SimCalc нужно сохранять пары с общей
геометрией. Изменения положения, масштаба и поворота применяются и к изображению,
и к маске. После экспорта набор импортируется одной из команд выше.

## 6. Обучение своего U-Net++

```bash
python pipeline.py train-unet --data data/prepared_masks --output outputs/unet_runs --epochs 150 --batch-size 1 --base-channels 32 --accumulation 4 --device cuda
```

Либо:

```bash
python pipeline.py train-unet --config configs/unet_training.json --output outputs/unet_runs
```

Основные настройки:

| Аргумент | Значение | Назначение |
|---|---|---|
| `--epochs` | 150 | максимальное число эпох |
| `--base-channels` | 32 | ширина первого уровня U-Net++ |
| `--batch-size` | 1 в конфигурации | число изображений за forward |
| `--accumulation` | 4 в конфигурации | накопление градиента |
| `--lr` | 0.0003 | learning rate AdamW |
| `--patience` | 25 | early stopping по validation Dice |
| `--min-delta` | 0.0001 | минимальное улучшение Dice |
| `--val-fraction`, `--test-fraction` | 0.15, 0.15 | доли групп |
| `--seed` | 42 | seed разбиения и обучения |
| `--image-size H W` | автоматически из данных | явная проверка геометрии |

Архитектура: вложенные skip-связи U-Net++, пять пространственных уровней, GroupNorm,
четыре выхода deep supervision. На каждом выходе используется
`0.5 * BCE + 0.5 * Dice loss`. Для предсказания усредняются четыре sigmoid-вероятности.
AdamW обновляет веса; ReduceLROnPlateau уменьшает learning rate по validation Dice.

Нормализация сегментатора выполняется отдельно для каждого изображения по p0.5/p99.5
с приведением к [0,1]. Размер изображения сохраняется; внутри сети справа и снизу
добавляется padding до кратности 16, затем удаляется.

При обучении используются совместные отражения изображения и маски, а также
изменение яркости, контраста, gamma и слабый шум изображения.

`best.pt` выбирается по validation Dice при пороге 0.5. В конце порог выбирается
по validation лучшего checkpoint и записывается в `best.pt`. Затем считаются test-метрики.
Early stopping срабатывает после `patience` эпох без улучшения более `min_delta`.

Продолжение:

```bash
python pipeline.py train-unet --data data/prepared_masks --resume outputs/unet_runs/RUN/last.pt --base-channels 32 --seed 42 --device cuda
```

Для продолжения рядом с `last.pt` должен находиться `best.pt` того же запуска.
Набор данных, маски, seed разбиения и архитектура должны совпадать с checkpoint.

## 7. Данные для диффузии

Диффузии нужны только изображения. Поддерживаются:

- папка отдельных PNG/TIFF, включая вложенные папки;
- одиночный TIFF-стек N,H,W;
- NPY-массив N,H,W;
- NPZ с `images` или `synthetic_images`, либо с указанным `--image-key`.

Примеры:

```bash
python pipeline.py prepare-diffusion --source my_images --output data/prepared_images
python pipeline.py prepare-diffusion --source recording.tif --split-method temporal --output data/prepared_recording
python pipeline.py prepare-diffusion --source own.npz --image-key frames --output data/prepared_npz
```

Каждая команда создаёт новую папку. В ней находятся `images.npy`, `split.npy`,
`groups.npy`, `names.npy`, `dataset.json`. Изображения сохраняют исходный числовой тип;
затем читаются через memory map. Нормализация вычисляется по train и хранится в JSON.

При `--split-method auto` для папки и NPZ с профилями SimCalc используется разбиение
по группам; для TIFF/NPY/обычного NPZ — временные блоки. Для временного разбиения
используются 10 блоков, validation — блок 4, test — блок 8, промежутки `--gap 32`.

Пользовательская группировка:

```text
name,group,split
fish1_001.png,fish1,train
fish1_002.png,fish1,train
fish2_001.png,fish2,val
fish3_001.png,fish3,test
```

Колонка `split` необязательна. Допустимые значения: `train,val,test,exclude`.
Для массива используются имена `frame_000000`, `frame_000001`, ...;
для папки — относительные пути к изображениям. CSV должен перечислять все изображения.

```bash
python pipeline.py prepare-diffusion --source my_images --groups-csv groups.csv --output data/prepared_images
```

Проверяются совпадение размеров и типов, конечность значений, состав split,
пересечение групп и точные дубликаты между частями. SHA-256 пикселей и разбиения
сохраняются и проверяются при последующей загрузке.

### Встроенный источник Zenodo

Источник исходного обучения: https://zenodo.org/records/6293696

Использован `01_ZebrafishMul_GCaMP6s_485x400x9800_highSNR.tif`: 9800 кадров,
400x485, GCaMP6s. В записи также находится lowSNR-вариант. Для обучения диффузии
выбирается highSNR. Маски для этого источника не требуются.

Если `--dataset` не задан, trainer скачивает highSNR TIFF, проверяет MD5 и читает
кадры лениво. В Colab кэш находится в `/content/zenodo_zebrafish`; локально —
в `data/cache/zenodo`. Исходное разбиение: 7712 train, 916 validation, 916 test,
256 кадров в исключённых промежутках.

## 8. Обучение своей диффузии

После подготовки изображений:

```bash
python pipeline.py train-diffusion --dataset data/prepared_images --runs outputs/diffusion_runs --steps 20000 --channels 48 --batch-size 1 --accumulation 4 --device cuda
```

Через конфигурацию:

```bash
python pipeline.py train-diffusion --config configs/diffusion_training.json --runs outputs/diffusion_runs
```

На встроенном Zenodo:

```bash
python pipeline.py train-diffusion --runs outputs/diffusion_runs --steps 20000 --device cuda
```

Архитектура: grayscale residual U-Net, ширины C/2C/3C/4C, GroupNorm, SiLU,
timestep embedding и attention на разрешении 1/8. По умолчанию C=48.
Исходный размер сохраняется, padding до кратности 8 удаляется на выходе.

Изображения приводятся к [-1,1] по общей шкале p1/p99.8 train. Выбирается timestep t,
добавляется Gaussian noise epsilon, сеть предсказывает velocity v:

```text
xt = sqrt(alpha_bar[t]) * x0 + sqrt(1-alpha_bar[t]) * epsilon
v  = sqrt(alpha_bar[t]) * epsilon - sqrt(1-alpha_bar[t]) * x0
loss = MSE(predicted_v, v)
```

Используются cosine schedule, 1000 timesteps и нулевой terminal SNR.
Оптимизатор — AdamW, lr=0.0002, weight decay=0.0001; clipping градиента — 1.
Есть warmup, cosine LR decay, gradient accumulation, FP16 AMP на CUDA,
gradient checkpointing и EMA. EMA decay ограничен 0.999 и имеет начальный прогрев.

`--steps` задаёт итоговый номер optimizer-шага. `--lr-horizon` задаёт горизонт
LR-расписания. По умолчанию оба равны 20000. После горизонта сохраняется минимальный LR.

Каждые `--save-every 500` шагов выполняется fixed-noise validation и сохраняется
`last.pt`. При уменьшении validation v-MSE сохраняется `best.pt`. Каждые
`--snapshot-every 5000` сохраняется отдельный checkpoint. Каждые
`--preview-every 1000` создаются генерации с фиксированными seed.
Автоматический early stopping диффузии не включён: остановка происходит по `--steps`.

Продолжение:

```bash
python pipeline.py train-diffusion --dataset data/prepared_images --resume outputs/diffusion_runs/RUN/last.pt --steps 30000 --device cuda
```

Восстанавливаются модель, EMA, optimizer, scaler, RNG и позиция sampler. Архитектура,
seed и исходный горизонт LR читаются из checkpoint. Подготовленный датасет должен
иметь тот же fingerprint. При изменении набора данных начинается новый запуск.

Сегментатор для обучения диффузии не нужен и в training loop не загружается.

## 9. Как выполняется генерация

1. Из checkpoint читаются архитектура, размер, нормализация, timesteps и EMA-веса.
2. Seed инициализирует генератор случайных чисел.
3. Создаётся один Gaussian noise tensor размера 1x1xHxW.
4. DDIM выполняет `steps` последовательных проходов U-Net.
5. Итоговое изображение переводится из [-1,1] в исходную числовую шкалу и dtype.
6. U-Net++ получает изображение со своей сохранённой нормализацией.
7. Средняя sigmoid-вероятность четырёх голов сравнивается с сохранённым порогом.

`steps=100` означает 100 проходов для одного изображения. Уменьшение этого параметра
снижает число вычислений. Для сравнения checkpoint используются одинаковые seed,
steps, устройство и параметры. Веса моделей во время генерации не обновляются.

## 10. Python API

```python
from zebrafish_diffusion.api import Pipeline

pipeline = Pipeline(
    diffusion_checkpoint='weights/diffusion/step_007500.pt',
    unet_checkpoint='weights/unetpp/best.pt',
    device='cpu',
)
result = pipeline.generate(seed=42, steps=100)
image = result['image']
mask = result['mask']
probability = result['probability']
paths = pipeline.save(result, 'outputs/example')
```

Модели загружаются один раз в конструкторе. Для следующего изображения вызывается
`pipeline.generate` с новым seed. Входы и выходы — NumPy-массивы HxW.
`mask` имеет тип bool; `probability` — float32 в [0,1].

## 11. Файлы результатов

Для каждого seed сохраняются:

| Суффикс | Содержимое |
|---|---|
| `_image.tif` | изображение в числовом типе и шкале обучающего датасета |
| `_mask.png` | бинарная маска uint8, 0/255 |
| `_probability.png` | вероятность foreground uint16; делить на 65535 |
| `_overlay.png` | наложение маски на изображение |
| `_preview.png` | изображение, маска, вероятность и наложение в одной строке |
| `_metadata.json` | seed, checkpoint, шаг, нормализации, порог, DDIM и пути |

Маски содержат один класс foreground. Instance ID и временные сигналы не создаются.
`--tta` включает усреднение предсказаний по четырём отражениям изображения.
`--threshold` задаёт пользовательский порог вместо значения checkpoint.

## 12. Веса и публикация проекта

Включённые веса скопированы без изменения содержимого. Их исходные имена,
размеры и SHA-256 перечислены в `weights/manifest.json`.

Импорт новых весов:

```bash
python tools/import_weights.py --unet /path/to/best.pt --diffusion /path/to/last.pt
```

Скрипт проверяет format и сохраняет диффузию под именем `step_NNNNNN.pt`.
При замене существующего файла создаётся копия `.previous`.

В `.gitignore` исключены checkpoint, outputs, кэш данных и ZIP-архивы.
Для публикации исходников в GitHub достаточно корня проекта без этих файлов.
Веса можно разместить отдельными файлами релиза и указать ссылки в
`weights/README.md`. Исходные пользовательские датасеты в репозиторий не добавляются.

Основные первоисточники:

- NAOMi: https://doi.org/10.1016/j.jneumeth.2021.109173
- U-Net++: https://arxiv.org/abs/1807.10165
- DDPM: https://arxiv.org/abs/2006.11239
- DDIM: https://arxiv.org/abs/2010.02502
- Исходные изображения DeepCAD-RT: https://zenodo.org/records/6293696

## 13. Проверки структуры и данных

```bash
python -m unittest discover -s tests -v
python pipeline.py --help
```

Проверки в `tests` используют небольшие временные массивы и проверяют импорт,
разбиение, бинаризацию instance labels и контроль целостности. Модели при этом
не обучаются и не выполняют генерацию.

## 14. Сообщения об ошибках

| Сообщение | Действие |
|---|---|
| checkpoint not found | проверить путь к файлу; добавить checkpoint в weights/diffusion или указать --diffusion |
| incompatible format | выбрать checkpoint соответствующей модели из таблицы раздела 3 |
| CUDA unavailable | выбрать `--device cpu` либо GPU-среду |
| Need at least 5 independent groups | дополнить набор или задать корректные группы |
| identical image in different splits | объединить дубликаты в одну группу |
| inconsistent shape/dtype | привести исходный набор к общей геометрии и dtype до импорта |
| dataset changed / fingerprint changed | заново подготовить данные и начать новый training run |
| out of memory | уменьшить batch-size; для нового запуска уменьшить channels/base-channels |
