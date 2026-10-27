"""Rebuild the main notebook without executing computational cells."""
from pathlib import Path
import json
import uuid

ROOT=Path(__file__).resolve().parents[1]

def md(text):
    return dict(cell_type='markdown',id=uuid.uuid4().hex[:8],metadata={},source=text.splitlines(keepends=True))
def code(text):
    return dict(cell_type='code',id=uuid.uuid4().hex[:8],metadata={},source=text.splitlines(keepends=True),
                execution_count=None,outputs=[])

cells=[md('''# Diffusion Zebrafish

Статические изображения и бинарные маски. Готовые веса, генерация по seed,
сегментация своего изображения и независимое обучение двух моделей.
Описание форматов и параметров: `README.md` в корне проекта.
'''),md('''## 1. Подключить проект

В Colab загрузить ZIP из `dist` на Drive. Архив с кодом имеет имя
`diffusion_zebrafish_code.zip`, архив с весами — `diffusion_zebrafish_full.zip`.
Указать нужный путь в COLAB_ZIP. Локально открыть ноутбук из папки проекта.
'''),code('''from pathlib import Path
import importlib.util
import sys, subprocess, zipfile

IN_COLAB = importlib.util.find_spec('google.colab') is not None if importlib.util.find_spec('google') else False
COLAB_ZIP = '/content/drive/MyDrive/diffusion_zebrafish_code.zip'
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    ROOT = Path('/content/diffusion_zebrafish')
    if not (ROOT / 'pipeline.py').is_file():
        ROOT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(COLAB_ZIP) as archive:
            for item in archive.infolist():
                if not (ROOT / item.filename).resolve().is_relative_to(ROOT.resolve()):
                    raise ValueError('Archive member outside project directory')
            archive.extractall(ROOT)
    OUTPUT_ROOT = Path('/content/drive/MyDrive/diffusion_zebrafish_outputs')
    DATA_WORK = Path('/content/zebrafish_prepared')
else:
    candidates = [Path.cwd(), *Path.cwd().parents]
    ROOT = next((p for p in candidates if (p / 'pipeline.py').is_file() and (p / 'src').is_dir()), None)
    if ROOT is None:
        raise FileNotFoundError('Open this notebook inside the diffusion zebrafish project folder')
    OUTPUT_ROOT = ROOT / 'outputs'
    DATA_WORK = ROOT / 'data'

packages = {'torch': 'torch>=2.3', 'numpy': 'numpy>=1.24', 'PIL': 'pillow>=9', 'tifffile': 'tifffile'}
missing = [value for module,value in packages.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.run([sys.executable,'-m','pip','install',*missing],check=True)
sys.path.insert(0, str(ROOT / 'src'))
OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
DATA_WORK.mkdir(parents=True,exist_ok=True)
print('Project:', ROOT)
print('Outputs:', OUTPUT_ROOT)
'''),md('''## 2. Выбрать режимы и параметры

Обучение включается отдельно. DEVICE управляет применением моделей;
TRAIN_DEVICE — обучением. Для Colab с GPU выбрать `cuda`.
'''),code('''RUN_GENERATION = True
RUN_SEGMENTATION = False
RUN_UNET_TRAINING = False
RUN_DIFFUSION_TRAINING = False

DEVICE = 'auto'            # 'cpu' / 'cuda' / 'auto'
TRAIN_DEVICE = 'cuda'
SEED = 20331027
SAMPLE_STEPS = 100
COUNT = 1
INPUT_IMAGE = ''           # одиночный PNG/TIFF для RUN_SEGMENTATION

UNET_WEIGHTS = ROOT / 'weights/unetpp/best.pt'
DIFFUSION_WEIGHTS = ROOT / 'weights/diffusion/step_007500.pt'
if not DIFFUSION_WEIGHTS.is_file():
    DIFFUSION_WEIGHTS = ROOT / 'weights/diffusion/step_003000.pt'

IMPORT_DRIVE_WEIGHTS = IN_COLAB and (RUN_GENERATION or RUN_SEGMENTATION) and (
    not UNET_WEIGHTS.is_file() or (RUN_GENERATION and not DIFFUSION_WEIGHTS.is_file()))
DRIVE_UNET = '/content/drive/MyDrive/best.pt'
DRIVE_DIFFUSION = ''       # точный last.pt; пусто = последний neural_naomi_runs/run_*/last.pt

UNET_SOURCE = ''           # исходная папка images/masks, NAOMi или NPZ SimCalc
UNET_PREPARED = DATA_WORK / 'prepared_masks'
UNET_EPOCHS = 150
UNET_RESUME = ''
DIFFUSION_SOURCE = 'zenodo' # либо свой каталог PNG/TIFF, стек TIFF, NPY или NPZ
DIFFUSION_PREPARED = DATA_WORK / 'prepared_images'
DIFFUSION_STEPS = 20000
DIFFUSION_CHANNELS = 48
DIFFUSION_RESUME = ''
'''),md('''## 3. Загрузить готовые веса

Веса локального полного комплекта используются напрямую. Для ZIP только с кодом
в Colab импортируются существующие файлы с Drive. Параметры моделей и сохранённый
шаг берутся из checkpoint. Импорт не запускает обучение.
'''),code('''import torch, shutil, json

if IMPORT_DRIVE_WEIGHTS:
    if not Path(DRIVE_UNET).is_file():
        raise FileNotFoundError(DRIVE_UNET)
    if not DRIVE_DIFFUSION and RUN_GENERATION:
        candidates = list(Path('/content/drive/MyDrive/neural_naomi_runs').glob('run_*/last.pt'))
        if not candidates:
            raise FileNotFoundError('Set DRIVE_DIFFUSION to your last.pt')
        DRIVE_DIFFUSION = str(max(candidates,key=lambda p:p.stat().st_mtime))
    if RUN_GENERATION:
        subprocess.run([sys.executable,str(ROOT/'tools/import_weights.py'),
                        '--unet',DRIVE_UNET,'--diffusion',DRIVE_DIFFUSION,'--root',str(ROOT)],check=True)
        imported = json.loads((ROOT/'weights/imported.json').read_text(encoding='utf-8'))
        UNET_WEIGHTS = ROOT / next(r['path'] for r in imported if r['kind']=='unetpp')
        DIFFUSION_WEIGHTS = ROOT / next(r['path'] for r in imported if r['kind']=='diffusion')
    else:
        UNET_WEIGHTS.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(DRIVE_UNET,UNET_WEIGHTS)

print('U-Net++:',UNET_WEIGHTS)
print('Diffusion:',DIFFUSION_WEIGHTS)
'''),md('''## 4. Сгенерировать изображение и маску

Изменить SEED для нового начального шума. Pipeline загружает модели один раз.
Результат содержит NumPy-массивы image, mask и probability. На CPU 100 DDIM steps
выполняются последовательно; прогресс выводится каждые 10 шагов.
'''),code('''from IPython.display import display, Image as DisplayImage

if RUN_GENERATION:
    from zebrafish_diffusion.api import Pipeline
    pipeline = Pipeline(DIFFUSION_WEIGHTS,UNET_WEIGHTS,device=DEVICE)
    print('Diffusion step:',pipeline.step,'; shape:',pipeline.shape,'; device:',pipeline.device)
    for index in range(COUNT):
        result = pipeline.generate(seed=SEED+index,steps=SAMPLE_STEPS)
        paths = pipeline.save(result,OUTPUT_ROOT/'generated')
        image, mask = result['image'],result['mask']
        display(DisplayImage(filename=paths['preview'],width=1500))
        print(paths)
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
'''),md('''## 5. Создать маску своего изображения

Указать INPUT_IMAGE и RUN_SEGMENTATION=True. Загружается только U-Net++.
На выходе — маска, вероятности, наложение и metadata.
'''),code('''if RUN_SEGMENTATION:
    from zebrafish_diffusion.api import segment_file
    if not INPUT_IMAGE:
        raise ValueError('Set INPUT_IMAGE')
    paths = segment_file(INPUT_IMAGE,UNET_WEIGHTS,OUTPUT_ROOT/'segmentation',device=DEVICE)
    display(DisplayImage(filename=paths['preview'],width=1500))
    print(paths)
'''),md('''## 6. Подготовить данные и обучить U-Net++

Включить RUN_UNET_TRAINING и задать UNET_SOURCE. Поддерживаются NAOMi, SimCalc NPZ
и собственные пары images/masks. Подготовленный набор создаётся один раз.
Для обновлённого исходного набора указать новую папку UNET_PREPARED.
'''),code('''if RUN_UNET_TRAINING:
    from zebrafish_diffusion.data import prepare_segmentation
    if not UNET_SOURCE:
        raise ValueError('Set UNET_SOURCE')
    if not (UNET_PREPARED/'pairs.csv').is_file():
        prepare_segmentation(UNET_SOURCE,UNET_PREPARED)
    command = [sys.executable,str(ROOT/'pipeline.py'),'train-unet',
               '--data',str(UNET_PREPARED),'--output',str(OUTPUT_ROOT/'unet_runs'),
               '--epochs',str(UNET_EPOCHS),'--base-channels','32',
               '--batch-size','1','--accumulation','4','--device',TRAIN_DEVICE]
    if UNET_RESUME:
        command += ['--resume',UNET_RESUME]
    subprocess.run(command,check=True)
'''),md('''## 7. Подготовить изображения и обучить диффузию

Включить RUN_DIFFUSION_TRAINING. DIFFUSION_SOURCE='zenodo' загружает встроенный
highSNR TIFF во временный кэш. Для своих данных указать источник и подготовить
новую DIFFUSION_PREPARED. Маски для этого этапа не используются.
'''),code('''if RUN_DIFFUSION_TRAINING:
    from zebrafish_diffusion.data import prepare_diffusion
    command = [sys.executable,str(ROOT/'pipeline.py'),'train-diffusion',
               '--runs',str(OUTPUT_ROOT/'diffusion_runs'),'--steps',str(DIFFUSION_STEPS),
               '--channels',str(DIFFUSION_CHANNELS),'--batch-size','1','--accumulation','4',
               '--device',TRAIN_DEVICE]
    if DIFFUSION_SOURCE != 'zenodo':
        if not (DIFFUSION_PREPARED/'dataset.json').is_file():
            prepare_diffusion(DIFFUSION_SOURCE,DIFFUSION_PREPARED)
        command += ['--dataset',str(DIFFUSION_PREPARED)]
    if DIFFUSION_RESUME:
        command += ['--resume',DIFFUSION_RESUME]
    subprocess.run(command,check=True)
'''),md('''## 8. Посмотреть последнее обучающее превью

Превью сохраняются в output/diffusion_runs/run_... каждые 1000 шагов и в конце.
Новый checkpoint для генерации задаётся в DIFFUSION_WEIGHTS.
'''),code('''previews = list((OUTPUT_ROOT/'diffusion_runs').glob('run_*/preview_*.png'))
if previews:
    latest = max(previews,key=lambda p:p.stat().st_mtime)
    display(DisplayImage(filename=str(latest),width=1500))
    print(latest)
''')]

notebook=dict(nbformat=4,nbformat_minor=5,metadata={
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python'},'colab':{'name':'Zebrafish_Framework.ipynb'}},cells=cells)
target=ROOT/'notebooks/Zebrafish_Framework.ipynb'
target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
print(target)
