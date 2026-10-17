# Face Recognition Project

Классификация изображений 5 знаменитостей с помощью PyTorch / torchvision / PyTorch Lightning.

Проект учебный: сравниваются три подхода к fine-tuning предобученных CNN на небольшом датасете лиц.

## Результат на валидационной выборке

| Модель | Подход | Val loss | Val accuracy |
|---|---|---:|---:|
| Model 1 | ResNet18 + PyTorch Lightning, полное дообучение | 0.0248 | **99.45%** |
| Model 2 | ResNet18 + PyTorch Lightning, frozen backbone + gradual unfreezing | 0.0493 | 98.80% |
| Model 3 | ResNet50 + `nn.Module`, frozen backbone, ручной training loop | 0.4434 | 84.68% |

Лучшая модель: **Model 1 — ResNet18, Lightning, полная донастройка**.  
Веса сохраняются локально в `best_model.pt`.

## Датасет

Ожидается структура:

```text
data/
  train/
    bill_gates/
    elon_musk/
    jeff_bezos/
    mark_zuckerberg/
    steve_jobs/
  valid/
    bill_gates/
    elon_musk/
    jeff_bezos/
    mark_zuckerberg/
    steve_jobs/
```

В текущем ноутбуке:

- train: **3000** изображений;
- valid: **914** изображений;
- классы: `bill_gates`, `elon_musk`, `jeff_bezos`, `mark_zuckerberg`, `steve_jobs`.

Папка `data/` не коммитится в git (см. `.gitignore`).

## Что сделано в ноутбуке

- Загрузка данных через `torchvision.datasets.ImageFolder`.
- Разные трансформации для train/valid:
  - train: `RandomResizedCrop`, `RandomHorizontalFlip`, `ToTensor`, `Normalize`;
  - valid: `Resize`, `CenterCrop`, `ToTensor`, `Normalize`.
- Model 1: предобученный ResNet18, замена `fc`, обучение всех параметров через `pl.Trainer`.
- Model 2: ResNet18 с замороженным backbone, обучение только классификатора; затем gradual unfreezing последнего блока `layer4`.
- Model 3: ResNet50 как чистый `nn.Module`, замороженный backbone, ручной цикл обучения.
- Сравнение моделей по `val_loss` / `val_accuracy`, выбор лучшей модели, сохранение весов.
- Визуализация предсказаний лучшей модели на валидационных изображениях.

## Как запустить

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

pip install torch torchvision pytorch-lightning opencv-python matplotlib scikit-learn
```

Затем открыть и выполнить:

```text
Face_recognition.ipynb
```

После обучения можно посчитать confusion matrix для лучшей модели — см. ячейку с `ConfusionMatrixDisplay` в конце ноутбука или сниппет в истории проекта.

## Файлы

- `Face_recognition.ipynb` — основной ноутбук с обучением и сравнением моделей.
- `data/` — локальный датасет, не коммитится.
- `best_model.pt`, `lightning_logs/` — локальные артефакты обучения, не коммитятся.
