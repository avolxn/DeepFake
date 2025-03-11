import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel

from src.preprocessing import prepare_datasets

# Определяем константы
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_EPOCHS = 15
DEFAULT_SAVE_PATH = Path("data/weights/")
DEFAULT_REPO_ID = "minchul/cvlface_adaface_vit_base_webface4m"
DEFAULT_MODEL_PATH = (
    Path.home() / ".cvlface_cache/minchul/cvlface_adaface_vit_base_webface4m"
)
# Загружаем токен из переменной окружения
DEFAULT_HF_TOKEN = os.environ.get("HF_TOKEN")


def save_model_weights(model: nn.Module, path: Union[str, Path]) -> None:
    """
    Сохраняет веса модели локально.

    Args:
        model: Модель для сохранения
        path: Путь для сохранения
    """
    torch.save(model.state_dict(), path)
    print(f"Model weights saved to {path}")


def download(
    repo_id: str, path: Union[str, Path], hf_token: Optional[str] = None
) -> None:
    """
    Загружает модель с Hugging Face.

    Args:
        repo_id: ID репозитория на Hugging Face
        path: Путь для сохранения локально
        hf_token: Токен Hugging Face API
    """
    os.makedirs(path, exist_ok=True)
    files_path = os.path.join(path, "files.txt")

    # Загружаем список файлов, если его нет
    if not os.path.exists(files_path):
        hf_hub_download(
            repo_id,
            "files.txt",
            token=hf_token,
            local_dir=path,
            local_dir_use_symlinks=False,
        )

    # Читаем список файлов
    with open(os.path.join(path, "files.txt"), "r") as f:
        files = f.read().split("\n")

    # Добавляем обязательные файлы
    required_files = [f for f in files if f] + [
        "config.json",
        "wrapper.py",
        "model.safetensors",
    ]

    # Загружаем каждый файл, если его нет
    for file in required_files:
        full_path = os.path.join(path, file)
        if not os.path.exists(full_path):
            hf_hub_download(
                repo_id,
                file,
                token=hf_token,
                local_dir=path,
                local_dir_use_symlinks=False,
            )


def load_model_from_local_path(
    path: Union[str, Path], hf_token: Optional[str] = None
) -> nn.Module:
    """
    Загружает модель из локального пути.

    Args:
        path: Путь к моделям
        hf_token: Токен Hugging Face API

    Returns:
        Модель PyTorch
    """
    # Сохраняем текущую директорию
    cwd = os.getcwd()

    # Переходим в директорию с моделью
    os.chdir(path)
    sys.path.insert(0, path)

    # Загружаем модель
    model = AutoModel.from_pretrained(path, trust_remote_code=True, token=hf_token)

    # Возвращаемся в исходную директорию
    os.chdir(cwd)
    sys.path.pop(0)

    return model


def load_model_by_repo_id(
    repo_id: str,
    save_path: Union[str, Path],
    hf_token: Optional[str] = None,
    force_download: bool = False,
) -> nn.Module:
    """
    Загружает модель по ID репозитория.

    Args:
        repo_id: ID репозитория на Hugging Face
        save_path: Путь для сохранения локально
        hf_token: Токен Hugging Face API
        force_download: Принудительная загрузка даже если уже существует

    Returns:
        Модель PyTorch
    """
    # Удаляем существующую директорию, если требуется принудительная загрузка
    if force_download and os.path.exists(save_path):
        shutil.rmtree(save_path)

    # Загружаем файлы модели
    download(repo_id, save_path, hf_token)

    # Загружаем модель из локального пути
    return load_model_from_local_path(save_path, hf_token)


def load_model_weights(
    model: nn.Module, path: Union[str, Path], device: torch.device
) -> nn.Module:
    """
    Загружает веса модели локально.

    Args:
        model: Модель для загрузки весов
        path: Путь к весам
        device: Устройство для загрузки (CPU/GPU)

    Returns:
        Модель с загруженными весами
    """
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Model weights loaded from {path}")
    else:
        print(f"No local weights found at {path}")
    return model


def _train_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    dataloader: DataLoader,
    summary_loss: List[float],
) -> Tuple[float, int]:
    """
    Выполняет одну эпоху обучения.

    Args:
        model: Модель для обучения
        optimizer: Оптимизатор
        criterion: Функция потерь
        dataloader: Загрузчик данных
        summary_loss: Список для сохранения значений потерь

    Returns:
        Кортеж (суммарная потеря, нормализующая переменная)
    """
    model.train()
    total_loss = 0
    norm_variable = 0

    for image0, image1, labels in dataloader:
        # Перемещаем данные на устройство
        image0 = image0.to(DEVICE)
        image1 = image1.to(DEVICE)
        labels = labels.to(DEVICE)

        # Обнуляем градиенты
        optimizer.zero_grad()

        # Получаем эмбеддинги
        embed0 = model(image0)
        embed1 = model(image1)

        # Вычисляем потерю
        loss = criterion(embed0, embed1, labels)
        summary_loss.append(loss.item())

        # Накапливаем потерю
        total_loss += loss.item() * image0.size(0)
        norm_variable += image0.size(0)

        # Обратное распространение и оптимизация
        loss.backward()
        optimizer.step()

    return total_loss, norm_variable


def _validate_epoch(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
    summary_loss: List[float],
) -> Tuple[float, int]:
    """
    Выполняет одну эпоху валидации.

    Args:
        model: Модель для валидации
        criterion: Функция потерь
        dataloader: Загрузчик данных
        summary_loss: Список для сохранения значений потерь

    Returns:
        Кортеж (суммарная потеря, нормализующая переменная)
    """
    model.eval()
    total_loss = 0
    norm_variable = 0

    with torch.no_grad():
        for image0, image1, labels in dataloader:
            # Перемещаем данные на устройство
            image0 = image0.to(DEVICE)
            image1 = image1.to(DEVICE)
            labels = labels.to(DEVICE)

            # Получаем эмбеддинги
            embed0 = model(image0)
            embed1 = model(image1)

            # Вычисляем потерю
            loss = criterion(embed0, embed1, labels)
            summary_loss.append(loss.item())

            # Накапливаем потерю
            total_loss += loss.item() * image0.size(0)
            norm_variable += image0.size(0)

    return total_loss, norm_variable


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    save_path: Union[str, Path] = DEFAULT_SAVE_PATH,
) -> None:
    """
    Обучает модель.

    Args:
        model: Модель для обучения
        optimizer: Оптимизатор
        criterion: Функция потерь
        epochs: Количество эпох
        train_dataloader: Загрузчик тренировочных данных
        val_dataloader: Загрузчик тестовых данных
        save_path: Путь для сохранения весов
    """
    # Создаем директорию для сохранения весов
    os.makedirs(save_path, exist_ok=True)

    # Списки для хранения значений потерь
    summary_loss_train = []
    summary_loss_test = []

    # Цикл по эпохам
    for epoch in tqdm(range(epochs), desc="Epochs", ncols=100):
        # Обучение
        train_loss, train_norm_variable = _train_epoch(
            model, optimizer, criterion, train_dataloader, summary_loss_train
        )
        print(f"Train loss: {train_loss / train_norm_variable:.6f}")

        # Сохранение весов
        model_path = os.path.join(save_path, f"model_weights_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model weights saved to {model_path}")

        # Валидация
        test_loss, test_norm_variable = _validate_epoch(
            model, criterion, val_dataloader, summary_loss_test
        )
        print(f"Test loss: {test_loss / test_norm_variable:.6f}")


def _prepare_model_for_training(model: nn.Module) -> nn.Module:
    """
    Подготавливает модель для обучения, замораживая и размораживая нужные слои.

    Args:
        model: Исходная модель

    Returns:
        Подготовленная модель
    """
    # Замораживаем все параметры
    for param in model.parameters():
        param.requires_grad = False

    # Размораживаем BatchNorm слои
    for layer in model.model.net.feature:
        if isinstance(layer, torch.nn.BatchNorm1d):
            for param in layer.parameters():
                param.requires_grad = False  # Замораживаем параметры

    # Размораживаем Linear слои
    for layer in model.model.net.feature:
        for param in layer.parameters():
            param.requires_grad = True

    # Проверяем какие параметры обучаемые
    for name, param in model.named_parameters():
        print(f"Параметр {name} обучаемый: {param.requires_grad}")

    return model


def train_face_recognition_model(
    hf_token: Optional[str] = DEFAULT_HF_TOKEN,
    epochs: int = DEFAULT_EPOCHS,
    repo_id: str = DEFAULT_REPO_ID,
) -> nn.Module:
    """
    Основная функция для обучения модели распознавания лиц.

    Args:
        hf_token: Токен Hugging Face API
        epochs: Количество эпох для обучения
        repo_id: ID репозитория с базовой моделью

    Returns:
        Обученная модель
    """
    # Подготовка данных
    train_dataloader, val_dataloader, _ = prepare_datasets()

    # Загрузка модели
    model = load_model_by_repo_id(repo_id, DEFAULT_MODEL_PATH, hf_token)

    # Подготовка модели для обучения
    model = _prepare_model_for_training(model)

    # Настройка оптимизатора и функции потерь
    optimizer = optim.Adam(model.head.parameters(), lr=3e-3)
    criterion = nn.CosineEmbeddingLoss()

    # Перемещаем модель на устройство
    model = model.to(DEVICE)

    # Обучаем модель
    train(model, optimizer, criterion, epochs, train_dataloader, val_dataloader)

    return model


if __name__ == "__main__":
    # Обучение модели
    trained_model = train_face_recognition_model()
