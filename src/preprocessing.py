import json
import os
import random
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.utils import FaceRecognitionDataset


def prepare_datasets(
    data_path: str = "data/train",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Подготовка датасетов для обучения и оценки модели.

    Args:
        data_path: Путь к директории с данными

    Returns:
        Кортеж (train_dataloader, val_dataloader, test_dataloader)
    """
    # Получаем список людей и количество их изображений
    people = {
        person: len(os.listdir(f"{data_path}/images/{person}"))
        for person in os.listdir(f"{data_path}/images")
    }

    # Загружаем метаданные
    meta_path = Path(data_path) / "meta.json"
    with open(meta_path, "r") as f:
        meta_data = json.load(f)

    # Словари для хранения путей к изображениям
    real_images: Dict[str, List[str]] = {}  # словарь для реальных изображений
    fake_images: Dict[str, List[str]] = {}  # словарь для синтетических изображений

    # Распределяем изображения по словарям
    for img_path, label in meta_data.items():
        person_id, _ = img_path.split(
            "/"
        )  # разделяем индекс человека и номер изображения

        if label == 0:  # Реальное изображение
            if person_id not in real_images:
                real_images[person_id] = []
            real_images[person_id].append(img_path)
        else:  # Синтетическое изображение
            fake_key = f"{person_id}_fake"
            if fake_key not in fake_images:
                fake_images[fake_key] = []
            fake_images[fake_key].append(img_path)

    # Создаем пары изображений
    positive_pairs = []  # Пары изображений одного человека (положительные примеры)
    negative_pairs = []  # Пары изображений разных людей или реальное/синтетическое (отрицательные примеры)

    # Создаем положительные пары (два реальных изображения одного человека)
    for person, images in real_images.items():
        if len(images) > 1:
            # Все возможные пары для одного человека
            positive_pairs.extend([(i, 1) for i in combinations(images, 2)])

    # Создаем отрицательные пары
    people_list = list(real_images.keys())

    for _ in range(len(positive_pairs)):
        if random.choice([True, False]):
            # Пара из реальных изображений разных людей
            p1, p2 = random.sample(people_list, 2)
            img1 = random.choice(real_images[p1])
            img2 = random.choice(real_images[p2])
        else:
            # Пара из реального и синтетического изображения одного человека
            p1 = random.choice(people_list)
            img1 = random.choice(real_images[p1])
            img2 = random.choice(fake_images[f"{p1}_fake"])

        negative_pairs.append(((img1, img2), -1))

    print(
        f"✔ Создано {len(positive_pairs)} позитивных пар и {len(negative_pairs)} негативных пар."
    )

    # Ограничиваем количество пар для обучения и тестирования
    all_pairs = positive_pairs[:15000] + negative_pairs[:22000]
    test_pairs = positive_pairs[15000:15600] + negative_pairs[22000:22800]
    random.shuffle(all_pairs)

    # Разделяем данные на обучающую и валидационную выборки
    train_pairs, val_pairs = train_test_split(
        all_pairs, test_size=0.25, random_state=42
    )

    # Создаем Dataset'ы
    train_dataset = FaceRecognitionDataset(train_pairs, "train")
    val_dataset = FaceRecognitionDataset(val_pairs, "test")
    test_dataset = FaceRecognitionDataset(test_pairs, "test")

    # Создаем DataLoader'ы
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    return train_dataloader, val_dataloader, test_dataloader
