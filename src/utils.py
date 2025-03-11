from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve
from torch.utils.data import Dataset
from torchvision import transforms


class FaceRecognitionDataset(Dataset):
    """Датасет для задачи распознавания лиц."""

    def __init__(self, pairs: List[Tuple], mode: str):
        """
        Инициализирует датасет для распознавания лиц.

        Args:
            pairs: Список пар изображений с метками
            mode: Режим работы ('train' или 'test')
        """
        self.pairs = pairs
        self.mode = mode

        # Общие параметры нормализации
        self.norm_mean = [0.5, 0.5, 0.5]
        self.norm_std = [0.5, 0.5, 0.5]

        # Трансформации для обучения
        self.transform_train = transforms.Compose(
            [
                transforms.Resize((112, 112)),
                transforms.RandomHorizontalFlip(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(self.norm_mean, self.norm_std),
            ]
        )

        # Трансформации для тестирования
        self.transform_test = transforms.Compose(
            [
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(self.norm_mean, self.norm_std),
            ]
        )

    def __len__(self) -> int:
        """Возвращает количество пар в датасете."""
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple:
        """
        Возвращает пару изображений и метку.

        Args:
            index: Индекс пары

        Returns:
            Кортеж (image1, image2, label)
        """
        # Получаем пути к изображениям
        path_to_image0 = Path(f"data/train/images/{self.pairs[index][0][0]}")
        path_to_image1 = Path(f"data/train/images/{self.pairs[index][0][1]}")

        # Получаем метку
        label = self.pairs[index][1]

        # Применяем соответствующие трансформации
        if self.mode == "train":
            image1 = self.transform_train(Image.open(path_to_image0))
            image1 = image1.requires_grad_(True)
            image2 = self.transform_train(Image.open(path_to_image1))
            image2 = image2.requires_grad_(True)
        else:  # режим test
            image1 = self.transform_test(Image.open(path_to_image0))
            image2 = self.transform_test(Image.open(path_to_image1))

        return image1, image2, label


def compute_eer(label: List[int], pred: List[float], positive_label: int = 1) -> float:
    """
    Вычисляет Equal Error Rate (EER).

    Args:
        label: Истинные метки
        pred: Предсказанные вероятности
        positive_label: Метка положительного класса

    Returns:
        Значение EER
    """
    # Вычисляем ROC-кривую
    fpr, tpr, _ = roc_curve(label, pred)
    fnr = 1 - tpr

    # Находим индекс, где FNR ≈ FPR
    idx = np.nanargmin(np.absolute((fnr - fpr)))

    # Вычисляем EER как среднее между FPR и FNR в этой точке
    eer_1 = fpr[idx]
    eer_2 = fnr[idx]

    return (eer_1 + eer_2) / 2
