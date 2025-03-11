import os
from pathlib import Path
from typing import List, Union

import pandas as pd
import torch
from oml import datasets as d
from oml.inference import inference
from torch.nn import functional as F
from torchvision import transforms

from src.train_model import load_model_by_repo_id, load_model_weights

# Константы
LOCAL_WEIGHTS_PATH = Path("experiments/model_weights_epoch_3.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_REPO_ID = "minchul/cvlface_adaface_vit_base_webface4m"
DEFAULT_MODEL_PATH = (
    Path.home() / ".cvlface_cache/minchul/cvlface_adaface_vit_base_webface4m"
)
# Загружаем токен из переменной окружения
HF_TOKEN = os.environ.get("HF_TOKEN")

# Трансформации для тестовых изображений
TRANSFORM_TEST = transforms.Compose(
    [
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


def create_sample_sub(pair_ids: List[str], sim_scores: List[float]) -> pd.DataFrame:
    """
    Создает DataFrame для сабмита.

    Args:
        pair_ids: ID пар изображений
        sim_scores: Оценки схожести

    Returns:
        DataFrame для сабмита
    """
    sub_sim_column = "similarity"
    id_column = "pair_id"
    return pd.DataFrame({id_column: pair_ids, sub_sim_column: sim_scores})


def load_model(weights_path: Union[str, Path] = LOCAL_WEIGHTS_PATH) -> torch.nn.Module:
    """
    Загружает модель для инференса.

    Args:
        weights_path: Путь к весам модели

    Returns:
        Загруженная модель
    """
    # Загрузка модели и весов
    model = load_model_by_repo_id(DEFAULT_REPO_ID, DEFAULT_MODEL_PATH, HF_TOKEN)
    model = load_model_weights(model, weights_path, DEVICE)

    # Перевод модели в режим оценки
    model = model.to(DEVICE).eval()

    return model


def predict_similarity(
    model: torch.nn.Module,
    test_path: str = "test.csv",
    output_path: str = "data/submission.csv",
) -> pd.DataFrame:
    """
    Выполняет предсказание схожести между парами изображений.

    Args:
        model: Модель для инференса
        test_path: Путь к тестовому CSV файлу
        output_path: Путь для сохранения результатов

    Returns:
        DataFrame с результатами
    """
    # Создаем директорию для данных, если она не существует
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Загрузка тестовых данных
    df_test = pd.read_csv(test_path)
    test_dataset = d.ImageQueryGalleryLabeledDataset(df_test, transform=TRANSFORM_TEST)

    # Выполнение инференса с помощью модели
    embeddings = inference(
        model, test_dataset, batch_size=32, num_workers=0, verbose=True
    )

    # Вычисление косинусного сходства между парами
    e1 = embeddings[::2]  # Четные индексы (0, 2, 4, ...)
    e2 = embeddings[1::2]  # Нечетные индексы (1, 3, 5, ...)
    sim_scores = F.cosine_similarity(e1, e2).detach().cpu().numpy().tolist()

    # Подготовка ID пар для сабмита
    pair_ids = df_test["label"].apply(lambda x: f"{x:08d}").to_list()
    pair_ids = pair_ids[::2]  # Берем только четные индексы, так как пары

    # Создание и сохранение файла сабмита
    sub_df = create_sample_sub(pair_ids, sim_scores)
    sub_df.to_csv(output_path, index=False)

    return sub_df


def run_prediction(
    test_path: str = "test.csv", output_path: str = "data/submission.csv"
) -> pd.DataFrame:
    """
    Запускает процесс предсказания: загружает модель и выполняет инференс.

    Args:
        test_path: Путь к тестовому CSV файлу
        output_path: Путь для сохранения результатов

    Returns:
        DataFrame с результатами
    """
    model = load_model()
    return predict_similarity(model, test_path, output_path)


if __name__ == "__main__":
    run_prediction()
