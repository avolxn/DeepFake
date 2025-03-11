import argparse
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


def evaluate_eer(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    gt_label_column: str = "label",
    sub_sim_column: str = "similarity",
    id_column: str = "pair_id",
) -> float:
    """
    Оценивает EER (Equal Error Rate) между предсказаниями и истинными метками.

    Args:
        gt_df: DataFrame с истинными метками
        pred_df: DataFrame с предсказаниями
        gt_label_column: Название колонки с истинными метками
        sub_sim_column: Название колонки с предсказанными вероятностями
        id_column: Название колонки с идентификаторами

    Returns:
        Значение EER
    """
    # Преобразование id_column в int для совместимости
    gt_df = gt_df.astype({id_column: int})
    pred_df = pred_df.astype({id_column: int})

    # Соединение DataFrame'ов по id_column
    gt_df = gt_df.join(pred_df.set_index(id_column), on=id_column, how="left")

    if gt_df[sub_sim_column].isna().any():
        print("Не все `pair_id` присутствуют в предсказаниях")

    # Получение списков для вычисления EER
    y_score = pred_df[sub_sim_column].tolist()
    y_true = gt_df[gt_label_column].tolist()

    return compute_eer(y_true, y_score)


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


def evaluate_model(
    public_test_url: str,
    public_prediction_url: str,
    private_test_url: Optional[str] = None,
    private_prediction_url: Optional[str] = None,
) -> Dict[str, Optional[float]]:
    """
    Оценивает модель на открытых и закрытых тестах.

    Args:
        public_test_url: Путь к общедоступному тесту
        public_prediction_url: Путь к предсказаниям для общедоступного теста
        private_test_url: Путь к приватному тесту
        private_prediction_url: Путь к предсказаниям для приватного теста

    Returns:
        Словарь с оценками
    """
    # Вычисление оценки на общедоступном тесте
    public_gt_df = pd.read_csv(public_test_url)
    public_pred_df = pd.read_csv(public_prediction_url)
    public_score = evaluate_eer(public_gt_df, public_pred_df)

    # Вычисление оценки на приватном тесте, если предоставлены пути
    private_score = None
    if private_test_url and private_prediction_url:
        private_gt_df = pd.read_csv(private_test_url)
        private_pred_df = pd.read_csv(private_prediction_url)
        private_score = evaluate_eer(private_gt_df, private_pred_df)

    return {"public_score": public_score, "private_score": private_score}


if __name__ == "__main__":
    # Настройка аргументов командной строки
    parser = argparse.ArgumentParser(description="Оценка модели на тестовых данных")
    parser.add_argument(
        "--public_test_url", type=str, required=True, help="Путь к общедоступному тесту"
    )
    parser.add_argument(
        "--public_prediction_url",
        type=str,
        required=True,
        help="Путь к предсказаниям для общедоступного теста",
    )
    parser.add_argument(
        "--private_test_url", type=str, required=False, help="Путь к приватному тесту"
    )
    parser.add_argument(
        "--private_prediction_url",
        type=str,
        required=False,
        help="Путь к предсказаниям для приватного теста",
    )
    args = parser.parse_args()

    # Оценка модели
    scores = evaluate_model(
        args.public_test_url,
        args.public_prediction_url,
        args.private_test_url,
        args.private_prediction_url,
    )

    # Вывод результатов
    print(json.dumps(scores, indent=2))
