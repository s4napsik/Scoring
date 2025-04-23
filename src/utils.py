import datetime
import os
import pandas as pd


def create_submission(submission: pd.DataFrame, prediction: pd.Series):
    os.makedirs('submissions', exist_ok=True)
    # Убедись, что имя колонки 'target' или 'TARGET' соответствует требованиям Kaggle
    submission['target'] = prediction  # Или 'TARGET'
    timestamp = str(datetime.datetime.now()).replace(':', '-')
    sub_name = f"submissions/sub_lgbm_{timestamp}.csv"
    # Сохраняем датафрейм в CSV-файл без индекса
    submission.to_csv(sub_name, index=False)
    print(f"Файл сабмишна создан: {sub_name}")
    return pd.read_csv(sub_name)