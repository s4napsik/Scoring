import datetime

import pandas as pd

def create_submission(submission: pd.DataFrame, prediction: pd.Series):
    # Добавляем столбец 'TARGET' с предсказаниями в датафрейм submission
    submission['target'] = prediction

    # Формируем имя файла с использованием текущей даты и времени
    sub_name = f"submissions/sub_{str(datetime.datetime.now())}.csv"

    # Сохраняем датафрейм в CSV-файл без индекса
    submission.to_csv(sub_name, index=False)

    # Выводим имя созданного файла
    print(sub_name)

    # Возвращаем датафрейм, прочитанный из созданного CSV-файла
    return pd.read_csv(sub_name)
