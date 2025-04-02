import pandas as pd
import numpy as np

class Preprocessor:
    # Метод fit: обучает препроцессор на данных X
    def fit(self, X: pd.DataFrame):
        df_train = X.copy()  # Создаём копию входного датафрейма

        # Определяем вспомогательные списки
        self.aux = ['sk_id_curr']  # Список столбцов, которые нужно исключить (например, идентификаторы)
        self.target = ['target']  # Список целевых переменных

        # Формируем список категориальных переменных
        self.categ = [col for col in df_train.columns if col not in (self.aux + self.target) and df_train[col].nunique() <= 7] + \
                     [col for col in df_train.columns if col not in (self.aux + self.target) and df_train[col].dtypes == np.dtype('O')]

        # Удаляем дубликаты из списка категориальных переменных
        self.categ = list(set(self.categ))

        # Формируем список числовых переменных
        self.nums = [col for col in df_train.columns if col not in (self.categ + self.target + self.aux)]

    # Метод predict: преобразует данные X на основе категориальных переменных
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()  # Создаём копию входного датафрейма

        # Преобразуем категориальные переменные в строковый формат и заполняем пропущенные значения строкой 'NaN'
        for col in self.categ:
            df[col] = df[col].astype(str).fillna('NaN')

        return df  # Возвращаем преобразованный датафрейм

    # Метод fit_predict: объединяет обучение и преобразование данных
    def fit_predict(self, X: pd.DataFrame) -> pd.DataFrame:
        # Выполняем обучение на данных X
        self.fit(X)

        # Выполняем преобразование данных X
        return self.predict(X)
