# Имя файла: Preprocessing.py (Версия с очисткой имен в fit)
import logging
import re

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Preprocessing:
    """
    Класс для предобработки данных (train и test).
    Включает обработку пропусков и One-Hot Encoding для категориальных признаков.
    Обучается на train данных (игнорируя target) и применяет те же трансформации к test данным.
    НОВАЯ ВЕРСИЯ: Метод transform обрабатывает признаки "на месте", не удаляя target.
    НОВАЯ ВЕРСИЯ 2: Имена OHE колонок очищаются от спецсимволов во время fit.
    """
    def __init__(self):
        self.num_cols = None
        self.cat_cols = None
        self.num_imputer = None
        self.cat_imputer = None
        self.encoder = None
        self.ohe_feature_names = None # Будут храниться ОЧИЩЕННЫЕ имена

    def _sanitize_column_names(self, columns: list) -> list:
        """Внутренний метод для очистки списка имен колонок."""
        sanitized_columns = []
        for col in columns:
            col_str = str(col)
            # Заменяем проблемные символы JSON и другие на '_'
            new_col = re.sub(r'[",:{}[\]<>/\s\.]', '_', col_str)
            # Заменяем последовательности подчеркиваний на одно
            new_col = re.sub(r'_+', '_', new_col)
            # Убираем подчеркивание в начале/конце
            new_col = new_col.strip('_')
            sanitized_columns.append(new_col)

        # Проверяем и обрабатываем дубликаты после очистки
        if len(sanitized_columns) != len(set(sanitized_columns)):
            logging.warning("Обнаружены дублирующиеся имена колонок после очистки! Применяем суффиксы.")
            counts = {}
            final_unique_columns = []
            for name in sanitized_columns:
                if name in counts:
                    counts[name] += 1
                    final_unique_columns.append(f"{name}_{counts[name]}")
                else:
                    counts[name] = 0
                    final_unique_columns.append(name)
            logging.warning(f"    Дубликаты после обработки: {[col for col in final_unique_columns if '_' in col.split('_')[-1] and col.split('_')[-1].isdigit()]}") # Показываем только переименованные
            return final_unique_columns
        else:
            return sanitized_columns

    def fit(self, df: pd.DataFrame, target_column: str = 'target'):
        """
        Обучает импьютеры и энкодер. Имена OHE колонок очищаются.
        Целевая колонка ИГНОРИРУЕТСЯ при обучении.
        """
        logging.info("Starting preprocessing fitting...")
        df_features = df.copy()

        if target_column in df_features.columns:
            df_features = df_features.drop(columns=[target_column])
            logging.info(f"Target column '{target_column}' excluded for fitting.")
        else:
            logging.warning(f"Target column '{target_column}' not found.")

        self.num_cols = df_features.select_dtypes(include=np.number).columns.tolist()
        self.cat_cols = df_features.select_dtypes(include='object').columns.tolist()
        logging.info(f"Identified {len(self.num_cols)} numerical feature columns.")
        logging.info(f"Identified {len(self.cat_cols)} categorical feature columns.")

        # --- Обучение импьютеров (без изменений) ---
        self.num_imputer = SimpleImputer(strategy='median')
        if self.num_cols:
             self.num_imputer.fit(df_features[self.num_cols])
             logging.info("Numerical imputer fitted.")
        else:
            logging.warning("No numerical columns found to fit numerical imputer.")

        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        if self.cat_cols:
            self.cat_imputer.fit(df_features[self.cat_cols])
            logging.info("Categorical imputer fitted.")
        else:
            logging.warning("No categorical columns found to fit categorical imputer.")

        # --- Обучение OneHotEncoder и ОЧИСТКА ИМЕН ---
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        if self.cat_cols:
            df_cat_imputed_for_fit = self.cat_imputer.transform(df_features[self.cat_cols])
            self.encoder.fit(df_cat_imputed_for_fit)

            # Получаем сырые имена от энкодера
            raw_ohe_names = self.encoder.get_feature_names_out(self.cat_cols)
            logging.info(f"Generated {len(raw_ohe_names)} raw OHE feature names.")

            # Очищаем имена
            self.ohe_feature_names = self._sanitize_column_names(raw_ohe_names)
            logging.info(f"Sanitized OHE feature names. Count: {len(self.ohe_feature_names)}")
            # logging.info(f"   Example sanitized names: {self.ohe_feature_names[:10]}") # Раскомментируйте для отладки

        else:
             self.ohe_feature_names = [] # Пустой список, если не было категорий
             logging.warning("No categorical columns found to fit OneHotEncoder.")

        logging.info("Preprocessing fitting finished.")
        return self

    # Используем версию transform, которая не удаляет target явно
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет обученные трансформации к датафрейму.
        Использует ОЧИЩЕННЫЕ имена для OHE колонок.
        """
        if self.num_imputer is None or self.cat_imputer is None or self.encoder is None:
            logging.error("Preprocessing instance has not been fitted yet. Call 'fit' first.")
            raise RuntimeError("The preprocessing instance has not been fitted yet. Call 'fit' first.")

        logging.info(f"Starting preprocessing transform for dataframe with shape {df.shape}...")
        df_processed = df.copy()

        # Проверка и добавление недостающих колонок
        missing_num = [col for col in self.num_cols if col not in df_processed.columns]
        missing_cat = [col for col in self.cat_cols if col not in df_processed.columns]
        if missing_num:
            logging.warning(f"Missing numerical columns in input df: {missing_num}. Adding them with NaN.")
            for col in missing_num:
                df_processed[col] = np.nan
        if missing_cat:
            logging.warning(f"Missing categorical columns in input df: {missing_cat}. Adding them with NaN.")
            for col in missing_cat:
                df_processed[col] = np.nan

        # 1. Применяем импьютеры
        if self.num_cols:
             try:
                 cols_to_impute = [col for col in self.num_cols if col in df_processed.columns]
                 if cols_to_impute:
                     df_processed[cols_to_impute] = self.num_imputer.transform(df_processed[cols_to_impute])
                     logging.info("Numerical imputation applied.")
             except Exception as e:
                 logging.error(f"Error during numerical imputation: {e}")
                 return None
        else:
            logging.info("No numerical columns specified during fit.")

        df_cat_encoded = pd.DataFrame(index=df_processed.index)
        if self.cat_cols:
            try:
                 cols_to_impute = [col for col in self.cat_cols if col in df_processed.columns]
                 if cols_to_impute:
                     df_processed[cols_to_impute] = self.cat_imputer.transform(df_processed[cols_to_impute])
                     logging.info("Categorical imputation applied.")

                     # 2. Применяем OHE (используем очищенные имена из self.ohe_feature_names)
                     if hasattr(self, 'ohe_feature_names') and self.ohe_feature_names:
                         cat_data_for_encode = df_processed[self.cat_cols]
                         encoded_data = self.encoder.transform(cat_data_for_encode)
                         # !!! Используем очищенные имена !!!
                         df_cat_encoded = pd.DataFrame(encoded_data, columns=self.ohe_feature_names, index=df_processed.index)
                         logging.info("OneHotEncoder transformation applied using sanitized names.")
                     else:
                         logging.warning("OneHotEncoder was not fitted or has no feature names.")
                 else:
                      logging.info("No categorical columns found in dataframe to transform.")
            except Exception as e:
                 logging.error(f"Error during categorical processing or OHE: {e}")
                 return None
        else:
             logging.info("No categorical columns specified during fit.")

        # 3. Удаляем ИСХОДНЫЕ категориальные колонки
        cols_to_drop = [col for col in self.cat_cols if col in df_processed.columns]
        if cols_to_drop:
             df_processed = df_processed.drop(columns=cols_to_drop)
             logging.info(f"Original categorical columns dropped: {cols_to_drop}")

        # 4. Соединяем датафрейм (без исходных категориальных) с OHE признаками
        # Сначала колонки, которые были изначально (кроме удаленных cat_cols), потом новые OHE
        final_cols_order = [col for col in df_processed.columns if col not in df_cat_encoded.columns] + df_cat_encoded.columns.tolist()
        # Используем reindex для гарантии порядка и наличия всех нужных колонок
        df_final = pd.concat([df_processed, df_cat_encoded], axis=1).reindex(columns=final_cols_order)

        # Опционально: Проверить и очистить ВСЕ финальные имена на всякий случай?
        # df_final.columns = self._sanitize_column_names(df_final.columns.tolist())

        logging.info(f"Processed dataframe shape: {df_final.shape}")
        logging.info("Preprocessing transform finished.")
        return df_final