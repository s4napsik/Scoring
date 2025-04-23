# Имя файла: training_pipeline_returning_models.py
# (Сохрани этот код в корень проекта или в src/)

import pandas as pd
import numpy as np
import time
import gc
from typing import Dict, Any, Tuple

# Модели
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Утилиты Sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, log_loss

# Optuna
import optuna

# MLflow
import mlflow
import logging  # Добавим логирование для вывода

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_training_pipeline(df: pd.DataFrame, target_col: str = 'target') -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], Dict[str, Any]]:
    """
    Запускает пайплайн: HPO(Optuna) + CV + Обучение финальных моделей.
    Возвращает OOF предсказания, CV метрики И СЛОВАРЬ С ФИНАЛЬНЫМИ МОДЕЛЯМИ.

    Args:
        df (pd.DataFrame): Входной датафрейм (предполагается *уже препроцессированный*),
                           содержащий признаки и целевую колонку.
        target_col (str): Имя целевой колонки (по умолчанию 'target').

    Returns:
        Tuple[pd.DataFrame, Dict[str, Dict[str, float]], Dict[str, Any]]:
            - Датафрейм с Out-of-Fold предсказаниями.
            - Словарь со средними метриками CV для каждой модели.
            - Словарь с финальными моделями (ключ - название, значение - объект модели).
    """
    logging.info("--- Запуск Пайплайна Обучения (с возвратом моделей) ---")
    # Используем настройки 2/2 для примера, НО для реальной работы нужны бОльшие значения!
    logging.warning("!!! ВНИМАНИЕ: Настройки N_SPLITS=2, N_TRIALS_OPTUNA=2 для быстрой проверки, НЕ для надежной оценки/тюнинга !!!")

    # --- 1. Жестко Заданные Параметры ---
    # TARGET_COL = 'target' # <-- УДАЛЕНО! Используем аргумент target_col
    # !!! ВАЖНО: Проверь доступность этого MLflow сервера или замени на локальный путь "./mlruns" !!!
    MLFLOW_TRACKING_URI = "http://82.202.137.136:8000"
    EXPERIMENT_NAME = "Pipeline Run Returning Models (2:2 Example)"
    N_SPLITS = 2        # !!! Мало для реальной работы !!!
    N_TRIALS_OPTUNA = 2 # !!! Мало для реальной работы !!!
    RANDOM_STATE = 42

    # --- Определение списка признаков ---
    # Используем все колонки, кроме целевой и известных ID
    logging.info("Определение списка признаков из предобработанного датафрейма...")
    known_id_cols = ['sk_id_curr', 'sk_id_bureau', 'sk_id_prev', 'index'] # Базовые ID
    # Приводим все к нижнему регистру для сравнения
    excluded_cols_lower = [target_col.lower()] + [id_col.lower() for id_col in known_id_cols]
    features_list = [col for col in df.columns if col.lower() not in excluded_cols_lower]

    if not features_list:
         raise ValueError("Список признаков пуст после исключения target/ID. Проверьте входной df.")
    logging.info(f"  Используется {len(features_list)} признаков.")


    # --- 2. Подготовка данных ---
    logging.info("2. Подготовка данных...")
    if target_col not in df.columns: # <-- Используем аргумент target_col
        raise ValueError(f"Целевая колонка '{target_col}' не найдена в df!")

    X = df[features_list].copy()
    y = df[target_col].copy() # <-- Используем аргумент target_col

    # Обработка inf остается полезной на всякий случай
    inf_cols = []
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
             # Проверяем наличие inf без создания лишних копий
             if np.isinf(X[col].values).any():
                 inf_cols.append(col)
                 X[col] = X[col].replace([np.inf, -np.inf], np.nan)
                 X[col] = X[col].fillna(0) # Заполняем нулем NaN, появившиеся из inf
    if inf_cols:
        logging.warning(f"  Заменены inf на 0 в колонках: {inf_cols}")


    logging.info(f"  Размер данных для обучения (X, y): {X.shape}, {y.shape}")
    gc.collect()

    # --- 3. Настройка MLflow ---
    logging.info("3. Настройка MLflow...")
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        logging.info(f"  Эксперимент MLflow: {EXPERIMENT_NAME} на {MLFLOW_TRACKING_URI}")
    except Exception as e:
        logging.error(f"!!! ОШИБКА настройки MLflow: {e}. Пайплайн будет остановлен.")
        raise ConnectionError(f"Не удалось подключиться или установить эксперимент в MLflow: {e}")


    # --- 4. Настройка Валидации ---
    cv_strategy = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    logging.info(f"  Используется стратегия: {N_SPLITS} фолда StratifiedKFold.")

    # --- 5. Функции Objective для Optuna ---
    # (Оставляем как были)
    def objective_lgbm(trial):
        # ... (код objective_lgbm) ...
        params = {
            'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
            'n_estimators': 10000, # Early stopping
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'n_jobs': -1, 'seed': RANDOM_STATE, 'verbose': -1,
        }
        oof_auc_list = []
        for fold, (train_idx_iloc, val_idx_iloc) in enumerate(cv_strategy.split(X, y)):
            # Используем iloc для индексов из KFold, но loc для доступа к данным по оригинальным индексам
            X_train, y_train = X.iloc[train_idx_iloc], y.iloc[train_idx_iloc]
            X_val, y_val = X.iloc[val_idx_iloc], y.iloc[val_idx_iloc]

            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc',
                      callbacks=[lgb.early_stopping(100, verbose=False)])

            # Проверяем наличие best_score_, иначе считаем AUC вручную
            if hasattr(model, 'best_score_') and model.best_score_ and 'valid_0' in model.best_score_ and 'auc' in model.best_score_['valid_0']:
                 best_auc = model.best_score_['valid_0']['auc']
            else:
                 preds_val = model.predict_proba(X_val)[:, 1]
                 best_auc = roc_auc_score(y_val, preds_val)
            oof_auc_list.append(best_auc)
        return np.mean(oof_auc_list)

    def objective_xgb(trial):
        # ... (код objective_xgb) ...
        params = {
            'objective': 'binary:logistic', 'eval_metric': 'auc', 'booster': 'gbtree',
            'n_estimators': 10000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-3, 1.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'seed': RANDOM_STATE, 'n_jobs': -1,
            'early_stopping_rounds': 100
        }
        oof_auc_list = []
        for fold, (train_idx_iloc, val_idx_iloc) in enumerate(cv_strategy.split(X, y)):
             X_train, y_train = X.iloc[train_idx_iloc], y.iloc[train_idx_iloc]
             X_val, y_val = X.iloc[val_idx_iloc], y.iloc[val_idx_iloc]

             model = xgb.XGBClassifier(**params)
             model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
             preds_val = model.predict_proba(X_val)[:, 1]
             oof_auc_list.append(roc_auc_score(y_val, preds_val))
        return np.mean(oof_auc_list)

    def objective_catboost(trial):
        # ... (код objective_catboost) ...
        params = {
            'objective': 'Logloss', 'eval_metric': 'AUC', 'iterations': 10000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0), # Добавим, если нужно
            'random_seed': RANDOM_STATE, 'verbose': 0, 'early_stopping_rounds': 100
        }
        oof_auc_list = []
        for fold, (train_idx_iloc, val_idx_iloc) in enumerate(cv_strategy.split(X, y)):
            X_train, y_train = X.iloc[train_idx_iloc], y.iloc[train_idx_iloc]
            X_val, y_val = X.iloc[val_idx_iloc], y.iloc[val_idx_iloc]

            model = cb.CatBoostClassifier(**params)
            # Указываем use_best_model=True, чтобы модель использовала итерацию с лучшей метрикой на eval_set
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], use_best_model=True)
            preds_val = model.predict_proba(X_val)[:, 1]
            oof_auc_list.append(roc_auc_score(y_val, preds_val))
        return np.mean(oof_auc_list)

    # --- 6. Основной цикл HPO и CV ---
    logging.info(f"4. Запуск основного цикла HPO и CV ({N_SPLITS} фолда CV)...")

    models_to_run = {
        "LightGBM": {"class": lgb.LGBMClassifier, "objective": objective_lgbm},
        "XGBoost": {"class": xgb.XGBClassifier, "objective": objective_xgb},
        "CatBoost": {"class": cb.CatBoostClassifier, "objective": objective_catboost}
    }

    oof_predictions_df = pd.DataFrame(index=X.index) # Используем индекс X
    all_cv_metrics = {}
    all_best_params = {}

    # Общий Parent Run для всего пайплайна в MLflow
    with mlflow.start_run(run_name=f"Pipeline Run {N_SPLITS} folds {N_TRIALS_OPTUNA} trials") as parent_run:
        logging.info(f"MLflow Parent Run ID: {parent_run.info.run_id}")
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_splits_cv", N_SPLITS)
        mlflow.log_param("n_trials_optuna", N_TRIALS_OPTUNA)
        mlflow.log_param("random_state", RANDOM_STATE)
        # Логируем часть списка фичей для информации
        mlflow.log_param("features_list_sample", features_list[:min(len(features_list), 50)])


        for model_name, config in models_to_run.items():
            logging.info(f"--- Обработка модели: {model_name} ---")
            start_time_model = time.time()
            # Используем вложенные run для каждой модели
            with mlflow.start_run(run_name=f"{model_name} HPO+CV", nested=True) as child_run:
                logging.info(f"  MLflow Child Run ID ({model_name}): {child_run.info.run_id}")
                mlflow.set_tag("model_type", model_name)

                # 6.1 Подбор гиперпараметров
                logging.info(f"  Подбор гиперпараметров ({N_TRIALS_OPTUNA} итерации)...")
                study = optuna.create_study(direction='maximize',
                                            study_name=f"{model_name}_Optuna_Parent_{parent_run.info.run_id}",
                                            # Pruner можно добавить для ускорения
                                            # pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
                                            )
                objective_func = config["objective"]
                study.optimize(objective_func, n_trials=N_TRIALS_OPTUNA)

                best_params = study.best_params
                all_best_params[model_name] = best_params # Сохраняем для финального обучения
                best_auc_optuna = study.best_value
                logging.info(f"  Лучший AUC (Optuna, {N_SPLITS}-Fold CV): {best_auc_optuna:.5f}")
                mlflow.log_params({f"best_{k}": v for k, v in best_params.items()}) # Логируем лучшие параметры
                mlflow.log_metric("optuna_cv_best_auc", best_auc_optuna)

                # 6.2 Кросс-валидация с лучшими параметрами (для OOF и точных метрик CV)
                logging.info(f"  Кросс-валидация ({N_SPLITS} фолдов) с лучшими параметрами...")
                model_class = config["class"]
                cv_params = best_params.copy() # Берем лучшие из Optuna

                 # --- Корректируем/Добавляем параметры для CV ---
                 # Убедимся, что основные параметры для работы установлены
                if model_name == "LightGBM":
                     cv_params.update({'objective': 'binary', 'metric': 'auc', 'n_estimators': 10000,
                                      'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': -1})
                elif model_name == "XGBoost":
                     cv_params.update({'objective': 'binary:logistic', 'eval_metric': 'auc', 'n_estimators': 10000,
                                      'random_state': RANDOM_STATE, 'n_jobs': -1, 'early_stopping_rounds': 100})
                elif model_name == "CatBoost":
                     cv_params.update({'objective': 'Logloss', 'eval_metric': 'AUC', 'iterations': 10000,
                                      'random_seed': RANDOM_STATE, 'verbose': 0, 'early_stopping_rounds': 100})


                # Массив для OOF предсказаний этой модели
                oof_preds_proba_model = np.zeros(len(X))
                fold_metrics = {'auc': [], 'logloss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
                fold_best_iterations = [] # Сохраним кол-во итераций для LGBM/CAT

                for fold, (train_idx_iloc, val_idx_iloc) in enumerate(cv_strategy.split(X, y)):
                    X_train, y_train = X.iloc[train_idx_iloc], y.iloc[train_idx_iloc]
                    X_val, y_val = X.iloc[val_idx_iloc], y.iloc[val_idx_iloc]

                    model = model_class(**cv_params)

                    # Обучение и предсказание на фолде
                    if model_name == "LightGBM":
                        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc',
                                  callbacks=[lgb.early_stopping(100, verbose=False)])
                        best_iter = model.best_iteration_ if model.best_iteration_ else cv_params.get('n_estimators', 10000)
                        preds_proba = model.predict_proba(X_val, num_iteration=best_iter)[:, 1]
                        fold_best_iterations.append(best_iter)
                    elif model_name == "XGBoost":
                        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                        # best_iter = model.best_iteration # XGBoost хранит его здесь
                        preds_proba = model.predict_proba(X_val)[:, 1]
                        # fold_best_iterations.append(best_iter) # Можно добавить, если нужно
                    elif model_name == "CatBoost":
                        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], use_best_model=True)
                        best_iter = model.get_best_iteration() if model.get_best_iteration() else cv_params.get('iterations', 10000)
                        preds_proba = model.predict_proba(X_val)[:, 1]
                        fold_best_iterations.append(best_iter)

                    # Записываем OOF предсказания по правильным индексам iloc
                    oof_preds_proba_model[val_idx_iloc] = preds_proba
                    preds_class = (preds_proba > 0.5).astype(int)

                    # Расчет метрик для фолда
                    fold_metrics['auc'].append(roc_auc_score(y_val, preds_proba))
                    eps = 1e-15
                    preds_proba_clipped = np.clip(preds_proba, eps, 1 - eps)
                    fold_metrics['logloss'].append(log_loss(y_val, preds_proba_clipped))
                    fold_metrics['accuracy'].append(accuracy_score(y_val, preds_class))
                    fold_metrics['precision'].append(precision_score(y_val, preds_class, zero_division=0))
                    fold_metrics['recall'].append(recall_score(y_val, preds_class, zero_division=0))
                    fold_metrics['f1'].append(f1_score(y_val, preds_class, zero_division=0))

                    del X_train, y_train, X_val, y_val, model
                    gc.collect()

                # Расчет и логирование средних CV метрик для модели
                model_avg_metrics = {}
                logging.info(f"  Средние метрики CV ({N_SPLITS} фолда) для {model_name}:")
                for metric_name, values in fold_metrics.items():
                    avg_value = np.mean(values)
                    std_value = np.std(values)
                    logging.info(f"    {metric_name.capitalize()}: {avg_value:.5f} +/- {std_value:.5f}")
                    mlflow.log_metric(f"CV_{metric_name}_mean", avg_value)
                    mlflow.log_metric(f"CV_{metric_name}_std", std_value)
                    model_avg_metrics[metric_name] = avg_value
                all_cv_metrics[model_name] = model_avg_metrics

                # Логируем среднее количество итераций (если применимо)
                if fold_best_iterations:
                     avg_iterations = np.mean(fold_best_iterations)
                     mlflow.log_metric("CV_mean_best_iterations", avg_iterations)
                     logging.info(f"    Среднее кол-во итераций (early stopping): {avg_iterations:.0f}")

                # Сохраняем OOF предсказания в общий DataFrame
                # Важно использовать оригинальный индекс из X
                oof_predictions_df[f'{model_name}_oof_proba'] = pd.Series(oof_preds_proba_model, index=X.index)

            # Конец вложенного MLflow run для модели
            end_time_model = time.time()
            logging.info(f"  Время обработки {model_name}: {end_time_model - start_time_model:.2f} сек.")
            # Логируем время в parent run
            mlflow.log_metric(f"{model_name}_total_time_sec", end_time_model - start_time_model, run_id=parent_run.info.run_id)
        # --- КОНЕЦ ЦИКЛА ПО МОДЕЛЯМ ---

    # --- Обучение финальных моделей на всех данных X, y ---
    logging.info("5. Обучение финальных моделей на всех данных X, y...")
    final_trained_models = {}

    for model_name, best_params in all_best_params.items():
        logging.info(f"  Обучение финальной модели: {model_name}")
        model_class = models_to_run[model_name]["class"]
        final_params = best_params.copy() # Берем лучшие из Optuna

        # Корректируем параметры для финального обучения (убираем early stopping, метрики валидации)
        if model_name == "LightGBM":
             # Можно использовать n_estimators из CV или оставить большим
             # avg_iters = int(np.mean(fold_best_iterations)) # Если считали fold_best_iterations
             # final_params['n_estimators'] = max(avg_iters, 100) # Пример: берем среднее из CV
             final_params.update({'objective': 'binary', 'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': -1})
             final_params.pop('metric', None) # Убираем метрику, т.к. нет eval_set
             # n_estimators можно оставить из Optuna или задать на основе CV итераций
             if 'n_estimators' not in final_params: 
                final_params['n_estimators'] = 1000 # Запас

        elif model_name == "XGBoost":
             final_params.update({'objective': 'binary:logistic', 'random_state': RANDOM_STATE, 'n_jobs': -1})
             final_params.pop('eval_metric', None)
             final_params.pop('early_stopping_rounds', None)
             if 'n_estimators' not in final_params:
                final_params['n_estimators'] = 1000 # Запас

        elif model_name == "CatBoost":
             final_params.update({'objective': 'Logloss', 'random_seed': RANDOM_STATE, 'verbose': 0}) # Меньше логов
             final_params.pop('eval_metric', None)
             final_params.pop('early_stopping_rounds', None)
             if 'iterations' not in final_params: 
                final_params['iterations'] = 1000 # Запас


        # Создаем и обучаем финальную модель
        final_model = model_class(**final_params)
        start_fit_final = time.time()
        final_model.fit(X, y) # Обучаем на полных X, y
        end_fit_final = time.time()
        logging.info(f"  Модель {model_name} обучена за {end_fit_final - start_fit_final:.2f} сек.")

        final_trained_models[model_name] = final_model

        # Опционально: Логируем финальные модели в MLflow (может быть долго)
        # try:
        #     with mlflow.start_run(run_id=parent_run.info.run_id): # Логируем в Parent Run
        #          if model_name == "LightGBM":
        #              mlflow.lightgbm.log_model(final_model, f"final_{model_name}_model")
        #          elif model_name == "XGBoost":
        #              mlflow.xgboost.log_model(final_model, f"final_{model_name}_model")
        #          elif model_name == "CatBoost":
        #              mlflow.catboost.log_model(final_model, f"final_{model_name}_model")
        #          logging.info(f"  Финальная модель {model_name} залогирована в MLflow.")
        # except Exception as e:
        #     logging.error(f"  Ошибка логирования финальной модели {model_name} в MLflow: {e}")


    logging.info("--- Пайплайн завершен ---")
    # Возвращаем OOF-предикты, средние CV метрики и обученные финальные модели
    return oof_predictions_df, all_cv_metrics, final_trained_models


# --- Как использовать этот файл ---
# 1. Сохраните как Python файл (например, training_pipeline_returning_models.py)
# 2. В вашем Jupyter Notebook или другом скрипте:
#    a. Убедитесь, что все зависимости установлены (pandas, numpy, xgboost, lightgbm, catboost, sklearn, optuna, mlflow).
#    b. Импортируйте функцию: from training_pipeline_returning_models import run_training_pipeline
#    c. Загрузите и *предобработайте* ваш ДАТАФРЕЙМ (например, 'train_processed' из предыдущих шагов).
#       Он должен содержать и признаки, и целевую колонку.
#    d. Вызовите функцию, указав имя целевой колонки:
#       oof_results, cv_metrics, trained_models = run_training_pipeline(df=train_processed, target_col='target') # или 'TARGET'
#    e. ВАЖНО: Запускайте с АДЕКВАТНЫМИ настройками N_SPLITS и N_TRIALS_OPTUNA для реальной работы.
#    f. Теперь словарь 'trained_models' содержит ваши финальные модели:
#       final_lgbm = trained_models['LightGBM']
#       final_xgb = trained_models['XGBoost']
#       final_cb = trained_models['CatBoost']
#    g. Используйте эти final_... модели для предсказания на тестовом наборе ('test_processed').
#       Нужно будет выбрать колонки из test_processed, соответствующие тем, на которых обучались модели (features_list).