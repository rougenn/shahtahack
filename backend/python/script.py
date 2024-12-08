import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from catboost import CatBoostRegressor
import numpy as np


def predict(filename):
    df = pd.read_csv(filename)

    BIG_result_df = pd.DataFrame()
    # Время работы каждой флотомашины
    time = {
        6 : 45, 
        5 : 30,
        4 : 30, 
        3 : 30, 
        2 : 60,
        1 : 45, 
        0 : 0}

    delay = 0
    for i in range(1, 7):
            for j in range(1, 3):
                    
                    # Находим строки где индификатор AU меняется с 0 на 1
                    # Создаём булевый массив, где находим изменения с 0 на 1
                    change_indices = (df[f'FM_{i}.{j}_A'] == 1) & (df[f'FM_{i}.{j}_A'].shift(1) == 0)

                    # Выбираем строки, где произошло изменение
                    result_df = df[change_indices | change_indices.shift(-1).fillna(False)]

                    # "Эталонные" диапазоны для конретной флотомашины
                    if i == 1:
                            gold_ranges = [f'Ni_{i}.{j}C_min', f'Ni_{i}.{j}C_max', 
                                        f'Cu_{i}.{j}C_min', f'Cu_{i}.{j}C_max']
                    elif i == 2 or i == 3:
                            gold_ranges = [f'Cu_{i}.{j}T_min', f'Cu_{i}.{j}T_max']
                    else:
                            gold_ranges = [f'Ni_{i}.{j}T_min', f'Ni_{i}.{j}T_max', 
                                        f'Ni_{i}.{j}C_min', f'Ni_{i}.{j}C_max']

                    # Сдвиги
                    if i == 1:
                            except_ = []
                    elif i == 2:
                            # Для обработки второй флотомашины нужно сдвинуть значения во всех столбцах кроме этих:
                            except_ = ['Cu_oreth', 'Ni_oreth', 'Ore_mass', 'Mass_1', 'Dens_1', 'Vol_1']
                    elif i == 3:
                            except_ = ['Cu_oreth', 'Ni_oreth', 'Ore_mass', 'Mass_1', 'Dens_1', 'Vol_1', 
                                    'Mass_2', 'Dens_2', 'Vol_2', 'Cu_2F', 'Ni_2F'
                                    'Ni_1.1C_min', 'Ni_1.1C_max', 'Cu_1.1C_min', 'Cu_1.1C_max', 
                                    'Ni_1.2C_min', 'Ni_1.2C_max', 'Cu_1.2C_min', 'Cu_1.2C_max']
                    elif i == 4:
                            except_ = ['Cu_oreth', 'Ni_oreth', 'Ore_mass', 'Mass_1', 'Dens_1', 'Vol_1', 
                                    'Mass_2', 'Dens_2', 'Vol_2', 'Cu_2F', 'Ni_2F'
                                    'Ni_1.1C_min', 'Ni_1.1C_max', 'Cu_1.1C_min', 'Cu_1.1C_max', 
                                    'Ni_1.2C_min', 'Ni_1.2C_max', 'Cu_1.2C_min', 'Cu_1.2C_max', 
                                    'Mass_3', 'Dens_3', 'Vol_3', 'Cu_3F', 'Ni_3F'
                                    'Ni_3.1C_min', 'Ni_3.1C_max', 'Cu_3.1C_min', 'Cu_3.1C_max', 
                                    'Ni_3.2C_min', 'Ni_3.2C_max', 'Cu_3.2C_min', 'Cu_3.2C_max']
                            
                    elif i == 5:
                            except_ = ['Cu_oreth', 'Ni_oreth', 'Ore_mass', 'Mass_1', 'Dens_1', 'Vol_1', 
                                    'Mass_2', 'Dens_2', 'Vol_2', 'Cu_2F', 'Ni_2F'
                                    'Ni_1.1C_min', 'Ni_1.1C_max', 'Cu_1.1C_min', 'Cu_1.1C_max', 
                                    'Ni_1.2C_min', 'Ni_1.2C_max', 'Cu_1.2C_min', 'Cu_1.2C_max', 
                                    'Mass_3', 'Dens_3', 'Vol_3', 'Cu_3F', 'Ni_3F'
                                    'Ni_3.1C_min', 'Ni_3.1C_max', 'Cu_3.1C_min', 'Cu_3.1C_max', 
                                    'Ni_3.2C_min', 'Ni_3.2C_max', 'Cu_3.2C_min', 'Cu_3.2C_max', 
                                    'Mass_4', 'Dens_4', 'Vol_4', 'Cu_4F', 'Ni_4F'
                                    'Ni_4.1C_min', 'Ni_4.1C_max', 'Cu_4.1C_min', 'Cu_4.1C_max', 
                                    'Ni_4.2C_min', 'Ni_4.2C_max', 'Cu_4.2C_min', 'Cu_4.2C_max']
                    elif i == 6:
                            except_ = ['Cu_oreth', 'Ni_oreth', 'Ore_mass', 'Mass_1', 'Dens_1', 'Vol_1', 
                                    'Mass_2', 'Dens_2', 'Vol_2', 'Cu_2F', 'Ni_2F'
                                    'Ni_1.1C_min', 'Ni_1.1C_max', 'Cu_1.1C_min', 'Cu_1.1C_max', 
                                    'Ni_1.2C_min', 'Ni_1.2C_max', 'Cu_1.2C_min', 'Cu_1.2C_max', 
                                    'Mass_3', 'Dens_3', 'Vol_3', 'Cu_3F', 'Ni_3F'
                                    'Ni_3.1C_min', 'Ni_3.1C_max', 'Cu_3.1C_min', 'Cu_3.1C_max', 
                                    'Ni_3.2C_min', 'Ni_3.2C_max', 'Cu_3.2C_min', 'Cu_3.2C_max', 
                                    'Mass_4', 'Dens_4', 'Vol_4', 'Cu_4F', 'Ni_4F'
                                    'Ni_4.1C_min', 'Ni_4.1C_max', 'Cu_4.1C_min', 'Cu_4.1C_max', 
                                    'Ni_4.2C_min', 'Ni_4.2C_max', 'Cu_4.2C_min', 'Cu_4.2C_max', 
                                    'Mass_5', 'Dens_5', 'Vol_5', 'Cu_5F', 'Ni_5F'
                                    'Ni_5.1C_min', 'Ni_5.1C_max', 'Cu_5.1C_min', 'Cu_5.1C_max', 
                                    'Ni_5.2C_min', 'Ni_5.2C_max', 'Cu_5.2C_min', 'Cu_5.2C_max']

                    delay += (-time[i - 1] // 15)

                    for col in result_df.columns:
                            if col not in except_:
                                    result_df[col] = result_df[col].shift(delay)

                    X = result_df[result_df[f'FM_{i}.{j}_A'] == 1].drop(columns = gold_ranges + ['MEAS_DT'])
                    y = result_df[result_df[f'FM_{i}.{j}_A'] == 1][gold_ranges]
                    
                    #CatBoost с предсказанием нескольких переменных
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = MultiOutputRegressor(CatBoostRegressor(verbose=0))

                    # Обучение модели только с "эталонными" диапазонами
                    # Применение полученной зависимости ко всем записям
                    model.fit(X_train, y_train)
                    X = df.drop(columns = gold_ranges + ['MEAS_DT'])
                    y_pred = model.predict(X)

                    y_pred_df = pd.DataFrame(y_pred, columns=gold_ranges)
                    BIG_result_df = pd.concat([BIG_result_df, y_pred_df], axis=1)

    # Здесь учитывается ограничение по количеству смен диапазонов в 2 часа путем делений строк на блоки по 8 строк
    # и в столбце с суффиксом min берем минимальное значение из всех 8 строк, аналогично для максмимальных значений

    # Обработка столбцов с суффиксами 'min' и 'max'
    min_columns = [col for col in BIG_result_df.columns if col.endswith('min')]
    max_columns = [col for col in BIG_result_df.columns if col.endswith('max')]

    # Обработка столбцов с суффиксом 'min': заменяем значения на минимальные в блоках
    for col in min_columns:
        BIG_result_df[col] = BIG_result_df[col].groupby(BIG_result_df.index // 8).transform('min')

    # Обработка столбцов с суффиксом 'max': заменяем значения на максимальные в блоках
    for col in max_columns:
        BIG_result_df[col] = BIG_result_df[col].groupby(BIG_result_df.index // 8).transform('max')

    # Учет кратности для всех диапазонов
    increment_map = {
        'Ni_1.1C_min': 0.1,
        'Ni_1.1C_max': 0.1,
        'Cu_1.1C_min': 0.1,
        'Cu_1.1C_max': 0.1,

        'Ni_1.2C_min': 0.1,
        'Ni_1.2C_max': 0.1,
        'Cu_1.2C_min': 0.1,
        'Cu_1.2C_max': 0.1,

        'Cu_2.1T_min': 0.01,
        'Cu_2.1T_max': 0.01,

        'Cu_2.2T_min': 0.01,
        'Cu_2.2T_max': 0.01,

        'Cu_3.1T_min': 0.05,
        'Cu_3.1T_max': 0.05,

        'Cu_3.2T_min': 0.05,
        'Cu_3.2T_max': 0.05,

        'Ni_4.1T_min': 0.01,
        'Ni_4.1T_max': 0.01,
        'Ni_4.1C_min': 0.05,
        'Ni_4.1C_max': 0.05,

        'Ni_4.2T_min': 0.01,
        'Ni_4.2T_max': 0.01,
        'Ni_4.2C_min': 0.05,
        'Ni_4.2C_max': 0.05,

        'Ni_5.1T_min': 0.01,
        'Ni_5.1T_max': 0.01,
        'Ni_5.1C_min': 0.05,
        'Ni_5.1C_max': 0.05,

        'Ni_5.2T_min': 0.01,
        'Ni_5.2T_max': 0.01,
        'Ni_5.2C_min': 0.05,
        'Ni_5.2C_max': 0.05,

        'Ni_6.1T_min': 0.01,
        'Ni_6.1T_max': 0.01,
        'Ni_6.1C_min': 0.05,
        'Ni_6.1C_max': 0.05,

        'Ni_6.2T_min': 0.01,
        'Ni_6.2T_max': 0.01,
        'Ni_6.2C_min': 0.05,
        'Ni_6.2C_max': 0.05,
    }

    y_pred = BIG_result_df

    # Функция округления значений предсказанных диапазонов
    def apply_increments(predictions, increment_map):
        # Для каждого признака применяем соответствующее приращение
        for column in predictions.columns:
            increment = increment_map.get(column)
            predictions[column] = (np.round(predictions[column] / increment) * increment).round(int(-np.log10(increment)))
        return predictions

    y_pred_df = pd.DataFrame(y_pred, columns=[*list(increment_map.keys())])

    y_pred_rounded = apply_increments(y_pred_df, increment_map)

    BIG_result_df = pd.concat([df['MEAS_DT'], y_pred_rounded], axis=1)

    return BIG_result_df