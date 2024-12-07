import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import numpy as np


def predict(filename):
    df = pd.read_csv(filename)
    # Создаём массив, где находим изменения с 0 на 1
    change_indices = (df['FM_1.1_A'] == 1) & (df['FM_1.1_A'].shift(1) == 0)

    # Выбираем строки, где произошло изменение
    result_df = df[change_indices | change_indices.shift(-1).fillna(False)]


    BIG_result_df = pd.DataFrame()



    for i in range(1, 3):

            gold_ranges = [f'Ni_1.{i}C_min', f'Ni_1.{i}C_max', f'Cu_1.{i}C_min', f'Cu_1.{i}C_max']

            X = result_df[result_df[f'FM_1.{i}_A'] == 1].drop(columns = gold_ranges + ['MEAS_DT'])  # Признаки

            y = result_df[result_df[f'FM_1.{i}_A'] == 1][gold_ranges] # Эталонные диапазоны

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
            
            model.fit(X_train, y_train)

            X = df.drop(columns = gold_ranges + ['MEAS_DT'])
            y_pred = model.predict(X)

            y_pred_df = pd.DataFrame(y_pred, columns=gold_ranges)
            BIG_result_df = pd.concat([BIG_result_df, y_pred_df], axis=1)
            
            
            
            
    for i in range(1, 3):

            gold_ranges = [f'Cu_2.{i}T_min', f'Cu_2.{i}T_max']

            X = result_df[result_df[f'FM_2.{i}_A'] == 1].drop(columns = gold_ranges + ['MEAS_DT'])  # Признаки

            y = result_df[result_df[f'FM_2.{i}_A'] == 1][gold_ranges] # Эталонные диапазоны

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
            
            model.fit(X_train, y_train)

            X = df.drop(columns = gold_ranges + ['MEAS_DT'])
            y_pred = model.predict(X)

            y_pred_df = pd.DataFrame(y_pred, columns=gold_ranges)
            BIG_result_df = pd.concat([BIG_result_df, y_pred_df], axis=1)
            
            
            
            
    for i in range(1, 3):

            gold_ranges = [f'Cu_3.{i}T_min', f'Cu_3.{i}T_max']

            X = result_df[result_df[f'FM_3.{i}_A'] == 1].drop(columns = gold_ranges + ['MEAS_DT'])  # Признаки

            y = result_df[result_df[f'FM_3.{i}_A'] == 1][gold_ranges] # Эталонные диапазоны

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
            
            model.fit(X_train, y_train)

            X = df.drop(columns = gold_ranges + ['MEAS_DT'])
            y_pred = model.predict(X)

            y_pred_df = pd.DataFrame(y_pred, columns=gold_ranges)
            BIG_result_df = pd.concat([BIG_result_df, y_pred_df], axis=1)
            
            
            
            
    for i in range(1, 3):

            gold_ranges = [f'Ni_4.{i}T_min', f'Ni_4.{i}T_max', f'Ni_4.{i}C_min', f'Ni_4.{i}C_max']

            X = result_df[result_df[f'FM_4.{i}_A'] == 1].drop(columns = gold_ranges + ['MEAS_DT'])  # Признаки

            y = result_df[result_df[f'FM_4.{i}_A'] == 1][gold_ranges] # Эталонные диапазоны

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
            
            model.fit(X_train, y_train)

            X = df.drop(columns = gold_ranges + ['MEAS_DT'])
            y_pred = model.predict(X)

            y_pred_df = pd.DataFrame(y_pred, columns=gold_ranges)
            BIG_result_df = pd.concat([BIG_result_df, y_pred_df], axis=1)
            
            
            
            
    for i in range(1, 3):

            gold_ranges = [f'Ni_5.{i}T_min', f'Ni_5.{i}T_max', f'Ni_5.{i}C_min', f'Ni_5.{i}C_max']

            X = result_df[result_df[f'FM_5.{i}_A'] == 1].drop(columns = gold_ranges + ['MEAS_DT'])  # Признаки

            y = result_df[result_df[f'FM_5.{i}_A'] == 1][gold_ranges] # Эталонные диапазоны

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
            
            model.fit(X_train, y_train)

            X = df.drop(columns = gold_ranges + ['MEAS_DT'])
            y_pred = model.predict(X)

            y_pred_df = pd.DataFrame(y_pred, columns=gold_ranges)
            BIG_result_df = pd.concat([BIG_result_df, y_pred_df], axis=1)
            
            
            
            

    for i in range(1, 3):

            gold_ranges = [f'Ni_6.{i}T_min', f'Ni_6.{i}T_max', f'Ni_6.{i}C_min', f'Ni_6.{i}C_max']

            X = result_df[result_df[f'FM_6.{i}_A'] == 1].drop(columns = gold_ranges + ['MEAS_DT'])  # Признаки

            y = result_df[result_df[f'FM_6.{i}_A'] == 1][gold_ranges] # Эталонные диапазоны

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
            
            model.fit(X_train, y_train)

            X = df.drop(columns = gold_ranges + ['MEAS_DT'])
            y_pred = model.predict(X)

            y_pred_df = pd.DataFrame(y_pred, columns=gold_ranges)
            BIG_result_df = pd.concat([BIG_result_df, y_pred_df], axis=1)
            
            
            
            

    def aggregate_blocks(BIG_result_df, block_size=8):
        aggregated_df = BIG_result_df.copy()
        for col in aggregated_df.columns:
            aggregated_df[col] = aggregated_df[col].groupby(aggregated_df.index // block_size).transform('mean')
        
        return aggregated_df

    df_aggregated = aggregate_blocks(BIG_result_df, block_size=8)






    increment_map = {
        'Ni_1.1C_min': 0.1,
        'Ni_1.1C_max': 0.1,
        'Cu_1.1C_min': 0.1,
        'Cu_1.1C_max': 0.1,
        'Cu_2.1T_min': 0.01,
        'Cu_2.1T_max': 0.01,
        'Cu_3.1T_min': 0.05,
        'Cu_3.1T_max': 0.05,
        'Ni_4.1T_min': 0.01,
        'Ni_4.1T_max': 0.01,
        'Ni_4.1C_min': 0.05,
        'Ni_4.1C_max': 0.05,
        'Ni_5.1T_min': 0.01,
        'Ni_5.1T_max': 0.01,
        'Ni_5.1C_min': 0.05,
        'Ni_5.1C_max': 0.05,
        'Ni_6.1T_min': 0.01,
        'Ni_6.1T_max': 0.01,
        'Ni_6.1C_min': 0.05,
        'Ni_6.1C_max': 0.05,
    }

    y_pred = df_aggregated

    # Функция округления значений предсказанных диапазонов
    def apply_increments(predictions, increment_map):
        # Для каждого признака применяем соответствующее приращение
        for column in predictions.columns:
            increment = increment_map.get(column, 0.1)  # Если нет в map, по умолчанию 0.1
            predictions[column] = np.round(predictions[column] / increment) * increment
        return predictions

    y_pred_df = pd.DataFrame(y_pred, columns=[
        'Ni_1.1C_min', 'Ni_1.1C_max', 'Cu_1.1C_min', 'Cu_1.1C_max',
        'Cu_2.1T_min', 'Cu_2.1T_max',
        'Cu_3.1T_min', 'Cu_3.1T_max',
        'Ni_4.1T_min', 'Ni_4.1T_max', 'Ni_4.1C_min', 'Ni_4.1C_max',
        'Ni_5.1T_min', 'Ni_5.1T_max', 'Ni_5.1C_min', 'Ni_5.1C_max',
        'Ni_6.1T_min', 'Ni_6.1T_max', 'Ni_6.1C_min', 'Ni_6.1C_max',
    ])

    y_pred_rounded = apply_increments(y_pred_df, increment_map)





    BIG_result_df = pd.concat([df['MEAS_DT'], y_pred_rounded], axis=1)


    BIG_result_df.round(4)

    return BIG_result_df
