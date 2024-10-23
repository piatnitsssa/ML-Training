import pandas as pd
import numpy as np
from geopy.distance import geodesic

# Функция для расчета расстояния между точками
def calculate_distance(row):
    pickup_coords = (row['pickup_latitude'], row['pickup_longitude'])
    dropoff_coords = (row['dropoff_latitude'], row['dropoff_longitude'])
    return geodesic(pickup_coords, dropoff_coords).meters

# Функция для обработки и разметки данных
def preprocess_data(input_csv_path, output_csv_path):
    # Загрузка данных
    data = pd.read_csv(input_csv_path)
    
    # Создание столбца с длительностью поездки
    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
    data['dropoff_datetime'] = pd.to_datetime(data['dropoff_datetime'])
    data['trip_duration'] = (data['dropoff_datetime'] - data['pickup_datetime']).dt.total_seconds()
    
    # Преобразование vendor_id в бинарный формат
    data['vendor_id'] = data['vendor_id'] - 1
    
    # Вычисление расстояния и средней скорости поездки
    data['travel_distance'] = data.apply(calculate_distance, axis=1)
    data['average_speed'] = data['travel_distance'] / data['trip_duration']
    
    # Удаляем ненужные столбцы
    data = data.drop(['pickup_datetime', 'dropoff_datetime', 'pickup_longitude', 'pickup_latitude',
                      'dropoff_longitude', 'dropoff_latitude', 'id', 'store_and_fwd_flag'], axis=1)
    
    # Удаляем строки с некорректным количеством пассажиров
    data = data[~data['passenger_count'].isin([0, 7, 8, 9])]
    
    # Преобразуем количество пассажиров в категории
    data = pd.concat((data, pd.get_dummies(data['passenger_count'].astype(int))), axis=1)
    data[[1, 2, 3, 4, 5, 6]] = data[[1, 2, 3, 4, 5, 6]].astype(int)
    data = data.drop(['passenger_count'], axis=1)
    
    # Переименовываем столбцы
    data = data.rename(columns={
        1: '1_passenger',
        2: '2_passenger',
        3: '3_passenger',
        4: '4_passenger',
        5: '5_passenger',
        6: '6_passenger'
    })
    
    # Сохранение разметанного датасета
    data.to_csv(output_csv_path, index=False)
    print(f"Data has been preprocessed and saved to {output_csv_path}")