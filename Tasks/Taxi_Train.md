# Model of Linear Regression

- Data [download](https://vk.com/doc331385305_679741102?hash=N7ULRVyk8RqTCw3Xu3elyLlcAfyXPG7a171xshEPPqD&dl=HYQWcE8QzYuTim2a1qvjItx1MOND3BCuZrov89c5Lmo&from_module=vkmsg_desktop) (190,8 MB)

```python
df = pd.read_csv('raw_data.csv')
```

```python
df.columns.to_list()
```

```bash
 'id',
 'vendor_id',
 'pickup_datetime',
 'dropoff_datetime',
 'passenger_count',
 'pickup_longitude',
 'pickup_latitude',
 'dropoff_longitude',
 'dropoff_latitude',
 'store_and_fwd_flag'
```

## Target

Мне надо обучить модель для прогнозирования длительности поездки. Для этого рассчитаю `trip_duration`

```python
df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['trip_duration'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds()
```

Я перевел время начала и окончания поездки в `datetime`, затем создал новую колонку `trip_duration`. Нашел длительность поездки путем вычитания времени высадки из времени посадки и это значение перевел в секунды, применив метод `dt.total_second()`.

## Features

Когда начал анализировать данные, я решил установить георгафические данные поездок, чтобы сформировать более точную выборку. С сайта `https://gadm.org` я скачал Geopackage для USA. Взяв координаты `pickup_longitude` и `pickup_latitude` я определил точку посадки и по ней определял штат поездки. Для данной задачи я использовал код ниже:

```python
import geopandas as gpd
from shapely.geometry import Point

gadm_gdf = gpd.read_file('gadm41_USA.gpkg', layer='ADM_ADM_1')

df['geometry'] = df.apply(lambda row: Point(row['pickup_longitude'], row['pickup_latitude']), axis=1)
gdf_trips = gpd.GeoDataFrame(df, geometry='geometry', crs=gadm_gdf.crs)

joined_gdf = gpd.sjoin(gdf_trips, gadm_gdf[['NAME_1', 'geometry']], how='left', predicate='within')

joined_gdf = joined_gdf.rename(columns={'NAME_1': 'state'})

df = joined_gdf.drop(columns=['geometry', 'index_right'])
```

После того как я определил штаты поездок, я получил следущие данные: 

```python
df['state'].value_counts()
```

```bash
state
New York                1457584
New Jersey                  681
Connecticut                  21
Pennsylvania                  5
District of Columbia          4
New Hampshire                 2
California                    2
Delaware                      1
Virginia                      1
North Carolina                1
Vermont                       1
Maryland                      1
Name: count, dtype: int64
```

Тут видно, что основная география поездок – `New York`. Поэтому именно для этого штата мы и будем проводить обучение модели, другие штаты я удалю из выборки:

```python
df = df[df['state'] == 'New York']
```
После очистки выборки `state` принимает только одно значение.

```python
df = df.drop(columns=['state'])
```
### pickup_day_of_week

```python
df['pickup_day_of_week'] = df['pickup_datetime'].dt.dayofweek
```

### pickup_month

```python
df['pickup_month'] = df['pickup_datetime'].dt.month
```

### time_pickup

0:00 будет соответствовать значение `time_pickup = 0`, а 23:59:59 значение `time_pickup = 86399`.

```python
df['time_pickup'] = (
    df['pickup_datetime'].dt.hour * 3600 +
    df['pickup_datetime'].dt.minute * 60 +
    df['pickup_datetime'].dt.second 
    )
```
###

### store_and_fwd_flag

Задержка передачи данных на сервер – может отражаться на фактической длительности поездки.

```python
df['store_and_fwd_flag'].value_counts()
```

```bash
store_and_fwd_flag
N    1450599
Y       8045
Name: count, dtype: int64
```

Приведу значения к бинарному виду. 

```python
df['store_and_fwd_flag'] = df['store_and_fwd_flag'].apply(lambda x: 1 if x == 'Y' else 0)
```

Теперь все значения `Y` заменяются на `1`, а `N` на `0`.

### vendor_id

Эта категория может означать компанию оказывающую услуги. Этот признак тоже может влиять на длительность поездки.

```python
df['vendor_id'].value_counts()
```

```bash
vendor_id
2    780302
1    678342
Name: count, dtype: int64
```

Приведу этот признак к бинарному виду.

```python
df['vendor_id'] -= 1
```

### passenger_count

Этот значение соответсвует количеству пассажиров в поездке.

```bash
passenger_count
1    1033540
2     210318
5      78088
3      59896
6      48333
4      28404
0         60
7          3
9          1
8          1
Name: count, dtype: int64
```
Мне не известны тарифы и максимальное количество пассажиров для тарифов, но я предположу, что максимально допутисмое количество пассажиров будет 6. Недопустимыми значениями будем считать `0,7,8,9`. Такие данные могли быть получены из за сбоя, этот фактор мы не учитываем, поэтому строки с этими значениями я удалю.

```python
 df = df[~df['passenger_count'].isin([0, 7, 8, 9])]
```

```bash
passenger_count
1    1033540
2     210318
5      78088
3      59896
6      48333
4      28404
Name: count, dtype: int64
```
### travel_distance | average_speed

Найдем расстояние и среднюю скорость поездки, используя координаты посадки и высадки.

```python
from geopy.distance import geodesic

def calculate_distance(row):
    pickup_coords = (row['pickup_latitude'], row['pickup_longitude'])
    dropoff_coords = (row['dropoff_latitude'], row['dropoff_longitude'])
    return geodesic(pickup_coords, dropoff_coords).meters

df['travel_distance'] = df.apply(calculate_distance, axis=1)
df['average_speed'] = df['travel_distance'] / df['trip_duration']
```
## Обучение модели 

Загружаю размеченные данные:

```python
df = pd.read_csv('path to data')
```

Фичи:

```python
X = df.drop('trip_duration', axis=1)
```

Таргет:

```python
y = df('trip_duration')
```
Разбиение данных:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
```

Модель Линейной регрессии:

```python
model = LinearRegression()
```

Обучение модели:

```python
model.fit(X_train, y_train)
```

Теперь у меня есть обученная модель. Следующим шагом сделаю оценку ее точности предсказания.

```python
fact_variable_target = y_test.clip(lower=0)  # Фактический таргет для набора тестовых данных | здесь тип данных pandas.DataFrame
predict_for_test_features = model.predict(X_test).clip(min=0)  # Предсказанное значение для тестового набора фичей |здесь тип данных numpay.ndarray
volume_test_data = len(y_test)  # Объем (количество строк), размер выборки.
mean_actual = y_test.mean() # Среднее значение таргета для тестового набора данных 
mean_predict = model.predict(X_test).mean() #Среднее значение предсказаний для тестового набора данных 

MSE = (((fact_variable_target - predict_for_test_features).pow(2)).sum() / volume_test_data) # Mean Squared Error среднеквадратичная ошибка
RMSE = np.sqrt(MSE) # Root Mean Squared Error корень средней квадратичной ошибки 
MAE = ((fact_variable_target - predict_for_test_features).abs()).sum() / volume_test_data # Mean absolute error средняя абсолютная ошибка
RMSLE = np.sqrt(((np.log1p(fact_variable_target) - np.log1p(predict_for_test_features)).pow(2)).sum() / volume_test_data) # Root Mean Squared Logarithmic Error Среднеквадратичная логарифическая ошибка

RSS_residual_sum_of_squares = ((fact_variable_target - predict_for_test_features).pow(2)).sum() #Сумма квадратов отклонений предсказаний
TSS_total_sum_of_squares = ((fact_variable_target - mean_actual).pow(2)).sum() # Сумма квадратов разности таргета от среднего значения таргета
R_squared = round(1 - (RSS_residual_sum_of_squares / TSS_total_sum_of_squares), 3) # Коэффициент детерминации



errors = fact_variable_target - predict_for_test_features # Новый pandas.core.series.Series с ошибками 
std_deviation_errors_predict_test = np.sqrt((errors.pow(2)).mean()) # Стандартное отклонение ошибок для таргета в тестовом наборе данных 

confidence_level = 0.95 # Уровень доверия 95% (p-value = 0.05)
z_value = 1.96 # Критическое значение z для 95% уровня доверия 

margin_of_errors = z_value * (std_deviation_errors_predict_test / np.sqrt(volume_test_data)) # Погрешность для предсказанний на тестовом наборе данных 

confidence_level_min_predict_test = mean_predict - margin_of_errors # Нижняя граница доверительного интервала для предсказанний на тестовом наборе данных
confidence_level_max_predict_test = mean_predict + margin_of_errors # Верхняя граница доверительного интервала для предсказанний на тестовом наборе данных

t_value_for_predict_test = abs(mean_predict / (std_deviation_errors_predict_test / np.sqrt(volume_test_data))) # Проверка значимости среднего значения предсказаний относительно дисперсии ошибок. 
#Чем дальше метрика от 0, тем более значимо отклонение.

print(f"MSE: {round(MSE, 3)}")
print(f"RMSE: {round(RMSE, 3)}")
print(f"MAE: {round(MAE, 3)}")
print(f"RMSLE: {round(RMSLE, 3)}")
print(f"Коэффициент детерминации (R^2): {R_squared} | Данная метрика показывает, что {R_squared * 100}% предсказаний объясняются моделью.")
print(f"Для предсказанных значений на тестовом наборе данных, доверительный интервал находится в промежутке {round(confidence_level_min_predict_test, 3)} – {round(confidence_level_max_predict_test, 3)}.")
print(f"Значимость среднего значения предсказаний относительно дисперсии ошибок: {t_value_for_predict_test}")
```

```bash
MSE: 36686950.905
RMSE: 6056.975
MAE: 389.846
RMSLE: 0.62
Коэффициент детерминации (R^2): 0.013 | Данная метрика показывает, что 1.3% предсказаний объясняются моделью.
Для предсказанных значений на тестовом наборе данных, доверительный интервал находится в промежутке 937.783 – 981.759.
Значимость среднего значения предсказаний относительно дисперсии ошибок: 85.55292930757557
```

Для того, чтобы более наглядно посмотреть на результаты, я построил следующий график.
![model_1](https://cdn.imgchest.com/files/my2pcbaq5z7.png)

На графике видно, что в основном поездки не превышают 1800с. (30мин).
Я считаю правильным установить минимальную продолжительность поездки, пусть она будет 180с. (3мин.)

```python
df = pd.read_csv('path to markup data')
df = df.drop('Unnamed: 0', axis=1)

def drop_rows_trip_duraton_300_1800(df):
    df = df[df['trip_duration'] <= 1800]
    df = df[df['trip_duration'] >= 180]
    return df

df = drop_rows_trip_duraton_300_1800(df)
```
У меня объем данных составляет `1457517` строк. После выставления интервла продолжительности поездки `300 - 1800` объем данных сократился до `1273040`.
Теперь данные представляют собой - поездки в штате New York продолжительностью от 3 до 30 минут. Основная часть поездок приходится именно на этот интервал.


## Features rush_hour 

Данная фича показывает, попадает ли поездка в часы пик в будние дни, или поездка совершенна в выходные.

```python
def rush_hour(row):
    if (7 * 3600 <= row['time_pickup'] <= 9.5 * 3600) or (16 * 3600 <= row['time_pickup'] <= 19 * 3600):
        return 1  # rush_hour
    else:
        return 0  # not rush_hour 
    
def control_day_of_week(row):
    if row['pickup_day_of_week'] < 5:
        return rush_hour(row)
    else:
        return 3  # weekend 


df['rush_hour'] = df.apply(control_day_of_week, axis=1)
```
## drop features store_and_fwd_flag

Этот признак отвечает за сбои. Строки, где этот признак принимает значение `1` я удалю. После этого и сам признак.

```python
def clean_store_and_fwd_flag(df):
    df = df[df['store_and_fwd_flag'] == 0]
    return df

```



