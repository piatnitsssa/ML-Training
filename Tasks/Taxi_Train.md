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

Теперь у меня есть обученная модель. Следующим шагом сделаю оценку ее точности предсказания.





