# Taxi Trip Duration Prediction

## Data Sources
- **Raw Data:** [Download Raw Data](https://vk.com/doc331385305_679741102?hash=N7ULRVyk8RqTCw3Xu3elyLlcAfyXPG7a171xshEPPqD&dl=HYQWcE8QzYuTim2a1qvjItx1MOND3BCuZrov89c5Lmo&from_module=vkmsg_desktop) (190,8 MB)
- **Labeled Data:** [Download Labeled Data](https://vk.com/doc331385305_679741313?hash=1WBuH1Nvz1wCZHzpmMZ7WnZmmj1Jgnz3X8imkHyGeyk&dl=4q6GLergevpFGTwRZWUlbua4HExBGFXIBEUPKj8eIeD&from_module=vkmsg_desktop) (83 MB)

## Installation

### Dependencies
To install the required dependencies, use the following command:

```bash
pip install -r requirements.txt
```

## Running the Code

To run the main script, execute the following command:

```bash
python3 main.py
```
## Project Overview
This is my first mini-project, which was part of a course assignment. The task involved taking raw data, labeling it, and then training a linear regression model. The objective of the model is to predict the duration of a taxi trip based on the available data.

## Data Preparation

### Raw Data Structure

The raw data includes the following columns:
- id
- vendor_id
- pickup_datetime
- dropoff_datetime
- passenger_count
- pickup_longitude
- pickup_latitude
- dropoff_longitude
- dropoff_latitude
- store_and_fwd_flag

## Target Variable

I selected the target variable as the trip duration. This was calculated by subtracting dropoff_datetime from pickup_datetime. Then, I applied the dt.total_seconds() method from the pandas framework to convert the duration into seconds:
```python
data['trip_duration'] = (data['dropoff_datetime'] - data['pickup_datetime']).dt.total_seconds()
```
After calculating the trip duration, I removed the dropoff_datetime and pickup_datetime columns since they were no longer needed.

## Feature Engineering
### vendor_id

I chose vendor_id as the first feature, which takes values 1 and 2. I transformed these values to a binary format by subtracting 1. I hypothesize that this feature may represent gender (e.g., 1 for male, 2 for female)

### travel_distance

For the next feature, I calculated the distance from the pickup point to the drop-off point using the four columns: pickup_longitude, pickup_latitude, dropoff_longitude, and dropoff_latitude. I used the geopy library for this purpose.
```python
from geopy.distance import geodesic

def calculate_distance(row):
    pickup_coords = (row['pickup_latitude'], row['pickup_longitude'])
    dropoff_coords = (row['dropoff_latitude'], row['dropoff_longitude'])
    return geodesic(pickup_coords, dropoff_coords).meters

data['travel_distance'] = data.apply(calculate_distance, axis=1)
```
Although I could have used the Pythagorean theorem, I opted for this approach for accuracy.

### average_speed

With both the trip duration and travel distance features, I calculated the average speed:
```python
data['average_speed'] = data['travel_distance'] / data['trip_duration']
```

## Additional Features
### passenger_count

The store_and_fwd_flag column was removed since it contained a single value (N), which would not yield any useful features. The passenger_count column was retained, but I filtered out specific values (0, 7, 8, 9) as they are considered outliers:
```python
data = data[~data['passenger_count'].isin([0, 7, 8, 9])]
```
I then created 6 categories using one-hot encoding:
```python
data = pd.concat((data, pd.get_dummies(data['passenger_count'].astype(int))), axis=1)
```
After this, I converted the new boolean columns to integers:
```python
data[[1, 2, 3, 4, 5, 6]] = data[[1, 2, 3, 4, 5, 6]].astype(int)
```
I subsequently dropped the original passenger_count column and renamed the new columns for clarity:
```python
data = data.rename(columns={
    1: '1_passenger',
    2: '2_passenger',
    3: '3_passenger',
    4: '4_passenger',
    5: '5_passenger',
    6: '6_passenger'
})
```
## Final Adjustments

Lastly, I removed the id column as it was unnecessary for model training. At this point, I had 4 features and 1 target variable ready for training.

