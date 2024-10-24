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
I selected the target variable as the trip duration. This was calculated by subtracting pickup_datetime from dropoff_datetime. Then, I applied the dt.total_seconds() method from the pandas framework to convert the duration into seconds:
```python
data['trip_duration'] = (data['dropoff_datetime'] - data['pickup_datetime']).dt.total_seconds()
```
After calculating the trip duration, I removed the dropoff_datetime and pickup_datetime columns since they were no longer needed.
## Feature Engineering
### vendor_id
I chose vendor_id as the first feature, which takes values 1 and 2. I transformed these values to a binary format by subtracting 1. I hypothesize that this feature may represent gender (e.g., 1 for male, 2 for female)
```python
data['vendor_id'] = data['vendor_id'] - 1
```
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

You could also use coordinates to create a categorical feature, meaning that based on the coordinates, you could define areas and predict the trip duration for each area. Each area may have its own characteristics: road traffic, road quality, weather conditions, and specific traffic rules in a particular state.

# Learning 

## Training and Evaluating the Model

In this section, I will train and evaluate a linear regression model to predict the duration of a taxi trip. The steps involve loading the dataset, splitting the data into training and test sets, fitting a linear regression model, and evaluating the performance using the Mean Squared Error (MSE).

### Code Breakdown

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def train_and_evaluate_model(data_path):
    # Load the dataset
    data = pd.read_csv(data_path)
    
    # Split the data into features (X) and target (y)
    X = data.drop(columns=['trip_duration'])  # Features (excluding target variable)
    y = data['trip_duration']  # Target variable (trip duration)
    
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict the trip duration on the test data
    y_pred = model.predict(X_test)
    
    # Calculate the Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    
    # Plot Actual vs Predicted trip duration
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', label='Predicted')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Actual')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Linear Regression: Actual vs Predicted (MSE: {mse:.2f})')
    plt.legend()
    plt.show()
```
## Explanation

1. Loading the dataset: The dataset is loaded from a CSV file using the pd.read_csv() function.
2. Feature and target selection: The features (X) are all columns except trip_duration, which is the target (y).
3. Train-test split: The data is split into training and test sets using an 80/20 ratio with the train_test_split() function from sklearn. The random_state=42 ensures reproducibility.
4. Model training: The LinearRegression() model from sklearn is instantiated and trained on the training set using model.fit().
5. Prediction: The model makes predictions on the test data using model.predict().
6. Mean Squared Error (MSE): The MSE is computed using mean_squared_error() to evaluate the model’s performance, which measures the average squared difference between actual and predicted trip durations.
7. Plotting Actual vs Predicted: A scatter plot is generated comparing the actual and predicted values. The red dashed line represents the ideal line where predicted values exactly match the actual values. The closer the blue scatter points are to this line, the better the model’s predictions.

## Results
The linear regression model demonstrates reasonable results for part of the data, but there are cases with significant deviations, which may indicate insufficient model complexity or the presence of unaccounted factors affecting trip duration.

I believe that using MAE (Mean Absolute Error) would be more appropriate for evaluating this model. MSE shows larger values because there are several predictions that significantly deviate from the actual values. MAE is less strict, and for this task, we could focus less on the magnitude of the error and more on the number of accurate predictions.

![Result](https://i.imgur.com/RLmDyVO.png)

It seems to me that for the first week of my learning, it looks pretty good!







