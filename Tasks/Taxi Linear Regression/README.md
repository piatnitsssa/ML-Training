# Project Title: Taxi Trip Duration Prediction

## Data Sources
- **Raw Data:** [Download Raw Data](https://vk.com/doc331385305_679741102?hash=N7ULRVyk8RqTCw3Xu3elyLlcAfyXPG7a171xshEPPqD&dl=HYQWcE8QzYuTim2a1qvjItx1MOND3BCuZrov89c5Lmo&from_module=vkmsg_desktop)
- **Labeled Data:** [Download Labeled Data](https://vk.com/doc331385305_679741313?hash=1WBuH1Nvz1wCZHzpmMZ7WnZmmj1Jgnz3X8imkHyGeyk&dl=4q6GLergevpFGTwRZWUlbua4HExBGFXIBEUPKj8eIeD&from_module=vkmsg_desktop)

## Installation

### Dependencies
To install the required dependencies, use the following command:
pip install -r requirements.txt

### Running the Code
To run the main script, execute the following command:

python3 main.py

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

### Target Variable
I selected the target variable as the trip duration. This was calculated by subtracting dropoff_datetime from pickup_datetime. Then, I applied the `dt.total_seconds()` method from the pandas framework to convert the duration into seconds:

data['trip_duration'] = (data['dropoff_datetime'] - data['pickup_datetime']).dt.total_seconds()

After calculating the trip duration, I removed the dropoff_datetime and pickup_datetime columns since they were no longer needed.

Feature Engineering

	1.	Vendor ID:
I chose vendor_id as the first feature, which takes values 1 and 2. I transformed these values to a binary format by subtracting 1. I hypothesize that this feature may represent gender (e.g., 1 for male, 2 for female).


