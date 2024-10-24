from train import preprocess_data
from learn import train_and_evaluate_model

input_csv_path = '/Users/vitaliipiatnitsa/Desktop/ML/training ML/HomwWork_1/Taxi Linear Reggresion/raw_data.csv'
output_csv_path = '/Users/vitaliipiatnitsa/Desktop/ML/training ML/HomwWork_1/Taxi Linear Reggresion/markup_data.csv'

preprocess_data(input_csv_path, output_csv_path)

train_and_evaluate_model(output_csv_path)
