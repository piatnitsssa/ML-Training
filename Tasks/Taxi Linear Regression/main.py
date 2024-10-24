from train import preprocess_data
from learn import train_and_evaluate_model

input_csv_path = 'path to raw data'
output_csv_path = 'path to markup data'

preprocess_data(input_csv_path, output_csv_path)

train_and_evaluate_model(output_csv_path)
