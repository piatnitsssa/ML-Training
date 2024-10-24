import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def train_and_evaluate_model(data_path):
    data = pd.read_csv(data_path)
    

    X = data.drop(columns=['trip_duration']) 
    y = data['trip_duration']  
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
 
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', label='Predicted')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Actual')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Linear Regression: Actual vs Predicted (MSE: {mse:.2f})')
    plt.legend()
    plt.show()
