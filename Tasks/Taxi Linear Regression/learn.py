import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Функция для обучения модели и построения графика
def train_and_evaluate_model(data_path):
    # Загрузка данных
    data = pd.read_csv(data_path)
    
    # Разделение данных на признаки и цель
    X = data.drop(columns=['trip_duration'])  # Признаки
    y = data['trip_duration']  # Целевая переменная (длительность поездки)
    
    # Разделение на тренировочные и тестовые данные
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Обучение модели линейной регрессии
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Предсказание для тестовых данных
    y_pred = model.predict(X_test)
    
    # Расчет MSE
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    
    # Построение графика "Фактические значения vs Предсказанные"
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', label='Predicted')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Actual')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Linear Regression: Actual vs Predicted (MSE: {mse:.2f})')
    plt.legend()
    plt.show()