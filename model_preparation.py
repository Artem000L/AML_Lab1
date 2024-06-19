import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Предобработанные данные для обучения
train_data = pd.read_csv('train/train_data_scaled.csv')

# Создание случайного признака
train_data['random_feature'] = np.random.randn(len(train_data))

# Обучение модели
model = LinearRegression()
model.fit(train_data.drop(columns='temperature'), train_data['temperature'])

# Сохранение обученной модель
joblib.dump(model, 'trained_model.pkl')