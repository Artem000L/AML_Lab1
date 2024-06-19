import pandas as pd
import joblib
import numpy as np

# Предобученная модель
model = joblib.load('trained_model.pkl')

# Данные для тестирования
test_data = pd.read_csv('test/test_data.csv')

# Добавление случайного признака
test_data['random_feature'] = np.random.randn(len(test_data))

print(test_data)

# Предсказание
predictions = model.predict(test_data.drop(columns='temperature'))

print('Предсказанные значения:')
print(predictions)