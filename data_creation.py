import os
import numpy as np
import pandas as pd

# Создание папок train и test
os.makedirs('train', exist_ok=True)
os.makedirs('test', exist_ok=True)

# Генерация данных для обучения
train_data = pd.DataFrame({'temperature': np.random.normal(loc=20, scale=5, size=200)})
train_data.to_csv('train/train_data.csv', index=False)

# Генерация данных для тестирования
test_data = pd.DataFrame({'temperature': np.random.normal(loc=25, scale=6, size=50)})
test_data.to_csv('test/test_data.csv', index=False)