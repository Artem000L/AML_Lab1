import pandas as pd
from sklearn.preprocessing import StandardScaler

# Данные для обучения
train_data = pd.read_csv('train/train_data.csv')

# Cтандартизация данных
scaler = StandardScaler()
scaled_data = scaler.fit_transform(train_data)

# Сохранение предобработанных данных
pd.DataFrame(scaled_data, columns=train_data.columns).to_csv('train/train_data_scaled.csv', index=False)