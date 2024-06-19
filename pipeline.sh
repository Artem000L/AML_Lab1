#!/bin/bash

# Cоздание данных
python data_creation.py

# Предобработка данных
python model_preprocessing.py

# Обучение модели
python model_preparation.py

# Тестирования модели
python model_testing.py