# -*- coding: utf-8 -*-
"""Created on May 20, 2025
@author: camilo"""

# Libreri8as necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# ====================== Cargar y limpiar datos ======================

# cargar el dataset
dataset = pd.read_csv('weather_classification_data.csv')

# mostrar valores únicos para revisar inconsistencias
print("Valores en 'Weather Type':", dataset['Weather Type'].unique())

# mapear la columna Weather Type
mapeo_clases = {'Sunny': 0, 'Cloudy': 1, 'Rainy': 2}
y = dataset['Weather Type'].map(mapeo_clases)

# Eliminar filas con clases no reconocidas 
dataset = dataset[~y.isna()]
y = y.dropna().astype(int).values

# Eliminar columnas 
dataset = dataset.drop(['Cloud Cover', 'Season', 'Location', 'Weather Type'], axis=1)

# ====================== Visualizacion ======================

# Definir colores y nombres de clase
color = ['orange', 'gray', 'blue']
class_names = ['Sunny', 'Cloudy', 'Rainy']

# gráfico de dispersion
plt.figure(dpi=500)
for label in np.unique(y):
    idx = int(label)  # Asegurar índice entero
    plt.scatter(dataset[y == label]['Wind Speed'], 
                dataset[y == label]['Atmospheric Pressure'], 
                c=color[idx], label=class_names[idx])
plt.title('Clasificacion de Clima')
plt.xlabel('Wind Speed')
plt.ylabel('Atmospheric Pressure')
plt.legend()

# ====================== Modelo SVM ======================

# dividir en conjunto de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(dataset, y, train_size=0.8, random_state=42)

# Añadir al grafico los datos de validación
plt.scatter(x_test['Wind Speed'], x_test['Atmospheric Pressure'], c='black', label='Datos de validacion', marker='x')
plt.legend()

# Definir y entrenar el modelo SVM
modelo = SVC(kernel='linear')
modelo.fit(x_train, y_train)

# Realizar predicciones
y_pred = modelo.predict(x_test)

# ====================== Evaluación ======================

# precision del modelo
acc = accuracy_score(y_test, y_pred)
print('Precisión del modelo SVM:', round(acc * 100, 2), '%')

# matriz 
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz')
plt.xlabel('Prediccion')
plt.ylabel('Real')
plt.tight_layout()
plt.show()



