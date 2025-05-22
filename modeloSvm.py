# -*- coding: utf-8 -*-
"""Created on May 20, 2025
@author: camilo"""

# ================= LIBRERIAS =================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ============ CARGAR EL DATASET =============
dataset = pd.read_csv('weather_classification_data.csv')

# etiquetas
# Codificar la columna objetivo ('Weather Type')
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(dataset['Weather Type'])

# Eliminar la columna objetivo del conjunto de caracteristicas
dataset = dataset.drop('Weather Type', axis=1)

# Convertir variables categoricas en variables dummy
dataset = pd.get_dummies(dataset)

# nombres de la clase
class_names = label_encoder.classes_

# colores grafico
colors = plt.cm.get_cmap('Set1', len(class_names))

#grafico de dispersion
plt.figure(dpi=500)
for label in np.unique(y):
    plt.scatter(
        dataset[y == label]['Atmospheric Pressure'],
        dataset[y == label]['Humidity'],
        c=np.array([colors(label)]),
        label=class_names[label]
    )
plt.title('Weather Dataset')
plt.xlabel('Atmospheric Pressure')
plt.ylabel('Humidity')
plt.legend()

# dataset
x_train, x_test, y_train, y_test = train_test_split(dataset, y, train_size=0.8, random_state=42)


print('Datos para entrenar:', x_train.shape[0])
print('Datos para validación:', x_test.shape[0])

# graficar
plt.scatter(
    x_test['Atmospheric Pressure'],
    x_test['Humidity'],
    c='black',
    marker='x',
    label='Datos de Validacion'
)

# entrenar modelo svm
model = SVC()
model.fit(x_train, y_train)

# validacion del modelo
y_pred = model.predict(x_test)

# evaluacion
acc = accuracy_score(y_test, y_pred)
print('Acierto:', round(acc * 100, 2), '%')

plt.title(f"Clasificacion del Clima - Acierto: {round(acc * 100, 2)} %")
plt.legend()
plt.show()
