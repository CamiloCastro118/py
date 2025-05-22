# -*- coding: utf-8 -*-
"""Created on May 20, 2025
@author: JPLOPEZ"""

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

# ============ ETIQUETAS ======================
# Codificar la columna objetivo ('Weather Type')
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(dataset['Weather Type'])

# Eliminar la columna objetivo del conjunto de caracteristicas
dataset = dataset.drop('Weather Type', axis=1)

# Convertir variables categoricas en variables dummy
dataset = pd.get_dummies(dataset)

# ============ NOMBRES DE LAS CLASES =========
class_names = label_encoder.classes_

# ============ COLORES PARA GRAFICAR =========
colors = plt.cm.get_cmap('Set1', len(class_names))

# ============ GRAFICO DE DISPERSION =========
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

# ============ PARTICION DEL DATASET =========
x_train, x_test, y_train, y_test = train_test_split(dataset, y, train_size=0.8, random_state=42)

# ============ INFO DE MUESTRAS ==============
print('Datos para entrenar:', x_train.shape[0])
print('Datos para validación:', x_test.shape[0])

# ============ GRAFICAR DATOS DE VALIDACION ==
plt.scatter(
    x_test['Atmospheric Pressure'],
    x_test['Humidity'],
    c='black',
    marker='x',
    label='Datos de Validacion'
)

# ============ ENTRENAR MODELO SVM ===========
model = SVC()
model.fit(x_train, y_train)

# ============ VALIDAR MODELO ================
y_pred = model.predict(x_test)

# ============ EVALUACION =====================
acc = accuracy_score(y_test, y_pred)
print('Acierto:', round(acc * 100, 2), '%')

plt.title(f"Clasificacion del Clima - Acierto: {round(acc * 100, 2)} %")
plt.legend()
plt.show()
