import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# CARGAR EL DATASET
dataset = pd.read_csv('data.csv')

# VARIABLES
x = dataset['Height'].values.reshape(-1, 1)  # Variable independiente
y = dataset['Weight'].values.reshape(-1, 1)  # Variable dependiente

# MODELO LINEAL
linear_model = LinearRegression()
linear_model.fit(x, y)

# MODELO POLINOMIAL 
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)
poly_model = LinearRegression()
poly_model.fit(x_poly, y)

# GRAFICO DE DISPERSION
plt.figure(dpi=500)
plt.scatter(x, y, label='Datos', color='blue')
plt.title('Modelos Lineal y Polinomial')
plt.xlabel('Altura (Height)')
plt.ylabel('Peso (Weight)')

# Graficar modelo lineal
y_pred_linear = linear_model.predict(x)
plt.plot(x, y_pred_linear, color='red', label='Modelo Lineal')

# Graficar modelo polinomial
y_pred_poly = poly_model.predict(x_poly)
plt.plot(x, y_pred_poly, color='green', label='Modelo Polinomial ')

# Mostrar leyenda y grafico
plt.legend()
plt.show()

# PREDICCION
altura_a_predecir = 1.79  # Cambia este valor para predecir con otra altura
peso_lineal = linear_model.predict([[altura_a_predecir]])
peso_polinomial = poly_model.predict(poly.transform([[altura_a_predecir]]))

print(f"Prediccion con modelo polinomial para altura {altura_a_predecir} cm: {peso_polinomial[0][0]:.2f} kg")