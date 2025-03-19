import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# CARGAR EL DATASET
dataset = pd.read_csv('seattle-weather.csv')

# VARIABLES
x = dataset['temp_max'].values.reshape(-1, 1)  # Variable independiente
y = dataset['temp_min'].values.reshape(-1, 1)  # Variable dependiente

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
plt.xlabel('Temperatura Máxima (temp_max)')
plt.ylabel('Temperatura Mínima (temp_min)')

# Graficar modelo lineal
y_pred_linear = linear_model.predict(x)
plt.plot(x, y_pred_linear, color='red', label='Modelo Lineal')

# graficar modelo polinomial
y_pred_poly = poly_model.predict(x_poly)
plt.plot(x, y_pred_poly, color='green', label='Modelo Polinomial ')

# Mostrar leyenda y grafico
plt.legend()
plt.show()

# PREDICCION
temp_max_a_predecir = [10, 15, 20, 25]  # Cambia estos valores para predecir con otras temperaturas maximas
for temp_max in temp_max_a_predecir:
    temp_min_lineal = linear_model.predict([[temp_max]])
    temp_min_polinomial = poly_model.predict(poly.transform([[temp_max]]))
    print(f"Prediccion con modelo lineal para temp_max={temp_max}°C: {temp_min_lineal[0][0]:.2f}°C")
    print(f"Prediccion con modelo polinomial para temp_max={temp_max}°C: {temp_min_polinomial[0][0]:.2f}°C")
    print("-" * 50)