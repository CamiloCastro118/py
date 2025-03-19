import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# CARGAR EL DATASET
dataset = pd.read_csv('score.csv')

# VARIABLES
x = dataset['Hours'].values.reshape(-1, 1)  # Variable independiente
y = dataset['Scores'].values.reshape(-1, 1)  # Variable dependiente

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
plt.xlabel('Horas de Estudio (Hours)')
plt.ylabel('Puntajes (Scores)')

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
horas_a_predecir = [2.5, 5.0, 8.0, 9.5]  # valores para predecir tiempo de estudio y puntaje
for horas in horas_a_predecir:
    score_lineal = linear_model.predict([[horas]])
    score_polinomial = poly_model.predict(poly.transform([[horas]]))
    print(f"Prediccion con modelo lineal para {horas} horas: {score_lineal[0][0]:.2f}")
    print(f"Prediccion con modelo polinomial para {horas} horas: {score_polinomial[0][0]:.2f}")
    print("-" * 50)