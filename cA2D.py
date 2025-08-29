# -*- coding: utf-8 -*-
"""Created on May 19, 2025
@author: Camilo"""

# Librerias
import matplotlib.pyplot as plt
import numpy as np
import random

# Se crean dos arreglos x, y que guardan la posicion en cada paso y iniciando en 0
def caminata_paciente_2D(n):
    # Inicializacion de coordenadas
    x, y = np.zeros(n), np.zeros(n)

    # Posibles movimientos adelante-atras mover eje x 
    # izquierda-derecha mover eje y
    movimientos = ["ADELANTE", "ATRAS", "IZQUIERDA", "DERECHA"]

    # Bucle de simulacion
    for i in range(1, n):
        paso = random.choice(movimientos)

        if paso == "ADELANTE":
            x[i] = x[i - 1] + 1
            y[i] = y[i - 1]
        elif paso == "ATRAS":
            x[i] = x[i - 1] - 1
            y[i] = y[i - 1]
        elif paso == "DERECHA":
            x[i] = x[i - 1]
            y[i] = y[i - 1] + 1
        elif paso == "IZQUIERDA":
            x[i] = x[i - 1]
            y[i] = y[i - 1] - 1

    return x, y

# Simulacion caminata de n pasos
n_pasos = 89
x_data, y_data = caminata_paciente_2D(n_pasos)

# Grafico en 2D
plt.figure(dpi=150)
plt.plot(x_data, y_data, alpha=0.9, color="blue")
plt.scatter(x_data[0], y_data[0], color="green", s=50, label="Posición inicial")
plt.scatter(x_data[-1], y_data[-1], color="red", s=50, label="Posición final")

# Etiquetas
plt.title("Caminata aleatoria de un paciente en el hospital 2D")
plt.xlabel("Pasillo X (adelante/atrás)")
plt.ylabel("Pasillo Y (izquierda/derecha)")
plt.legend()
plt.grid(True)
plt.axis("equal")  # Mantiene escala igual en ambos ejes

# Mostrar nnmero de pasos cerca del punto final
plt.text(
    x_data[-1] + 0.3, y_data[-1] + 0.3, 
    f"N = {n_pasos} pasos", 
    fontsize=9, color="red", weight="bold"
)

plt.show()
