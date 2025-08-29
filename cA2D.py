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
# Se elige un movimiento al azar en cada interaccion con el bucle hasta completar n pasos

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
x_data, y_data = caminata_paciente_2D(89)

# Grafico en 2D
plt.figure(dpi=150)
plt.plot(x_data, y_data, alpha=0.9, color="blue")
plt.scatter(x_data[0], y_data[0], color="green", s=50, label="Posicion inicial")
plt.scatter(x_data[-1], y_data[-1], color="red", s=50, label="Posicion final")

# Etiquetas
plt.title("Caminata aleatoria de un paciente en el hospital 2D")
plt.xlabel("Pasillo X (adelante/atras)")
plt.ylabel("Pasillo Y (izquierda/derecha)")
plt.legend()
plt.grid(True)
plt.axis("equal")  # Mantiene escala igual en ambos ejes
plt.show()

