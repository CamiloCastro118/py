# -*- coding: utf-8 -*-
"""Created on May 19, 2025
@author: Camilo"""

# Librerias 

import matplotlib.pyplot as plt
import numpy as np
import random

# Se crean tres arreglos x, y, z que guardan la posicion en cada paso y iniciando en 0

def caminata_paciente(n):
    # Inicialización de coordenadas
    x, y, z = np.zeros(n), np.zeros(n), np.zeros(n)

# Posibles movimientos adelante-atras mover eje x 
# iozquierda-derecha mover eje y
# subir-bajar cambiar piso en eje Z

    movimientos = ["ADELANTE", "ATRAS", "IZQUIERDA", "DERECHA", "SUBIR", "BAJAR"]

# Bucle de simulacion
# Se elige un moviviento al azar en cada interaccion con el bucle hasta completar n pasos

    for i in range(1, n):
        paso = random.choice(movimientos)

        if paso == "ADELANTE":
            x[i] = x[i - 1] + 1
            y[i] = y[i - 1]
            z[i] = z[i - 1]
        elif paso == "ATRAS":
            x[i] = x[i - 1] - 1
            y[i] = y[i - 1]
            z[i] = z[i - 1]
        elif paso == "DERECHA":
            x[i] = x[i - 1]
            y[i] = y[i - 1] + 1
            z[i] = z[i - 1]
        elif paso == "IZQUIERDA":
            x[i] = x[i - 1]
            y[i] = y[i - 1] - 1
            z[i] = z[i - 1]
        elif paso == "SUBIR":
            x[i] = x[i - 1]
            y[i] = y[i - 1]
            z[i] = z[i - 1] + 1   # Subir un piso
        elif paso == "BAJAR":
            x[i] = x[i - 1]
            y[i] = y[i - 1]
            z[i] = z[i - 1] - 1   # Bajar un piso

    return x, y, z
# Simulacion caminata de n pasos
n = 100
x_data, y_data, z_data = caminata_paciente(n)

# Grafico en 3D
fig = plt.figure(dpi=150)
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_data, y_data, z_data, alpha=0.9, color="blue")
ax.scatter(x_data[-1], y_data[-1], z_data[-1], color="red", s=50, label="Posición final")

# Etiquetas
ax.set_title(f"Caminata aleatoria de un paciente en el hospital\nNumero de pasos: {n}")
ax.set_xlabel("Pasillo X (adelante/atrás)")
ax.set_ylabel("Pasillo Y (izquierda/derecha)")
ax.set_zlabel("Pisos (subir/bajar)")
ax.legend()
plt.show()
