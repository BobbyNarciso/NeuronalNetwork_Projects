# -*- coding: utf-8 -*-
"""
@author: Iván Vázquez, Juan Daniel Rosales y Sandra Gómez

Este script implementa una red de Hopfield para la recuperación de patrones numéricos, en este caso, dígitos del 0 al 9. 
Las redes de Hopfield son un tipo de red neuronal recurrente que almacena patrones y los recuerda mediante un proceso 
de convergencia. Esta red asume que los patrones son representados por vectores binarios, donde los valores 1 y -1 
indican la activación o inactivación de neuronas.

El código entrena una red de Hopfield utilizando conjuntos de patrones (dígitos del 0 al 9). 
Cada patrón es un vector que representa un dígito en una cuadrícula de 5x5. El proceso de entrenamiento calcula 
una matriz de pesos sinápticos (W) usando la regla de Hebb, la cual facilita que la red recuerde patrones específicos. 
Luego, se introduce ruido en cada patrón y se recupera usando el método de actualización asincrónica. 

El resultado muestra cómo la red puede corregir el ruido en los dígitos y recuperar la representación original de 
los patrones. Este enfoque es útil para comprender cómo las redes de Hopfield pueden aplicarse a problemas de memoria 
asociativa y recuperación de información.
"""


import numpy as np
import matplotlib.pyplot as plt

# Patrones para dígitos del 0 al 4
patterns_0_4 = np.array([
    [-1,  1,  1,  1, -1,
     1, -1, -1, -1,  1,
     1, -1, -1, -1,  1,
     1, -1, -1, -1,  1,
    -1,  1,  1,  1, -1],  # 0

    [-1, -1,  1, -1, -1,
     -1,  1,  1, -1, -1,
     -1, -1,  1, -1, -1,
     -1, -1,  1, -1, -1,
    -1,  1,  1,  1, -1],  # 1

    [ 1,  1,  1,  1, 1,
     -1, -1, -1, -1,  1,
    -1,  1,  1,  1, -1,
     1, -1, -1, -1, -1,
     1,  1,  1,  1,  1],  # 2

    [ -1,  1,  1,  1, -1,
    -1, -1, -1, -1,  1,
    -1,  1,  1,  1, -1,
    -1, -1, -1, -1,  1,
     -1,  1,  1,  1, -1],  # 3

    [ 1, -1, -1,  1, -1,
     1, -1, -1,  1, -1,
     1,  1,  1,  1,  1,
    -1, -1, -1,  1, -1,
    -1, -1, -1,  1, -1]   # 4
])

# Patrones corregidos para dígitos del 5 al 9, con pequeñas diferencias añadidas
patterns_5_9 = np.array([
    [ -1,  1,  1,  1,  1,
     1, -1, -1, -1, -1,
     -1,  1,  1,  1, -1,
    -1, -1, -1, -1,  1,
     1,  1,  1,  1, -1],  # 5

     [-1,  1,  -1,  -1,  -1,
      -1, 1, -1, -1, -1,
      -1,  1,  1,  1, -1,
      -1, 1, -1, -1,  1,
     -1,  1,  1, 1, -1],  # 6

    [ 1,  1,  1,  1,  1,
    -1, -1, -1,  1, -1,
    -1, -1,  1, -1, -1,
    -1,  1, -1, -1, -1,
    1, -1, -1, -1, -1],  # 7

    [ 1,  1,  1,  -1,  -1,
     1, -1, 1, -1,  -1,
    1,  1,  1,  -1, -1,
     1, -1, 1, -1,  -1,
     1,  1,  1, -1, -1],  # 8

    [-1,  -1,  1,  1, 1,
     -1, 1, -1, -1,  1,
    -1,  -1,  1,  1,  1,
    -1, -1, -1, -1,  1,
     -1,  -1, -1,  -1, 1]   # 9
])

# Función para entrenar la red de Hopfield con normalización adicional
def train_hopfield(patterns, normalization_factor=0.05):
    num_neurons = patterns.shape[1]
    W = np.zeros((num_neurons, num_neurons))
    for p in patterns:
        W += np.outer(p, p)
    W /= num_neurons
    np.fill_diagonal(W, 0)
    return W * normalization_factor

# Función para la actualización asincrónica
def hopfield_step_async(x, W):
    for i in range(len(x)):
        x[i] = np.sign(np.dot(W[i], x))
    return x

# Función para realizar la iteración de recuperación hasta converger
def recall_async(pattern, W, steps=50):
    x = pattern.copy()
    for _ in range(steps):
        x = hopfield_step_async(x, W)
    return x

# Entrenamos redes para ambos conjuntos de patrones
W_0_4 = train_hopfield(patterns_0_4)
W_5_9 = train_hopfield(patterns_5_9)

# Configurar gráficos para ambos conjuntos
fig, axes = plt.subplots(5, 6, figsize=(16, 12))
fig.suptitle("Red de Hopfield - Dígitos 0-4 y 5-9", fontsize=16)

# Mostrar dígitos 0-4 con ruido y su recuperación
for i, digit in enumerate(patterns_0_4):
    noisy_digit = digit.copy()
    noisy_indices = np.random.choice(len(noisy_digit), size=2, replace=False)
    noisy_digit[noisy_indices] = -noisy_digit[noisy_indices]
    recovered_digit = recall_async(noisy_digit, W_0_4)
    
    axes[i, 0].imshow(digit.reshape(5, 5), cmap="binary")
    axes[i, 0].set_title(f"Dígito {i} Original")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(noisy_digit.reshape(5, 5), cmap="binary")
    axes[i, 1].set_title(f"Dígito {i} Ruidoso")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(recovered_digit.reshape(5, 5), cmap="binary")
    axes[i, 2].set_title(f"Dígito {i} Recuperado")
    axes[i, 2].axis("off")

# Mostrar dígitos 5-9 con ruido y su recuperación
for i, digit in enumerate(patterns_5_9, start=5):
    noisy_digit = digit.copy()
    noisy_indices = np.random.choice(len(noisy_digit), size=1, replace=False)
    noisy_digit[noisy_indices] = -noisy_digit[noisy_indices]
    recovered_digit = recall_async(noisy_digit, W_5_9)
    
    axes[i-5, 3].imshow(digit.reshape(5, 5), cmap="binary")
    axes[i-5, 3].set_title(f"Dígito {i} Original")
    axes[i-5, 3].axis("off")

    axes[i-5, 4].imshow(noisy_digit.reshape(5, 5), cmap="binary")
    axes[i-5, 4].set_title(f"Dígito {i} Ruidoso")
    axes[i-5, 4].axis("off")

    axes[i-5, 5].imshow(recovered_digit.reshape(5, 5), cmap="binary")
    axes[i-5, 5].set_title(f"Dígito {i} Recuperado")
    axes[i-5, 5].axis("off")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
