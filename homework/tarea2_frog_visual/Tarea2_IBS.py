# -*- coding: utf-8 -*-
"""
@authors: Iván Vázquez, Juan Daniel Rosales y Sandra Gómez

Simulación del comportamiento de una rana ante estímulos visuales utilizando integración numérica y esquema motor.

Descripción general:
---------------------
Este programa simula cómo una rana procesa estímulos visuales en su campo de visión, elige el estímulo más intenso, y calcula la dirección en la que probablemente lanzaría su lengua para capturar la presa asociada con dicho estímulo. La simulación involucra la modelación del potencial de membrana de las neuronas y un esquema motor para traducir el estímulo seleccionado en una dirección de movimiento.

Fórmulas y Cálculos:
---------------------
Integración numérica del potencial de membrana (método de Euler):
   La ecuación diferencial que modela el cambio en el potencial de membrana de una neurona es:
   
   \[\frac{dV}{dt} = -\frac{V}{\tau} + f(t)\]

   Donde:
   - V es el potencial de membrana.
   - tau es la constante de tiempo que regula la velocidad de cambio del potencial.
   - f(t) es el estímulo visual percibido por la neurona.
   
   El método de Euler aproxima esta ecuación mediante la siguiente fórmula:

   \[ V(t + dt) = V(t) + dt \left( -\frac{V(t)}{\tau} + f(t) \right)\]

   Esto nos permite calcular cómo varía el potencial de membrana a lo largo del tiempo, dada una intensidad de estímulo.

2. Selección de estímulo:
   El sistema selecciona el estímulo más intenso percibido por la rana. Si dos o más estímulos tienen la misma intensidad máxima, se seleccionan todos ellos. Se ajusta despues para no seleccionar un estimulo en caso de haber dos o más ganadores

3.Hísteresis:
   La histeresis es un fenómeno observado en sistemas dinámicos, donde la respuesta del sistema depende no solo de las entradas actuales, sino también de su estado previo. En el contexto neuronal, 
   este mecanismo permite que una neurona conserve su estado de activación, incluso cuando los estímulos presentes no son suficientemente fuertes para desencadenar una respuesta.

4. Esquema motor:
   Una vez que se ha seleccionado un estímulo o un conjunto de estímulos, el programa traduce la posición de ese estímulo en el espacio visual en una dirección de movimiento utilizando la fórmula:

   \[ \text{dirección} = i - \frac{(n-1)}{2}\]

   Donde:
   - i es el índice del estímulo seleccionado (su posición en el campo visual, numerada de 0 a 9).
   - n es el número total de estímulos (en este caso, n = 10 ).

   El resultado de esta fórmula es un valor que indica hacia dónde la rana movería su lengua:
   - Si el valor es negativo, la dirección está hacia la izquierda del campo visual.
   - Si el valor es positivo, la dirección está hacia la derecha del campo visual.
   - El valor 0 indica que la rana lanzaría su lengua directamente al centro.

Posibles resultados y su significado:
-------------------------------------
- Estímulo seleccionado(s):
  - Esta variable indica el índice del estímulo visual más intenso que la rana ha detectado. El índice 0 representa el estímulo más a la izquierda del campo visual y el índice 9 representa el estímulo más a la derecha.
  
- Dirección del movimiento:
  - Si la rana selecciona un único estímulo, la dirección se calcula según la fórmula descrita anteriormente.
  - Ejemplos:
    - Dirección = -4.5: El estímulo seleccionado está en la posición más a la izquierda (índice 0), y la rana movería su lengua fuertemente hacia la izquierda.
    - Dirección = 0: El estímulo está en el centro (índice 4 o 5), lo que indica que la rana lanzaría su lengua directamente hacia adelante.
    - Dirección = 4.5: El estímulo está en la posición más a la derecha (índice 9), lo que significa que la rana lanzaría su lengua hacia la derecha.

- Múltiples estímulos seleccionados:
  - Si varios estímulos tienen la misma intensidad máxima, el programa toma el promedio de sus posiciones para determinar una dirección de movimiento.
  - Por ejemplo, si los estímulos seleccionados son los índices 3 y 6, la dirección sería el promedio de sus posiciones: \((3 + 6)/2 - 4.5 = 1\), lo que indica que la rana lanzaría su lengua ligeramente hacia la derecha.

- Histeresís
  La neurona tiene un umbral (por ejemplo, 0.75) que debe superarse para activar un potencial de acción. 
  Si el estímulo máximo no alcanza el umbral, la neurona retiene su estado anterior (por ejemplo, 0.7).
"""

import numpy as np
import matplotlib.pyplot as plt
# Método de integración numérica de Euler
def euler_method(f, y0, t0, tf, dt, I_ext):
    n_steps = int((tf - t0) / dt)
    y = y0
    t = t0
    ys = [y0]
    ts = [t0]
    
    for _ in range(n_steps):
        y = y + f(y, t, I_ext) * dt
        t = t + dt
        ys.append(y)
        ts.append(t)
    
    return ts, ys

# Función que define el cambio del potencial de membrana en función de la corriente externa (influenciada por estímulos)
def membrane_potential(v, t, I_ext):
    tau = 10  # Constante de tiempo
    v_rest = -65  # Potencial de reposo
    return (-(v - v_rest) + I_ext) / tau

# Selección cuando hay estímulos iguales
def select_stimulus(stimuli):
    max_value = max(stimuli)
    max_indices = [i for i, val in enumerate(stimuli) if val == max_value]
    
    if len(max_indices) > 1:
        print("Estímulos seleccionados (iguales):", max_indices)
        return max_indices
    else:
        print(f"Estímulo seleccionado: {max_indices[0]}")
        return max_indices[0]

# Selección modificada para evitar ganador cuando hay estímulos iguales
def select_stimulus_no_winner(stimuli):
    max_value = max(stimuli)
    max_indices = [i for i, val in enumerate(stimuli) if val == max_value]
    
    if len(max_indices) > 1:
        print("No hay ganador, hay múltiples estímulos iguales.")
        return None  # No hay ganador
    else:
        print(f"Estímulo seleccionado: {max_indices[0]}")
        return max_indices[0]


# Gráfica de Hinton para visualizar la tasa de disparo basada en los estímulos
def hinton(matrix, title="Tasa de Disparo de Neuronas UF"):
    plt.figure()
    plt.title(title)
    plt.imshow(matrix, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.show()

# Gráfica de voltaje contra tiempo
def plot_voltage_vs_time(time, voltage, title="Potencial de Membrana vs Tiempo"):
    plt.figure()
    plt.plot(time, voltage)
    plt.title(title)
    plt.xlabel("Tiempo")
    plt.ylabel("Voltaje")
    plt.show()

# Histéresis
def hysteresis(stimuli, previous_output, threshold):
    max_stimulus = max(stimuli)
    if max_stimulus > threshold:
        return max_stimulus
    else:
        return previous_output

# Esquema motor que codifica la dirección
def motor_scheme(stimuli):
    selected_indices = select_stimulus(stimuli)
    n = len(stimuli)
    
    if isinstance(selected_indices, list):
        # Si hay más de un estímulo seleccionado, promediamos sus índices
        direction = sum([i - (n - 1) / 2 for i in selected_indices]) / len(selected_indices)
        print(f"Dirección promedio (dos o más estímulos ganadores): {direction}")
    else:
        direction = selected_indices - (n - 1) / 2
        print(f"Dirección (un estímulo ganador): {direction}")
    
    return direction
    
# ----- Simulaciones y Gráficas -----

# Simulación 1: Selección del estímulo mayor
print("\nSimulación 1: Selección del estímulo mayor")
stimuli_1 = [0.1, 0.5, 0.8, 0.6, 0.2, 0.7, 0.4, 0.3, 0.9, 0.2]
direction_1 = motor_scheme(stimuli_1)

# Crear una matriz de disparos de las neuronas en función de los estímulos
stimuli_matrix_1 = np.tile(stimuli_1, (10, 1))  # Repetimos los estímulos en 10 filas para una matriz 10x10
hinton(stimuli_matrix_1, title="Tasa de Disparo - Simulación 1")

# Simulación del potencial de membrana para el estímulo mayor
I_ext_1 = max(stimuli_1) * 10  # Tomamos el estímulo mayor para definir la corriente externa
t0, tf, dt = 0, 100, 0.1
v0 = -70  # Potencial inicial
time, voltage = euler_method(membrane_potential, v0, t0, tf, dt, I_ext_1)
plot_voltage_vs_time(time, voltage, title="Potencial de Membrana - Simulación 1")

# Simulación 2: Dos estímulos iguales
print("\nSimulación 2: Dos estímulos iguales")
stimuli_2 = [0.1, 0.5, 0.8, 0.9, 0.2, 0.7, 0.4, 0.3, 0.9, 0.2]
direction_2 = motor_scheme(stimuli_2)

# Crear una matriz de disparos de las neuronas en función de los estímulos
stimuli_matrix_2 = np.tile(stimuli_2, (10, 1))
hinton(stimuli_matrix_2, title="Tasa de Disparo - Simulación 2")

# Simulación del potencial de membrana para los estímulos iguales
I_ext_2 = max(stimuli_2) * 10  # Tomamos el estímulo mayor para definir la corriente externa
time, voltage = euler_method(membrane_potential, v0, t0, tf, dt, I_ext_2)
plot_voltage_vs_time(time, voltage, title="Potencial de Membrana - Simulación 2")

# Simulación 3: No seleccionar ganador con múltiples estímulos iguales
print("\nSimulación 3: No seleccionar ganador (múltiples estímulos iguales)")
stimuli_3 = [0.1, 0.5, 0.7, 0.4, 0.9, 0.8, 0.2, 0.9, 0.3, 0.5]
select_stimulus_no_winner(stimuli_3)

# Crear una matriz de disparos de las neuronas en función de los estímulos
stimuli_matrix_3 = np.tile(stimuli_3, (10, 1))
hinton(stimuli_matrix_3, title="Tasa de Disparo - Simulación 3")

# Simulación del potencial de membrana con histéresis
I_ext_3 = max(stimuli_3) * 10
time, voltage = euler_method(membrane_potential, v0, t0, tf, dt, I_ext_3)
plot_voltage_vs_time(time, voltage, title="Potencial de Membrana - Simulación 3")

# Simulación 4: Histéresis
print("\nSimulación 4: Histéresis")
previous_output = 0.7  # Estado anterior
threshold = 0.75
stimuli_4 = [0.1, 0.5, 0.73, 0.4, 0.6, 0.44, 0.2, 0.62, 0.3, 0.5]
max_stimulus = max(stimuli_4)

# Comprobamos si el estímulo supera el umbral
if max_stimulus > threshold:
    result_hysteresis = max_stimulus
else:
    result_hysteresis = previous_output

# Descripción del resultado
print(f"El valor más alto de los estímulos es {max_stimulus}, pero no supera el umbral ({threshold}). "
      f"Según el mecanismo de histéresis, si no se supera el umbral, el sistema conserva su estado anterior "
      f"({previous_output}), por lo que el resultado es {result_hysteresis}.")

# Crear una matriz de disparos de las neuronas en función de los estímulos
stimuli_matrix_4 = np.tile([0 if val <= threshold else val for val in stimuli_4], (10, 1))
hinton(stimuli_matrix_4, title="Tasa de Disparo - Simulación 4 (Histéresis)")

# Simulación del potencial de membrana con histéresis
# Ajustamos la corriente externa para mantener el estado anterior
I_ext_4 = 0  # No hay suficiente estímulo, el sistema conserva su estado anterior
# Funcion que representa el cambio del potencial de membrana
def potencialMembrana(voltaje, tiempo, corrienteExterna):
    tau = 10  # Constante de tiempo
    voltajeReposo = -65  # Potencial de reposo en mV
    cambioVoltaje = (-(voltaje - voltajeReposo) + corrienteExterna) / tau
    return cambioVoltaje

# Metodo de integracion numerica de Euler con histeresis y estimulos dinamicos
def metodoEulerHisteresis(funcion, voltajeInicial, tiempoInicial, tiempoFinal, pasoTiempo, estimulos, umbral, voltajeAnterior):
    numeroPasos = int((tiempoFinal - tiempoInicial) / pasoTiempo)
    voltaje = voltajeInicial
    tiempo = tiempoInicial
    voltajes = [voltajeInicial]
    tiempos = [tiempoInicial]
    
    # Aplicar los estimulos en cada paso de tiempo
    for i in range(numeroPasos):
        if i < len(estimulos):
            corrienteExterna = estimulos[i] * 10  # Convertir el estimulo en corriente externa
        else:
            corrienteExterna = 0
        
        if corrienteExterna > umbral:
            voltaje = voltaje + funcion(voltaje, tiempo, corrienteExterna) * pasoTiempo
            voltajeAnterior = voltaje  # Actualizamos el estado anterior
        else:
            voltaje = voltajeAnterior  # Mantener el estado anterior si no supera el umbral
        
        tiempo = tiempo + pasoTiempo
        voltajes.append(voltaje)
        tiempos.append(tiempo)
    
    return tiempos, voltajes

# Parametros de la simulacion
voltajeAnterior = -65  # Potencial de membrana inicial en reposo (estado anterior)
umbral = 0.5  # Umbral de histeresis ajustado
estimulos = [0.1, 0.5, 0.73, 0.4, 0.6, 0.44, 0.2, 0.62, 0.3, 0.5]  # Estimulos de la simulacion

# Variables de tiempo
tiempoInicial = 0
tiempoFinal = 100
pasoTiempo = 0.1

# Ejecutar la simulacion con histeresis estricta y estimulos dinamicos
tiempos, voltajes = metodoEulerHisteresis(potencialMembrana, voltajeAnterior, tiempoInicial, tiempoFinal, pasoTiempo, estimulos, umbral, voltajeAnterior)

# Funcion para graficar el voltaje en funcion del tiempo
def graficarVoltajeVsTiempo(tiempos, voltajes, titulo):
    plt.figure(figsize=(10, 6))
    plt.plot(tiempos, voltajes)
    plt.title(titulo)
    plt.xlabel('Tiempo (ms)')
    plt.ylabel('Voltaje (mV)')
    plt.grid(True)
    plt.show()

# Mostrar la grafica
graficarVoltajeVsTiempo(tiempos, voltajes, titulo="Potencial de Membrana - Simulacion #4 con Estimulos Dinamicos")


