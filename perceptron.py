# Importar las bibliotecas necesarias
import tensorflow as tf
import numpy as np

# Paso 2.2: Definir Entradas y Pesos
# En un perceptrón simple, las entradas están ponderadas por ciertos pesos.
# Definición de las entradas y pesos iniciales:

# Definir las entradas (features)
entradas = np.array([2.0, 3.0], dtype=float)

# Definir los pesos iniciales y el sesgo (bias)
pesos = tf.Variable([1.0, -1.0], dtype=float)
sesgo = tf.Variable(1.0, dtype=float)

# Paso 2.3: Construir el Perceptrón
# Ahora, construyamos el perceptrón combinando las entradas con los pesos
# y aplicando una función de activación. En este caso, utilizaremos la función escalón:

# Calcular la suma ponderada de las entradas y pesos más el sesgo
suma_ponderada = tf.reduce_sum(tf.multiply(entradas, pesos)) + sesgo

# Aplicar la función de activación (función escalón)
resultado = tf.where(suma_ponderada > 0.0, 1.0, 0.0)

# Paso 2.4: Inicializar Variables y Ejecutar el Grafo
# Antes de ejecutar el perceptrón, inicialicemos las variables en TensorFlow
# y creemos una sesión:

# Inicializar las variables
inicializador = tf.global_variables_initializer()

# Crear una sesión
with tf.Session() as sesion:
    sesion.run(inicializador)

    # Obtener el resultado del perceptrón
    salida = sesion.run(resultado)
    print("Salida del perceptrón:", salida)


