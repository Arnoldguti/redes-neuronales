import tensorflow as tf
import numpy as np

# Disable eager execution
tf.compat.v1.disable_eager_execution()

# Definir las entradas (features)
entradas = np.array([2.0, 3.0], dtype=np.float32)  # Use np.float32 to match TensorFlow dtype

# Definir los pesos iniciales y el sesgo (bias)
pesos = tf.Variable([1.0, -1.0], dtype=np.float32)  # Use np.float32 to match TensorFlow dtype
sesgo = tf.Variable(1.0, dtype=np.float32)  # Use np.float32 to match TensorFlow dtype

# Calcular la suma ponderada de las entradas y pesos más el sesgo
suma_ponderada = tf.reduce_sum(tf.multiply(entradas, pesos)) + sesgo

# Aplicar la función de activación (función escalón)
resultado = tf.where(suma_ponderada > 0.0, 1.0, 0.0)

# Crear una sesión
with tf.compat.v1.Session() as sesion:
    # Inicializar las variables
    sesion.run(tf.compat.v1.global_variables_initializer())
    
    # Obtener el resultado del perceptrón
    salida = sesion.run(resultado)
    print("Salida del perceptrón:", salida)
