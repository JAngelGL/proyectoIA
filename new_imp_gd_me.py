# Importa las bibliotecas necesarias
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define una función llamada 'descend' para realizar el descenso de gradiente en una regresión lineal.
# Esta función toma como entrada datos 'x' y 'y', así como los parámetros 'w' (pendiente) y 'b' (intercepto),
# y un valor de tasa de aprendizaje 'learning_rate'.
def descend(x, y, w, b, learning_rate):
    dldw = 0.0
    dldb = 0.0
    N = x.shape[0]

    # Calcula las derivadas parciales de la función de pérdida con respecto a 'w' y 'b'.
    for xi, yi in zip(x, y):
        dldw += -2 * xi * (yi - (w * xi + b))
        dldb += -2 * (yi - (w * xi + b))

    # Actualiza los parámetros 'w' y 'b' utilizando el descenso de gradiente.
    w = w - learning_rate * (1 / N) * dldw
    b = b - learning_rate * (1 / N) * dldb

    return w, b

# Define una función llamada 'scaling' para escalar las características de las muestras.
# Esta función toma como entrada un conjunto de muestras 'samples'.
def scaling(samples):
    return (samples - np.mean(samples, axis=0)) / np.max(samples, axis=0)

# Define las columnas del conjunto de datos y carga los datos desde un archivo CSV llamado 'wine.data'.
columns = ["Class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids",
           "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]
df = pd.read_csv('wine.data', names=columns)

# Selecciona las columnas 'Flavanoids' y 'Total phenols' del DataFrame y escala los datos en un rango de 0 a 1.
# (La escala se encuentra comentada y no se utiliza en este código).
df = df[["Flavanoids", "Total phenols"]]

# Elimina las filas que contienen valores faltantes y convierte los datos en matrices NumPy.
x = (df.drop(['Flavanoids'], axis=1)).to_numpy()
y = df['Flavanoids'].to_numpy()

# Inicializa los parámetros 'w' y 'b' para la regresión lineal, así como la tasa de aprendizaje y una lista para almacenar errores.
w = 0.0
b = 0.0
learning_rate = 0.01
error = []

# Realiza iteraciones para ajustar los parámetros 'w' y 'b' utilizando el descenso de gradiente.
for epoch in range(800):
    w, b = descend(x, y, w, b, learning_rate)
    yhat = w * x + b

    # Calcula la pérdida (error cuadrático medio) en cada iteración.
    loss = np.sum((y - yhat) ** 2) / x.shape[0]
    print(f'{epoch} loss is {loss}, parameters w:{w}, b:{b}')

    # Almacena la pérdida en la lista de errores.
    error.append(loss)

# Grafica la evolución de la pérdida a lo largo de las iteraciones.
plt.plot(range(800), error, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.show()

# Grafica los valores reales y la regresión lineal resultante.
plt.scatter(x, y, label='Valores reales')
plt.plot(x, w * x + b, color='red', label='Regresión lineal')
plt.xlabel('Total phenols')
plt.ylabel('Flavanoids')
plt.title('Regresión lineal y valores reales')
plt.legend()
plt.show()

