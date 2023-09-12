# ProyectoIA - (Portafolio Implementación) Regresión Lineal con Descenso de Gradiente
José Ángel García López - A01275108

Implementacion del Gradiente decendiente sin usar Framework

Los documentos para presentar se hicieron con el data set de wine.data
Mi implementación a revisar está ubicada en el archivo new_imp_gd_me.py

## Conjunto de Datos

Este proyecto utiliza el conjunto de datos de vinos, que se encuentra en el archivo 'wine.data'. El conjunto de datos contiene información sobre diversas propiedades químicas de diferentes variedades de vinos. Aquí se enumeran las columnas del conjunto de datos:

- Class
- Alcohol
- Malic acid
- Ash
- Alcalinity of ash
- Magnesium
- Total phenols
- Flavanoids
- Nonflavanoid phenols
- Proanthocyanins
- Color intensity
- Hue
- OD280/OD315 of diluted wines
- Proline

El análisis y la regresión lineal se realizan específicamente en las variables 'Total phenols' y 'Flavanoids'.

## Descripción del Código

El código implementa una regresión lineal para ajustar una línea a un conjunto de datos de dos variables: 'Total phenols' y 'Flavanoids'. Aquí se describen los componentes principales del código:

- `descend(x, y, w, b, learning_rate)`: Una función que realiza el descenso de gradiente para ajustar los parámetros `w` (pendiente) y `b` (intercepto) de la regresión lineal.

- `scaling(samples)`: Una función para escalar las características de las muestras (comentada y no utilizada en este código).

- Carga de datos desde un archivo CSV llamado 'wine.data' en un DataFrame de Pandas.

- Preprocesamiento de datos: selección de las columnas relevantes, eliminación de filas con valores faltantes y conversión de datos a matrices NumPy.

- Inicialización de parámetros como `w`, `b` y la tasa de aprendizaje.

- Realización de iteraciones de descenso de gradiente para ajustar la regresión lineal.

- Cálculo y almacenamiento de la pérdida (error cuadrático medio) en cada iteración.

- Visualización de la evolución de la pérdida a lo largo de las iteraciones.

- Visualización de los valores reales y la regresión lineal resultante.

## Requisitos

Asegúrate de tener instaladas las siguientes bibliotecas antes de ejecutar el código:

- NumPy
- pandas
- Seaborn
- Matplotlib
