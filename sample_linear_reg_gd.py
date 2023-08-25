import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Función para calcular h(x), la función hipótesis

import numpy as np

def h(params, sample):   
    """
    This evaluates a generic linear function h(x) with current parameters. h stands for hypothesis.
    
    Args:
        params (lst) a list containing the corresponding parameter for each element x of the sample
        sample (lst) a list containing the values of a sample 
        
    Returns:
        Evaluation of h(x)
    """
    pp_h = np.dot(params, sample)
    return pp_h


# Función para calcular el error cuadrático medio
def show_errors(params, samples, y):
    """Appends the errors/loss that are generated by the estimated values of h and the real value y
    
    Args:
        params (lst) a list containing the corresponding parameter for each element x of the sample
        samples (lst) a 2 dimensional list containing the input samples 
        y (lst) a list containing the corresponding real result for each sample
        
    Returns:
        Mean squared error
    """
    errors = np.square(np.dot(samples, params) - y)
    mean_s_error = np.mean(errors)
    return mean_s_error

# Implementación del algoritmo de Descenso de Gradiente
def GD(params, samples, y, alfa):

    """Gradient Descent algorithm 
	Args:
		params (lst) a list containing the corresponding parameter for each element x of the sample
		samples (lst) a 2 dimensional list containing the input samples 
		y (lst) a list containing the corresponding real result for each sample
		alfa(float) the learning rate
	Returns:
		temp(lst) a list with the new values for the parameters after 1 run of the sample set
	"""
    errors = np.dot(samples, params) - y
    gradient = np.dot(samples.T, errors) / len(samples)  # Gradiente utilizando operaciones matriciales
    params -= alfa * gradient
    return params

# Función para escalar las características de las muestras
def scaling(samples):
    """Normalizes sample values so that gradient descent can converge
	Args:
		params (lst) a list containing the corresponding parameter for each element x of the sample
	Returns:
		samples(lst) a list with the normalized version of the original samples
	"""

    return (samples - np.mean(samples, axis=0)) / np.max(samples, axis=0)

# Parámetros iniciales, muestras y valores reales



#New data set wine.data
columns= ["Class","Alcohol","Malic acid","Ash","Alcalinity of ash", "Magnesium","Total phenols","Flavanoids",
	  "Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue","OD280/OD315 of diluted wines","Proline  "]
df = pd.read_csv('wine.data',names = columns)

df = df[["Class","Alcohol","Malic acid","Ash","Alcalinity of ash"]]
df = df.dropna()
X = (df.drop(['Class'], axis = 1)).to_numpy()
Y = df['Class'].to_numpy()
params = np.zeros(4)

#print(X.shape)
#print(Y.shape)
#print(X)


# Agregar una columna de unos al principio de las muestras y normalizar

scaled_samples = scaling(X)

alfa = 0.01
epochs = 0
errors = []

while True:
    old_params = params.copy()
    params = GD(params, scaled_samples, Y, alfa)
    error = show_errors(params, scaled_samples, Y)
    errors.append(error)
    epochs += 1
    if np.allclose(old_params, params) or epochs == 2:
        break

print("Final params:", params)

# Graficar el proceso de reducción del error a lo largo de las épocas
plt.plot(errors)
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Error Reduction over Epochs')
plt.show()
