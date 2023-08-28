
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def descend(x, y, w, b, learning_rate): 
    dldw = 0.0 
    dldb = 0.0 
    N = x.shape[0]
    # loss = (y-(wx+b)))**2
    for xi, yi in zip(x,y): 
       dldw += -2*xi*(yi-(w*xi+b))
       dldb += -2*(yi-(w*xi+b))
    
    # Actualiza los parametros
    w = w - learning_rate*(1/N)*dldw
    b = b - learning_rate*(1/N)*dldb
    return w, b 

# Función para escalar las características de las muestras
def scaling(samples):
    return (samples - np.mean(samples, axis=0)) / np.max(samples, axis=0)

#importa los datos del data frame
columns= ["Class","Alcohol","Malic acid","Ash","Alcalinity of ash", "Magnesium","Total phenols","Flavanoids",
	  "Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue","OD280/OD315 of diluted wines","Proline"]
df = pd.read_csv('wine.data',names = columns)

## Escala los datos en un valor de 0 a 1
#df = scaling(df[["Flavanoids","Total phenols"]])
df = df[["Flavanoids","Total phenols"]]

#Descarta una de las columnas y establece los ejes
df = df.dropna()
x = (df.drop(['Flavanoids'], axis = 1)).to_numpy()
y = df['Flavanoids'].to_numpy()

# Parameteros
w = 0.0 
b = 0.0 

learning_rate = 0.01
error=[]


# Realiza iteraciones
for epoch in range(800): 
    w,b = descend(x,y,w,b,learning_rate)
    yhat = w*x + b
    loss = np.divide(np.sum((y-yhat)**2, axis=0), x.shape[0]) 
    print(f'{epoch} loss is {loss}, paramters w:{w}, b:{b}')

    error.append(loss)

#Grafica de resultados
plt.plot(range(800), error, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch')
plt.show()



