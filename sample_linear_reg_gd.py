import numpy  #numpy is used to make some operrations with arrays more easily
import pandas
import matplotlib.pyplot as plt



def h(params, sample):

	return numpy.dot(params,sample)

def show_errors(params, samples,y):

	error2 = numpy.dot(params,samples)-y
	new_error = numpy.power(error2,2)
	mean_error = numpy.mean(new_error)
	return mean_error



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
	##temp = list(params)
	error = numpy.dot(params,samples)-y
	gradiente = numpy.dot(samples.T,error)/len(samples)
	new_params -= alfa*gradiente
	return new_params
	
def scaling(samples):
	"""Normalizes sample values so that gradient descent can converge
	Args:
		params (lst) a list containing the corresponding parameter for each element x of the sample
	Returns:
		samples(lst) a list with the normalized version of the original samples
	"""
	mean_data = numpy.mean(samples,axis=0)
	std_data = numpy.mean(samples,axis=0)
	datos_escalados = (samples-mean_data)/std_data
	return datos_escalados

#  multivariate example trivial

params = numpy.zeros(6)
columns= ["Class","Alcohol","Malic acid","Ash","Alcalinity of ash", "Magnesium","Total phenols","Flavanoids",
	  "Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue","OD280/OD315 of diluted wines","Proline  "]
samples = pandas.read_csv(r"C:\Users\drago\Downloads\wine.data",names = columns)
##print(samples)

y = samples[["Class"]].to_numpy()
alfa =.03  #  learning rate
epochs = 0

scaled_sample = scaling(samples)
#print("datos escalados: ",scaled_sample)

__errors__= [];  #global variable to store the errors/loss for visualisation




while True:  #  run gradient descent until local minima is reached
	old_params= params.copy
	params= GD(params,scaled_sample, y ,alfa)
	error = show_errors(params,scaled_sample,y)
	__errors__.append(error)
	epochs += 1
	if numpy.allclose(old_params, params) or epochs == 1000:
		break
	print("Final Params",params)

	plt.plot(__errors__)
	plt.xlabel("Epochs")
	plt.ylabel("Mean Squared Error")
	plt.title("Error Reduction")
	plt.show()






