# Neurons
In this repository there are two models of neurons, the perceptron and the adaline.

## Perceptron Class

This is a module that contains an implementation for the Perceptron model neuron with stochastic gradient descent algorithm in python.

### Parameters:
- eta (float): Learn rate
- max_epocas (int): Maximum number of epochs.

### Methods:
- entrenar (fit).
  Parameters:
    - X (array[array[float]]): Training input data.
    - d (array[int]): Expected output values.
- entrenamiento_promediado (averaged fit).
  Parameters:
    - X (array[array[float]]): Training input data.
    - d (array[int]): Expected output values.
- entrenamiento_MIRA (MIRA variant of training).
  Parameters:
    - X (array[array[float]]): Training input data.
    - d (array[int]): Expected output values.
- evaluar (classify).
  Parameters:
    - X (array[array[float]]): Data to be classified.
  Returns:
    - An array with the classifications of the given data.

### Requirements:
- Python.
- numpy module.
 
### How to install this module
First, clone this repository with the git command below:
```
git clone https://github.com/PanquecaFuriosa/Neurons
```

### Examples
```
from .modelos import Perceptron

simple_perceptron = Perceptron(eta=0.5, max_epocas=10000)
simple_perceptron.entrenar(X, Y)
Y_pred1 = simple_perceptron.evaluar(X)

avg_perceptron = Perceptron(eta=0.5, max_epocas=10000)
avg_perceptron.entrenamiento_promediado(X, Y)
Y_pred2 = avg_perceptron.evaluar(X)

mira_perceptron = Perceptron(eta=0.5, max_epocas=10000)
mira_perceptron.entrenamiento_MIRA(X, Y)
Y_pred3 = avg_perceptron.evaluar(X)
```

## Adaline Class

This is a module that contains an implementation for the Adaline model neuron with stochastic gradient descent algorithm in python.

### Parameters:
- eta (float): Learn rate
- e (float): Error tolerance.
- max_epocas (int): Maximum number of epochs.

### Methods:
- entrenar (fit).
  Parameters:
    - X (array[array[float]]): Training input data.
    - d (array[int]): Expected output values.
- evaluar (classify).
  Parameters:
    - X (array[array[float]]): Data to be classified.
  Returns:
    - An array with the classifications of the given data.

### Requirements:
- Python.
- numpy module.
 
### How to install this module
First, clone this repository with the git command below:
```
git clone https://github.com/PanquecaFuriosa/Neurons
```

### Examples
```
from .modelos import Adaline

adaline = Adaline(eta=0.5, e=1e-9, max_epocas=10000)
adaline.entrenar(X, Y)
Y_pred = adaline.evaluar(X)
```
