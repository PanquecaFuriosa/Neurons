# Perceptron Class

This is a module that contains an implementation for the Perceptron neuron with stochastic gradient descent algorithm in python.

## Parameters:
- eta (float): Learn rate
- max_epocas (int): Maximum number of epochs.

## Methods:
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

## Requirements:
- Python.
- numpy module.
 
## How to install this module
First, clone this repository with the git command below:
```
git clone https://github.com/PanquecaFuriosa/Simple_Perceptron
```

## Examples
```
from .Simple_Perceptron import Perceptron

simple_perceptron = Perceptron(eta=0.5, max_epocas=10000)
simple_perceptron.entrenar(X, Y)

avg_perceptron = Perceptron(eta=0.5, max_epocas=10000)
avg_perceptron.entrenamiento_promediado(X, Y)

mira_perceptron = Perceptron(eta=0.5, max_epocas=10000)
mira_perceptron.entrenamiento_MIRA(X, Y)
```
