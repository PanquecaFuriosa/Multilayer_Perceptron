# Multilayer Perceptron

In this repository there is an implementation of the multilayer perceptron with gradient descent algorithm.

## Multilayer Perceptron Class

### Parameters:
- n (int): Number of neurons in hidden layers.
- eta (float): Learn rate.
- alpha (float): Momentum factor.
- eps (float): Error tolerance.
- k (float): Number of hidden layers.
- max_epocas (int): Maximum number of epochs.

### Methods:
- establecer_f (set functions to neurons in hidden layers and output layer).
  Parameters:
    - funcion_o (string): Function to be set in the neurons of hidden layers. It can be 'sigmoide', 'tanh' or 'relu'.
    - funcion_s (string): Function to be set in the neuron of output layer. It can be 'sigmoide', 'tanh' or 'relu'.
- entrenar (fit).
  Parameters:
    - X (array[array[float]]): Training input data.
    - d (array[int]): Expected output values.
- clasificar (classify).
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
git clone https://github.com/PanquecaFuriosa/Multilayer_Perceptron
```

### Examples
```
from .modelo import MLP

mlp = MLP(n = 5, eta=0.5, alpha=0.9, eps=1e-9, k = 1, max_epocas=10000)
mlp.establecer_f('relu', 'sigmoide')
mlp.entrenar(X, Y)
Y_pred = mlp.clasificar(X)
```
