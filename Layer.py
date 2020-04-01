import numpy as np
from utils import acitvation_functions, backward_activation_functions

class Layer():
    def __init__(self, layer_dims, l, activation='relu'):
        self.l = l
        self.W = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        self.b = np.zeros((layer_dims[l], 1))
        self.activation = activation

    def __str__(self):
        return f"W_{self.l} shape: {self.W.shape}\nb_{self.l} shape: {self.b.shape}\nActivation_{self.l}: {self.activation}\n"

    def linear_activation_forward(self, A_prev):
        """
        :returns: A, Z
        """
        Z = self.W @ A_prev + self.b
        A = acitvation_functions[self.activation](Z)

        return A, Z

    def linear_activation_backward(self, dA, cache):
        A_prev, Z = cache
        m = A_prev.shape[1]

        dZ = backward_activation_functions[self.activation](dA, Z)

        dA_prev = self.W.T @ dZ

        self.dW = dZ @ A_prev.T / m
        self.db = np.sum(dZ, axis=1, keepdims=True) / m

        return dA_prev 

    def update_params(self, learning_rate):
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db