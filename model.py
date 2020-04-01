import numpy as np
import matplotlib.pyplot as plt

from utils import compute_cost
from Layer import Layer

class DNN():
    def __init__(self, layer_dims):
        self.layers = []

        self.num_layers = len(layer_dims) - 1
        
        # Initialize all layers except last
        for l in range(1, self.num_layers): 
            self.layers.append(Layer(layer_dims, l, 'relu'))

        # Initialize the last layer
        self.layers.append(Layer(layer_dims, self.num_layers, 'sigmoid'))

        for layer in self.layers:
            print(str(layer))

    def forward_propagate(self, X):
        caches = []
        A = X

        for layer in self.layers:
            A_prev = A

            A, Z = layer.linear_activation_forward(A_prev)

            caches.append((A_prev, Z))


        return A, caches

    def backward_propagate(self, AL, Y, caches):
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        dA_prev = dAL
        for l in reversed(range(self.num_layers)):
            dA = dA_prev
            dA_prev = self.layers[l].linear_activation_backward(dA, caches[l])


    def train(self, X, Y, learning_rate=0.0075, num_iterations=3000, print_cost=False):
        costs = []

        for i in range(num_iterations):
            AL, caches = self.forward_propagate(X)

            cost = compute_cost(AL, Y)

            self.backward_propagate(AL, Y, caches)

            for layer in self.layers:
                layer.update_params(learning_rate)
            
            if print_cost and i % 100 == 0:
                print (f"Cost after iteration {i}: {cost}")
                costs.append(cost)


        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

    
    def accuracy(self, X, Y):
        A, _  = self.forward_propagate(X)

        A[A >= 0.5] = 1
        A[A <= 0.5] = 0
        
        total = float(A.shape[1])
        accuracy = float(np.sum(A == Y)) / total

        print(f"Accuracy: {accuracy * 100}%")
