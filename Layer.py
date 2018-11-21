# coding: utf-8
import random
import math
import numpy as np

class Layer:

    # Inputs : nombre de neurones de la couche précédente, neurons: nombre de neurones de cette couche
    def __init__ (self, inputs_n, neurons_n):
        self.weights = []
        self.bias = 0
        self.neurons_n = neurons_n
        self.inputs_n = inputs_n
        self.random_neurons(inputs_n, neurons_n)

    # On crée nos neurones avec des poids à 0
    def random_neurons(self, inputs_n, neurons_n):
        for i in range(neurons_n):
            _weights = []
            self.bias = random.uniform(-1, 1)
            for j in range(inputs_n):
                _weights.append(random.uniform(-1, 1))
            self.weights.append(_weights)

    def process(self, inputs):
        output = []
        for i in range(self.neurons_n):
            sum = 0
            for j in range(self.inputs_n):
                sum += inputs[j] * self.weights[i][j]
            sum += self.bias
            output.append(self.activate(sum))
        return output

    def activate(self, input):
        return 1 / (1 + math.exp(-np.clip(input, -500, 500)))
