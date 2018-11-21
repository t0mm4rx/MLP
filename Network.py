# coding: utf-8
import numpy as np
from Layer import Layer

class Network:

    def __init__ (self, inputs_n, hiddens, outputs):
        self.learning_rate = 0.1
        self.layers = []
        for i in range(len(hiddens)):
            if (i == 0):
                self.layers.append(Layer(inputs_n, hiddens[i]))
            else:
                self.layers.append(Layer(hiddens[len(hiddens) - 2], hiddens[i]))
        self.layers.append(Layer(hiddens[len(hiddens) - 1], outputs))

    # On donne une entrée au réseau qui réalise une prédiction
    def process(self, inputs):
        prev_output = inputs
        for l in self.layers:
            prev_output = l.process(prev_output)
        return prev_output

    # Même fonction que process mais on enegistre le résultat de tous les layers, pour la rétropropagation
    def process_all(self, inputs):
        prev_output = [len(self.layers) + 1]
        prev_output[0] = inputs
        for l in self.layers:
            prev_output.append(l.process(prev_output[len(prev_output) - 1]))
        return prev_output

    # On donne une entrée et la sortie associée, le réseau s'améliore
    def train(self, inputs, outputs):
        guess = self.process_all(inputs)
        # On calcule l'erreur de tous les layers
        error_output = np.subtract(outputs, guess[len(guess) - 1])
        error = error_output ** 2
        error *= np.sign(error_output)

        errors = [[]] * len(self.layers)
        errors[len(self.layers) - 1] = error
        for a in range(len(self.layers) - 1):
            layer = self.layers[len(self.layers) - a - 1]
            previous_layer = self.layers[len(self.layers) - a - 2]
            previous_error = [0] * previous_layer.neurons_n

            for i in range(previous_layer.neurons_n):
                e = 0
                for n in range(layer.neurons_n):
                    e += errors[len(self.layers) - a - 1][n] * layer.weights[n][i]
                previous_error[i] = np.sign(e) * e ** 2

            errors[len(self.layers) - a - 2] = previous_error

        # On corrige les poids et les bias
        for a in range(len(self.layers)):
            l = len(self.layers) - a - 1
            layer = self.layers[l]
            for n in range(layer.neurons_n):
                # Les poids
                for i in range(layer.inputs_n):
                    delta = errors[l][n] * -1
                    delta *= np.absolute(layer.weights[n][i])
                    delta *= self.unactivate(np.absolute(guess[l][i]))
                    delta *= self.learning_rate
                    layer.weights[n][i] += delta
                    layer.weights[n][i] = np.clip(layer.weights[n][i], -500, 500)
            # Les bias
            bias_error = 0
            for n in range(layer.neurons_n):
                bias_error += errors[l][n]
            bias_error = -1 * np.sign(bias_error) * bias_error ** 2
            layer.bias += bias_error * np.absolute(layer.bias) * self.learning_rate
            layer.bias = np.clip(layer.bias, -500, 500)


        return np.linalg.norm(error)


    def unactivate(self, n):
        return 1 * (1 - np.clip(n, -500, 500))
