import numpy as np

class Network:

    def __init__(self):
        self.learning_rate = .1
        self.w1 = (np.random.random() - 0.5) * 2
        self.w2 = (np.random.random() - 0.5) * 2
        self.b = (np.random.random() - 0.5) * 2

    def guess(self, inputs):
        return self.sigmoid(self.w1 * inputs[0] + self.w2 * inputs[1] + self.b)

    def train(self, inputs):
        s = self.w1 * inputs[0] + self.w2 * inputs[1] + self.b
        pred = self.sigmoid(s)
        target = inputs[2]
        cost = self.cost(pred, target)
        delta = self.delta(pred, target)

        der = self.sigmoid_derivate(pred)

        self.w1 += delta * der * inputs[0]
        self.w2 += delta * der * inputs[1]
        self.b += delta * der
        return cost


    def cost(self, prediction, expected):
        return (expected - prediction) ** 2

    def cost_slope(self, prediction, expected):
        return 2 * (prediction - expected)

    def delta(self, prediction, expected):
        return self.cost_slope(prediction, expected) * -self.learning_rate

    def sigmoid (self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivate(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
