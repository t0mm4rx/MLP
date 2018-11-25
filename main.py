from Network import Network
import matplotlib.pyplot as plt
import numpy as np
import pyttsx3

data = [
    [4, 3, 0],
    [2, 2, 0],
    [3, 1, 0],
    [5, 3, 0],
    [3, 3, 0],
    [4, 2, 0],
    [1, 4, 0],
    [-1, 0, 1],
    [1, 0, 1],
    [0, -2, 1],
    [-2, 1, 1],
    [1, -1, 1],
    [1, 1, 1],
    [2, -1, 1]
]

to_guess = [3, 1]

net = Network()

costs = []
for _ in range(100000):
    index = np.random.randint(len(data))
    point = data[index]
    costs.append(net.train(point))

plt.plot(costs)
plt.show()

"""
res = net.guess(to_guess)
if (res > .5):
    color = "blue"
else:
    color = "red"

engine = pyttsx3.init()
engine.say(color)
engine.runAndWait()

for a in data:
    color = 'ro'
    if (a[2] == 1):
        color = 'bo'
    plt.plot(a[0], a[1], color)

plt.plot(to_guess[0], to_guess[1], 'yo')
plt.show()
"""
