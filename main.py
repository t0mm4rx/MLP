# coding: utf-8
import math
import random
from Network import Network

net = Network(2, [3], 1)

#for _ in range(10000000):
#    print(net.train([1, 0, 0], [0, 1]))

for _ in range(10000):
    num1 = math.floor(random.uniform(0, 2))
    num2 = math.floor(random.uniform(0, 2))
    output = [1]
    if (num1 == num2):
        output = [0]

    print(net.train([num1, num2], output))

# print(net.process([1, 1, 0]))
