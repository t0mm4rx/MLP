# coding: utf-8
import math
import random
from Network import Network

net = Network(2, [4, 2], 1)

#for _ in range(10000000):
#    print(net.train([1, 0, 0], [0, 1]))

for _ in range(10000):
    print(net.train([0, 1], [0]))

print()
print("Guess [0, 1]")
print(net.process([0, 1]))
