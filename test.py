import numpy

from neuralnet import NeuralNet
from matplotlib import pyplot as plt

net = NeuralNet([3, 3, 3], -1, 1)
x = [[[1], [2], [3]], [[3], [5], [2]], [[4], [6], [8]]]
y = [[[1], [0], [0]], [[1], [0], [0]], [[0], [1], [0]]]
training_set = [x, y]
J = net.learn(training_set, 1000, 0.5)
plt.plot(J)
plt.show()
