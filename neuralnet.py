import numpy

from neurallayer import NeuralLayer


class NeuralNet:
    """:type : list[NeuralLayer]"""
    neural_net = []
    amount_of_layers = 0

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + numpy.exp((-1) * z))

    def __init__(self, design, minimum, maximum):
        self.neural_net = []
        l = len(design)
        self.amount_of_layers = l
        for i in range(l):
            print(i)
            if i != (l - 1):
                layer = NeuralLayer(design[i + 1], design[i] + 1, minimum, maximum)
            else:
                layer = NeuralLayer(1, design[i] + 1, 1, 1)
            self.neural_net.append(layer)
            print(layer.return_matrix().shape)

    def return_net(self):
        """:rtype : list[neural_layer]"""
        return self.neural_net

    def calculate_layer(self, matrix, x):
        a = numpy.dot(matrix, x)
        z = self.sigmoid(a)
        """:rtype : tuple[numpy.ndarray]"""
        return (a, z)

    def forward_prop(self, x):
        """:type : list[numpy.ndarray]"""
        list_a = [];
        """:type : list[numpy.ndarray]"""
        a = numpy.insert(x, [0], [1], axis=1)
        list_a.append(a);
        for i in range(self.amount_of_layers):
            layer = self.neural_net[i];
            layer_matrix = layer.return_matrix();

