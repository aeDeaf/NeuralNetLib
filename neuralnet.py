import numpy

from neurallayer import NeuralLayer


class NeuralNet:
    """:type: list[NeuralLayer]"""
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
            if i != (l - 1):
                layer = NeuralLayer(design[i + 1], design[i] + 1, minimum, maximum)
            else:
                layer = NeuralLayer(1, design[i] + 1, 1, 1)
            self.neural_net.append(layer)

    def return_net(self):
        """:rtype : list[neural_layer]"""
        return self.neural_net

    def set_net(self, net):
        """

        :type net: list[NeuralLayer]
        """
        self.neural_net = net

    def calculate_layer(self, matrix, x):
        """:rtype : tuple[numpy.ndarray]"""
        z = numpy.dot(matrix, x)
        a = self.sigmoid(z)
        return (a, z)

    def forward_prop(self, x):
        """
        :rtype : tuple[list[numpy.ndarray]]
        """
        list_a = []
        list_z = [None]
        a = numpy.insert(x, 0, 1, axis=0)
        list_a.append(a)
        for i in range(self.amount_of_layers - 1):
            layer = self.neural_net[i]
            layer_matrix = layer.return_matrix()
            print(layer_matrix)
            print(layer_matrix.shape)
            a = list_a[i]
            print(a)
            print(a.shape)
            return_tuple = self.calculate_layer(layer_matrix, a)
            a = return_tuple[0]
            lines = a.shape[0]
            a = numpy.reshape(a, (lines, 1))
            #print(a)
            #print(a.shape)
            a = numpy.insert(a, 0, 1, axis=0)
            #print(a)
            z = return_tuple[1]
            list_a.append(a)
            list_z.append(z)
            #print(list_a[1])
            print('-------------------')
        return (list_a, list_z)
