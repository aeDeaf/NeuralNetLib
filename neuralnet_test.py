import unittest

from neuralnet import NeuralNet


class NeuralNetTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def net_create_test(self):
        net = NeuralNet([1, 3, 1], -1, 1)
        net_layers = net.return_net()
        l = len(net_layers)
        design = [1, 3, 1]
        for i in range(l):
            with self.subTest(i=i):
                layer = net_layers[i]
                self.assertEqual(layer.return_amount_of_neurons(), design[i], i)

    def calculate_layer_test(self):
        net = NeuralNet([1, 3, 1], -1, 1)
        net_layers = net.return_net()
        l = len(net_layers)
        for i in range(l):
            with self.subTest(name=i):
                layer = net_layers[i]
                matrix = layer.return_matrix()
                print(matrix.shape)
                res = net.calculate_layer(matrix, [1, 1])
                z = res[1]
                #print(i)
                self.assertEqual(z.shape, (3,), i)


if __name__ == 'main':
    unittest.main()
