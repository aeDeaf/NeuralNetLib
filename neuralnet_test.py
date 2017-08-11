import unittest
import numpy

from neuralnet import NeuralNet
from matplotlib import pyplot as plt


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

    def forward_prop_test(self):
        net = NeuralNet([3, 3, 3], -1, 1)
        net_layers = net.return_net()
        net_layers[0].set_matrix(numpy.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]))
        net_layers[1].set_matrix(numpy.array([[-1, 1, -1, 1], [-2, 2, -2, 2], [3, -3, 3, -3]]))
        net.set_net(net_layers)
        x = numpy.array([[0.5], [-0.5], [-0.7]])
        res = net.forward_prop(x)
        res_a = res[0]
        a = res_a[len(res_a) - 1]
        # a = a[1:]
        expected_res = numpy.array([[0.41089559], [0.3272766], [0.746644]])
        self.assertEqual(a.all(), expected_res.all())

    def back_prop_test(self):
        net = NeuralNet([1, 3, 1], -1, 1)
        net_layers = net.return_net()
        net_layers[0].set_matrix(numpy.array([[1, 1], [2, 2], [3, 3]]))
        net_layers[1].set_matrix(numpy.array([1, 2, 3, 4]))
        net_layers[1].set_matrix(numpy.reshape(net_layers[1].return_matrix(), (1, 4)))
        net.set_net(net_layers)
        x = numpy.array([[-3]])
        forward_res = net.forward_prop(x)
        res = net.back_prop(1, forward_res)
        # print(res)

    def calculate_gradient_test(self):
        net = NeuralNet([1, 3, 1], -1, 1)
        net_layers = net.return_net()
        net_layers[0].set_matrix(numpy.array([[1, 1], [2, 2], [3, 3]]))
        net_layers[1].set_matrix(numpy.array([1, 2, 3, 4]))
        net_layers[1].set_matrix(numpy.reshape(net_layers[1].return_matrix(), (1, 4)))
        net.set_net(net_layers)
        x = numpy.array([[-3]])
        res = net.calculate_gradients([[x], [1]])
        # print(res)

    def learn_test(self):
        net = NeuralNet([1, 3, 1], -1, 1)
        x = [[[-3]], [[2]], [[0]], [[-2]]]
        y = [[[1]], [[1]], [[0]], [[0]]]
        training_set = [x, y]
        J = net.learn(training_set, 5000, 0.5)
        plt.plot(J)
        plt.show()
        res = net.forward_prop([[-3]])
        res_a = res[0]
        a = res_a[len(res_a) - 1]
        print(a)
        print('-----------')
        res = net.forward_prop([[2]])
        res_a = res[0]
        a = res_a[len(res_a) - 1]
        print(a)
        print('-----------')
        res = net.forward_prop([[0]])
        res_a = res[0]
        a = res_a[len(res_a) - 1]
        print(a)
        print('-----------')
        res = net.forward_prop([[-2]])
        res_a = res[0]
        a = res_a[len(res_a) - 1]
        print(a)
        print('-----------')


if __name__ == 'main':
    unittest.main()
