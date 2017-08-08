import unittest

# import numpy

from neurallayer import NeuralLayer


class NeuralLayerTest(unittest.TestCase):
    def setUp(self):
        self.layer = NeuralLayer(3, 2, -1, 1)

    def matrix_create_test(self):
        matrix = self.layer.return_matrix()
        print(matrix)
        shape = matrix.shape
        self.assertEqual(shape, (3, 2))
