import random

import numpy


class NeuralLayer:
    matrix = []
    amount_of_neurons = 0

    def __init__(self, rows, columns, minimum, maximum):
        self.amount_of_neurons = columns - 1
        matrix = []
        for i in range(rows):
            row = []
            for j in range(columns):
                row.append(random.uniform(minimum, maximum))
            matrix.append(row)
        self.matrix = numpy.array(matrix)
        if columns == 1:
            self.matrix = numpy.reshape(self.matrix, (rows, 1))

    def return_matrix(self):
        return self.matrix

    def set_matrix(self, matrix):
        self.matrix = matrix

    def return_amount_of_neurons(self):
        return self.amount_of_neurons
