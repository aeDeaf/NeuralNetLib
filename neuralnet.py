import numpy

from neurallayer import NeuralLayer


class NeuralNet:
    """:type: list[NeuralLayer]"""
    neural_net = []
    amount_of_layers = 0

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + numpy.exp((-1) * z))

    def der_sigmoid(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)

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
            layer_matrix = layer.return_matrix()  # Получаем матрицу слоя
            a = list_a[i]
            return_tuple = self.calculate_layer(layer_matrix, a)  # Вычисляем входные значения нейронов следующего слоя
            a = return_tuple[0]
            lines = a.shape[0]
            # a = numpy.reshape(a, (lines, 1))  # Костыль из-за бага в numpy
            a = numpy.insert(a, 0, 1, axis=0)  # Добавляем 1 из-за наличия bias unit
            z = return_tuple[1]
            list_a.append(a)
            list_z.append(z)
        a = list_a[len(list_a) - 1]
        a = a[1:]
        list_a[len(list_a) - 1] = a
        return (list_a, list_z)

    def back_prop(self, expected_results, forward_result):
        list_a = forward_result[0]
        list_z = forward_result[1]
        amount_of_a = len(list_a)
        last_a = list_a[amount_of_a - 1]
        little_delta = [last_a - expected_results]  # Находим первую маленькую delta
        for i in range(self.amount_of_layers - 2):
            layer_number = self.amount_of_layers - 1 - i  # Вычисляем номер слоя для которого мы будем вычислять delta
            layer_matrix = self.neural_net[layer_number - 1].return_matrix()  # Получаем матрицу слоя
            z = list_z[layer_number - 1]
            first = numpy.dot((numpy.transpose(layer_matrix)), little_delta[i])
            first = first[1:]
            delta = first * self.der_sigmoid(z)  # Вычисляем маленькую delta для слоя
            little_delta.append(delta)  # Добавляем найденную delta в список
        return little_delta

    def initial_big_delta(self, num):
        res = [0 for i in range(num)]
        return res

    def calculate_gradients(self, training_set):
        x = training_set[0]
        y = training_set[1]
        m = len(x)
        big_delta = self.initial_big_delta(self.amount_of_layers - 1)  # Заполняем список big_delta нулями
        for i in range(m):
            cur_x = x[i]
            cur_y = y[i]
            # print(cur_x)
            forward_result = self.forward_prop(cur_x)  # Делаем forward propagation
            list_a = forward_result[0]
            back_result = self.back_prop(cur_y, forward_result)  # Находим delta
            back_result.reverse()
            for j in range(len(big_delta)):
                a = numpy.transpose(list_a[j])
                big_delta[j] += numpy.dot(back_result[j], a)
            gradients = []
            for delta in big_delta:
                gradients.append(delta / m)
        return gradients

    def learn(self, training_set, max_iters, alpha):
        J = []
        for i in range(max_iters):
            gradients = self.calculate_gradients(training_set)
            for j in range(len(gradients)):
                matrix = self.neural_net[j].return_matrix()
                matrix -= alpha * gradients[j]
                self.neural_net[j].set_matrix(matrix)
            J.append(self.calculate_cost(training_set))
        return J

    def calculate_cost(self, training_set):
        x = training_set[0]
        y = training_set[1]
        m = len(x)
        matrix_a = self.create_matrix_a(m, x)
        matrix_y = self.create_y_matrix(y)

        temp_matrix = (-1 * matrix_y) * numpy.log(matrix_a) - (1 - matrix_y) * numpy.log(1 - matrix_a)
        J = (1 / m) * numpy.sum(temp_matrix)
        return J

    def create_y_matrix(self, y):
        matrix_y = numpy.array(y[0])
        for i in range(1, len(y)):
            matrix_y = numpy.append(matrix_y, y[i], axis=1)
        return matrix_y

    def create_matrix_a(self, m, x):
        res = self.forward_prop(x[0])
        list_a = res[0]
        a = list_a[len(list_a) - 1]
        matrix_a = a
        for i in range(1, m):
            res = self.forward_prop(x[i])
            list_a = res[0]
            a = list_a[len(list_a) - 1]
            matrix_a = numpy.append(matrix_a, a, axis=1)
        return matrix_a
