import numpy as np
import math


class Network:
    def __init__(self, *layers):
        self.__weights = np.array([np.random.standard_normal((layers[i-1], layers[i])) for i in range(1, len(layers))])
        self.__biases = np.array([np.random.standard_normal(layers[i]) for i in range(1, len(layers))])
        self.__size = layers

    @staticmethod
    def logistic_func(x):
        return 1 / (1 + np.exp(-x))

    @property
    def weights(self):
        return self.__weights

    def calculate(self, *_input):
        output = np.array(_input)
        nodes = [output]
        for weights, biases in zip(self.__weights, self.__biases):
            output = np.matmul(output, weights)
            output = np.add(output, biases)
            output = np.vectorize(self.logistic_func)(output)
            nodes.append(output)
        return np.array(nodes)

    def difference_vector(self, _input, exp_out):
        act_out = self.calculate(*_input)[-1]
        exp_out = np.array(exp_out)
        return np.subtract(act_out, exp_out)

    def error_grad(self, _input, exp_out):
        return np.multiply(2, self.difference_vector(_input, exp_out))

    def error(self, _input, exp_out):
        return np.sum(np.square(self.difference_vector(_input, exp_out)))

    def grad(self, nodes, error_grad):
        last_lay = np.multiply(np.multiply(nodes[-1], np.subtract(1, nodes[-1])), error_grad)
        weights_grads = np.array([np.ones((self.__size[i-1], self.__size[i])) for i in range(1, len(self.__size))])
        biases_grads = np.array([np.ones((self.__size[i])) for i in range(1, len(self.__size))])
        for i in range(len(weights_grads[-1])):
            weights_grads[-1][i] = np.multiply(last_lay, nodes[-2][i])
        for i in range(len(weights_grads) - 2, -1, -1):
            for j in range(len(weights_grads[i])):
                for k in range(len(weights_grads[i][j])):
                    weights_grads[i][j][k] = np.multiply(np.sum(np.multiply(weights_grads[i + 1][k], np.subtract(1, nodes[i + 1][k]))), nodes[i][j])
        return weights_grads

    def learn(self, inputs, exp_outs, rate):
        assert len(inputs) == len(exp_outs)
        nodes = self.calculate(*inputs[0])
        errors = self.error_grad(inputs[0], exp_outs[0])
        for i in range(1, len(inputs)):
            np.add(nodes, self.calculate(*inputs[i]))
            np.add(errors, self.error_grad(inputs[i], exp_outs[i]))
        nodes = np.divide(nodes, len(inputs))
        errors = np.divide(errors, len(inputs))

        self.__weights = np.subtract(self.__weights, np.multiply(rate, self.grad(nodes, errors)))

net = Network(3,4,3)
for i in range(100):
    # print(net.weights)
    before_err = net.error([3,-60,0], [1,0.5,0.89])
    net.learn([[3,-60,0]], [[1,0.5,0.89]], 0.1)
    # print(net.weights)
    now_err = net.error([3,-60,0], [1,0.5,0.89])
    print('diff: {}'.format(now_err - before_err))
    print('curr err: {}'.format(now_err))
