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

    @property
    def biases(self):
        return self.__biases

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
        biases_grads[-1] = last_lay
        for i in range(len(weights_grads) - 2, -1, -1):
            bias_sum = sum(biases_grads[i + 1])
            biases_grads[i] = np.multiply(np.multiply(nodes[i + 1], np.subtract(1, nodes[i + 1])), bias_sum)
            for j in range(len(weights_grads[i])):
                for k in range(len(weights_grads[i][j])):
                    weights_grads[i][j][k] = np.multiply(np.sum(np.multiply(weights_grads[i + 1][k], np.subtract(1, nodes[i + 1][k]))), nodes[i][j])
        return weights_grads, biases_grads

    def learn(self, inputs, exp_outs, rate):
        nodes = self.calculate(*inputs)
        errors = self.error_grad(inputs, exp_outs)
        errors = np.divide(errors, len(inputs))
        print('Error: {}'.format(errors))

        grad = self.grad(nodes, errors)
        self.__weights = np.subtract(self.__weights, np.multiply(rate, grad[0]))
        self.__biases = np.subtract(self.__biases, np.multiply(rate, grad[1]))


net = Network(1,2,4,8,4,2,1)

for _ in range(10000):
    inp = np.random.standard_normal((1, ))
    exp_out = [int(inp > 0.5)]
    net.learn(inp, exp_out, 1)

print(net.calculate(0)[-1])
print(net.calculate(1)[-1])
print(net.calculate(2)[-1])
print(net.calculate(-0.4)[-1])