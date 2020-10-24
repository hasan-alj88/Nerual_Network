import math
import numpy as np
from functools import reduce
import networkx as nx
from itertools import product
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from functools import reduce
import pandas as pd


@np.vectorize
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


@np.vectorize
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def consecutive(end: int):
    for x in range(end - 1):
        yield x, x + 1


def consecutive_item_pair(itr):
    for x, y in consecutive(len(itr)):
        yield itr[x], itr[y]


def consecutive_reverse(end: int):
    for x in range(end, 0, -1):
        yield x, x - 1


def consecutive_item_pair_reverse(itr):
    for x, y in consecutive_reverse(len(itr) - 1):
        yield itr[x], itr[y]


class NeuralNetwork:
    def __init__(self, input_size: int, output_size: int = 10):
        self._input_size = input_size
        self.layer_sizes = [input_size, input_size * 2, input_size * 2, output_size * 10, output_size * 10, output_size]
        self._weights = [np.random.rand(ln, lnm1) * 0.25 for ln, lnm1 in consecutive_item_pair(self.layer_sizes)]
        self._biases = [np.random.rand(L) * 0.25 for L in self.layer_sizes[1:]]
        self._neurons = [np.zeros(L) for L in self.layer_sizes[1:]]
        self.propagate(np.random.rand(input_size))

    @property
    def input_size(self):
        return self._input_size

    @property
    def number_of_layers(self):
        return len(self._weights)

    @property
    def output_size(self):
        return self.layer_sizes[-1]

    def propagate(self, input_nn):
        input_nn = input_nn.flatten('F')
        if len(input_nn) != self.input_size:
            raise ValueError()
        for layer in range(self.number_of_layers - 1):
            a_lm1 = input_nn if layer == 0 else self._neurons[layer - 1]
            self._neurons[layer] = np.dot(a_lm1, self._weights[layer]) + self._biases[layer]

    def cost(self, input_data: np.array, answer_key: int) -> np.array:
        input_data = input_data.flatten('F')
        if len(input_data) != self.input_size:
            raise ValueError(
                'Input raw data size mismatch with neural network ({},{})'.format(len(input_data), self.input_size))
        answer = np.zeros(self.output_size)
        answer[answer_key] = 1.0
        output = self.propagate(input_data)
        cost_score = output - answer
        cost_score = np.array([el ** 2 for el in cost_score])
        return np.sum(cost_score)

    def back_propagate(self, input_data: np.array, answer_key: int):
        input_data = input_data.flatten('F')
        self.propagate(input_data)
        print('start back propagation')
        y = np.zeros(self.output_size)
        y[answer_key] = 1.0
        for layer_n in range(self.number_of_layers - 1, 0, 1):
            layer_nm1 = layer_n - 1
            print('Layer #{}'.format(layer_n))
            a_l = self._neurons[layer_n]
            a_lm1 = input_data if layer_nm1 == -1 else self._neurons[layer_nm1]
            dc_over_da_l = 2 * (y - a_l)
            w = self._weights[layer_n]
            b = self._biases[layer_n]
            z = np.dot(a_lm1, w) + b
            da_over_dz = sigmoid_derivative(z)
            dc_over_dz = np.multiply(dc_over_da_l, da_over_dz)
            dz_over_dw = a_lm1
            dz_over_da_lm1 = self._weights[layer_n]
            dc_over_dw = np.outer(dz_over_dw, dc_over_dz)
            dc_over_db = dc_over_dz
            dc_over_a_lm1 = np.dot(dz_over_da_lm1, da_over_dz)
            self._weights[layer_n] += dc_over_dw
            self._biases[layer_n] += dc_over_db
            a_l = a_lm1
            y = a_lm1 + dc_over_a_lm1

    @input_size.setter
    def input_size(self, value):
        self._input_size = value
