#coding: utf-8

import random
import numpy as np
import pandas as pd

class Perceptron(object):
    """a basic perceptron class
    axis (or inputs) should smaller than 100
    if axis is over 100 the out put will be 1
    """
    def __init__(self, axis):
        super(Perceptron, self).__init__()
        self.axis = axis
        self.w = np.random.random(axis)
        self.b = np.random.random()
        self.learningRate = 0.1

    def response(self, x):
        """perceptron output"""
        y = np.dot(x, self.w) + self.b
        return sigmoid(y)

    def train(self, data, iterations=3000, learning_rate=0.002):
        x = np.array(data.iloc[:, :-1])
        _y = np.array(data.iloc[:, -1])
        Z = self.response(x)
        error = _y - Z
        deltaZ = error * sigmoid_(Z)
        print(deltaZ)
        for i in range(iterations):
            error = 0.0



def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def sigmoid_(x): return x * (1.0 - x)


if __name__ == '__main__':
    data = pd.DataFrame({
        'x1':np.ones(10),
        'x2':np.ones(10),
        'y':np.ones(10)
        })
    rows, colums = data.shape
    p = Perceptron(colums - 1)
    p.train(data, 1)
#   for n in range(1, 100, 2):
#       p = Perceptron(100)
#       x = np.random.random(100)
#       print(n, p.response(x))
