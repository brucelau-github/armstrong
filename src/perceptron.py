#coding: utf-8

import random
import numpy as np
import pandas as pd
from genetic import Chromosome

class Perceptron(Chromosome):
    """a basic perceptron class
    axis (or inputs) should smaller than 100
    if axis is over 100 the out put will be 1
    """
    _fit_data = None
    def __init__(self, **kwargs):
        if 'axis' in kwargs:
            axis = kwargs['axis']
            gene = np.random.random(axis+1)
        if 'gene' in kwargs:
            gene = kwargs['gene']
        if not kwargs:
            rows, colums = self._fit_data.shape
            gene = np.random.random(colums)
        self.w = gene[:-1]
        self.b = gene[-1]
        super(Perceptron, self).__init__(gene)
        self.fitness = self.fit()

    def response(self, x):
        """perceptron output"""
        y = np.dot(x, self.w) + self.b
        return sigmoid(y)

    def mate(self, pmate):
        g1, g2 = super(Perceptron, self).mate(pmate)
        return Perceptron(gene=g1), Perceptron(gene=g2)

    def mutate(self):
        g1 = super(Perceptron, self).mutate()
        return Perceptron(gene=g1)

    def fit(self):
        data = Perceptron._fit_data
        x = np.array(data.iloc[:, :-1])
        y = np.array(data.iloc[:, -1])
        z = self.response(x)
        error = abs(y - z)
        avg_er = np.average(error)
        return avg_er

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def sigmoid_(x): return x * (1.0 - x)

if __name__ == '__main__':
    data = pd.DataFrame({
        'x1':np.array([0.2]),
        'x2':np.array([0.2]),
        'y':np.array([0.4])
        })
    rows, colums = data.shape
    print(colums - 1)
    Perceptron._fit_data = data

    pf = Perceptron(axis=2)
    pm = Perceptron(axis=2)
    print(pf.gene, pm.gene)
    print(pf.fitness, pm.fitness)
    print(pf.mutate().gene)
    c1, c2 = pf.mate(pm)
    print(c1.gene, c2.gene)
#   for n in range(1, 100, 2):
#       p = Perceptron(100)
#       x = np.random.random(100)
#       print(n, p.response(x))
