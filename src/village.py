from random import (choice, random, randint)
import pandas as pd
import numpy as np

from perceptron import Perceptron


class Population:
    """
    A class representing a population for a genetic algorithm simulation.

    A population is simply a sorted collection of chromosomes
    (sorted by fitness) that has a convenience method for evolution.  This
    implementation of a population uses a tournament selection algorithm for
    selecting parents for crossover during each generation's evolution.

    Note that this object is mutable, and calls to the evolve()
    method will generate a new collection of chromosome objects.
    """

    _tournamentSize = 3

    def __init__(self, size=1024, crossover=0.8, elitism=0.1, mutation=0.03):
        self.elitism = elitism
        self.mutation = mutation
        self.crossover = crossover

        buf = []
        for i in range(size): buf.append(Perceptron())
        self.population = list(sorted(buf, key=lambda x: x.fitness))

    def _tournament_selection(self):
        """
        A helper method used to select a random chromosome from the
        population using a tournament selection algorithm.
        """
        best = choice(self.population)
        for i in range(Population._tournamentSize):
            cont = choice(self.population)
            if (cont.fitness < best.fitness): best = cont

        return best

    def _selectParents(self):
        """
        A helper method used to select two parents from the population using a
        tournament selection algorithm.
        """

        return (self._tournament_selection(), self._tournament_selection())

    def evolve(self):
        """
        Method to evolve the population of chromosomes.
        """
        size = len(self.population)
        idx = int(round(size * self.elitism))
        buf = self.population[:idx]

        while (idx < size):
            if random() <= self.crossover:
                (p1, p2) = self._selectParents()
                children = p1.mate(p2)
                for c in children:
                    if random() <= self.mutation:
                        buf.append(c.mutate())
                    else:
                        buf.append(c)
                idx += 2
            else:
                if random() <= self.mutation:
                    buf.append(self.population[idx].mutate())
                else:
                    buf.append(self.population[idx])
                idx += 1

        self.population = list(sorted(buf[:size], key=lambda x: x.fitness))

if __name__ == "__main__":
    x = np.arange(0.2, 0.5, 0.05)
    data = pd.DataFrame({
        'x1': x,
        'x2': x,
        'x3': x,
        'x4': x,
        'y':2*x
        })
    Perceptron._fit_data = data
    print(data)

    maxGenerations = 16384
    pop = Population(size=2048, crossover=0.8, elitism=0.1, mutation=0.8)

    for i in range(1, maxGenerations + 1):
        print("Generation %d: fitness %s, gene %s"
                % (i, pop.population[0].fitness,pop.population[0].gene))
        if pop.population[0].fitness <= 0.0001: break
        else:pop.evolve()
    else:
        print("Maximum generations reached without success.")
