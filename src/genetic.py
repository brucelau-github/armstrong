from random import (choice, random, randint)
import numpy as np

__all__ = ['Chromosome', 'Population']

class Chromosome:
    """
    This class is used to define a chromosome for the gentic algorithm
    simulation.

    This class is essentially nothing more than a container for the details
    of the chromosome, namely the gene (the string that represents our
    target string) and the fitness (how close the gene is to the target
    string).

    Note that this class is immutable.  Calling mate() or mutate() will
    result in a new chromosome instance being created.
    """

    def __init__(self, gene):
        self.gene = gene
        self.fitness = None

    def mate(self, mate):
        """
        Method used to mate the chromosome with another chromosome,
        resulting in a new chromosome being returned.
        """
        pivot = randint(0, len(self.gene) - 1)
        gene1 = np.hstack((self.gene[:pivot], mate.gene[pivot:]))
        gene2 = np.hstack((mate.gene[:pivot], self.gene[pivot:]))

        return gene1, gene2

    def mutate(self):
        """
        Method used to generate a new chromosome based on a change in a
        random character in the gene of this chromosome.  A new chromosome
        will be created, but this original will not be affected.
        """
        gene = np.array(self.gene)
        n = np.random.random()
        idx = randint(0, len(gene) - 1)
        gene[idx] = n

        return gene
