import random

import numpy as np

from .augmenter import Augmenter


def angle_p(x, xI, yI, p1, p2):
    if x <= xI:
        return p1 * (x - xI) + yI
    else:
        return p2 * (x - xI) + yI


v_angle_p = np.vectorize(angle_p)


class Rotate_Translate(Augmenter):
    def augment(self, X, per_sample=True):
        """rotate and translate signal"""

        def deformation(x):
            x_range = np.linspace(0, 1, x.shape[-1])
            p2 = random.uniform(-2, 2)
            p1 = random.uniform(-2, 2)
            xI = random.uniform(0, 1)
            yI = random.uniform(0, np.max(x) / 3)
            distor = v_angle_p(x_range, xI, yI, p1, p2)
            return distor

        if per_sample:
            distor = np.array([deformation(x) * np.std(x) for x in X])
        else:
            distor = deformation(X) * np.std(X)
        y_distor = X + distor

        return y_distor


class Random_Y_Shift(Augmenter):
    def augment(self, X):
        """Additive delta on y"""
        spec_range = np.max(X) - np.min(X)
        y_distor = X + random.uniform(-spec_range, spec_range) / 20
        return y_distor


class Random_Multiplicative_Shift(Augmenter):
    def augment(self, X):
        """Multiplicative delta on y"""
        y_distor = X * random.uniform(0.98, 1.03)
        return y_distor
