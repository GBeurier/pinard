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
    def __init__(self, random_state=None, per_sample=True, *, copy=True, p_range=2, y_factor=3):
        self.p_range = p_range
        self.y_factor = y_factor
        super().__init__(random_state, per_sample, copy=copy)

    def augment(self, X, per_sample=True):
        """rotate and translate signal"""

        def deformation(x):
            x_range = np.linspace(0, 1, x.shape[-1])
            p2 = random.uniform(-self.p_range, self.p_range)
            p1 = random.uniform(-self.p_range, self.p_range)
            xI = random.uniform(0, 1)
            yI = random.uniform(0, np.max(x) / self.y_factor)
            distor = v_angle_p(x_range, xI, yI, p1, p2)
            return distor

        if per_sample:
            increment = np.array([deformation(x) * np.std(x) for x in X])
        else:
            increment = deformation(X) * np.std(X)

        new_X = X + increment

        return new_X


class Random_Y_Shift(Augmenter):
    def __init__(self, random_state=None, per_sample=True, *, copy=True, y_factor=20):
        self.y_factor = y_factor
        super().__init__(random_state, per_sample, copy=copy)

    def augment(self, X, per_sample=True):
        """Additive delta on y"""
        spec_range = np.max(X) - np.min(X)

        if per_sample:
            increment = np.array(
                [random.uniform(-spec_range, spec_range) / self.y_factor for _ in X])
        else:
            increment = random.uniform(-spec_range, spec_range) / self.y_factor

        new_X = X + increment

        return new_X


class Random_Multiplicative_Shift(Augmenter):
    def __init__(self, random_state=None, per_sample=True, *, copy=True, multiplier_range=(0.97, 1.03)):
        self.multiplier_range = multiplier_range
        super().__init__(random_state, per_sample, copy=copy)

    def augment(self, X, per_sample=True):
        """Multiplicative delta on y"""
        if per_sample:
            factor = np.array(
                [random.uniform(*self.multiplier_range) for _ in X])
        else:
            factor = random.uniform(*self.multiplier_range)

        new_X = X * factor

        return new_X
