from ast import operator
import random
import operator

import numpy as np

from .augmenter import Augmenter


def angle_p(x, xI, yI, p1, p2):
    if x <= xI:
        return p1 * (x - xI) + yI
    else:
        return p2 * (x - xI) + yI


v_angle_p = np.vectorize(angle_p)


class Rotate_Translate(Augmenter):
    def __init__(self, apply_on="samples", random_state=None, *, copy=True, p_range=2, y_factor=3):
        self.p_range = p_range
        self.y_factor = y_factor
        super().__init__(apply_on, random_state, copy=copy)

    def augment(self, X, apply_on="samples"):
        """rotate and translate signal"""

        def deformation(x):
            x_range = np.linspace(0, 1, x.shape[-1])
            p2 = random.uniform(-self.p_range, self.p_range)
            p1 = random.uniform(-self.p_range, self.p_range)
            xI = random.uniform(0, 1)
            yI = random.uniform(0, np.max(x) / self.y_factor)
            distor = v_angle_p(x_range, xI, yI, p1, p2)
            return distor

        if apply_on == "samples":
            increment = np.array([deformation(x) * np.std(x) for x in X])
        elif apply_on == "global":
            increment = deformation(X) * np.std(X)
        else:
            raise ValueError(
                "Rotation transform can only be applied on samples or globally."
            )

        new_X = X + increment

        return new_X


class Random_X_Operation(Augmenter):
    def __init__(self, apply_on="features", random_state=None, *, copy=True, operator_func=operator.mul, operator_range=(0.97, 1.03)):
        self.operator_func = operator_func
        self.operator_range = operator_range
        super().__init__(apply_on, random_state, copy=copy)

    def augment(self, X, apply_on="samples"):
        min_val = self.operator_range[0]
        interval = self.operator_range[1] - self.operator_range[0]

        if apply_on == "samples":
            increment = np.random.rand(len(X)) * interval + min_val
        elif apply_on == "features":
            increment = np.random.rand(*X.shape) * interval + min_val
        else:
            increment = random.uniform(self.operator_range)

        new_X = self.operator_func(X, increment)

        return new_X
