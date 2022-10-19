import numpy as np


def angle_p(x, xI, yI, p1, p2):
    if x <= xI:
        return p1 * (x - xI) + yI
    else:
        return p2 * (x - xI) + yI


v_angle_p = np.vectorize(angle_p)


class Rotate_Translate(Augmenter):
    def augment(self, spectrum, y):
        """rotate and translate signal"""
        x_range = np.linspace(0, 1, len(spectrum))
        p2 = random.uniform(-2, 2)
        p1 = random.uniform(-2, 2)
        xI = random.uniform(0, 1)
        yI = random.uniform(0, np.max(spectrum) / 3)
        distor = v_angle_p(x_range, xI, yI, p1, p2)
        distor = distor * np.std(spectrum)
        y_distor = spectrum + distor
        # y_distor[y_distor < 0] = 0
        return np.array(y_distor), y


class Random_Y_Shift(Augmenter):
    def augment(self, spectrum, y):
        """Additive delta on y"""
        spec_range = np.max(spectrum) - np.min(spectrum)
        y_distor = spectrum + random.uniform(-spec_range, spec_range) / 20
        return y_distor, y


class Random_Multiplicative_Shift(Augmenter):
    def augment(self, spectrum, y):
        """Multiplicative delta on y"""
        y_distor = spectrum * random.uniform(-0.05, 0.05)
        return y_distor, y
