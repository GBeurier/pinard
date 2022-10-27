import random

import numpy as np
import scipy.interpolate as interpolate

from .augmenter import Augmenter


def segment_length(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


v_segment_length = np.vectorize(segment_length)


def X_length(x, y):
    x1 = x[range((len(x) - 1))]
    y1 = y[range((len(y) - 1))]
    x2 = x[range(1, len(x))]
    y2 = y[range(1, len(y))]
    # y1 = np.reshape(np.array(y1), (-1, 1))
    # y2 = np.reshape(np.array(y2), (-1, 1))

    SpecLen_seg = v_segment_length(x1, y1, x2, y2)
    SpecLen = np.sum(SpecLen_seg)
    SpecLen_seg_cumsum = np.cumsum(SpecLen_seg)
    return (SpecLen, SpecLen_seg, SpecLen_seg_cumsum)


def segment_pt_coord(x1, y1, x2, y2, fracL, L):
    propL = fracL / L
    xp = x1 + propL * (x2 - x1)
    yp = y1 + propL * (y2 - y1)
    return (xp, yp)


def interval_selection(n_l, CumVect):
    i1 = np.where(n_l <= CumVect)
    i2 = np.where(n_l >= CumVect)
    return (np.min(i1), np.max(i2))


class Random_X_Spline_Deformation(Augmenter):
    # def __init__(self, apply_on="samples", random_state=None, *, copy=True):
    #     super().__init__(apply_on, random_state, copy=copy)

    def augment(self, X, apply_on="samples"):
        """Random modification of x based on subsampled spline"""
        x = np.arange(0, len(X[0]), 1)
        t, c, k = interpolate.splrep(x, X[0], s=0, k=3)
        delta_x_size = int(np.around(len(t) / 20))
        delta_x = np.linspace(np.min(x), np.max(x), delta_x_size)
        delta_y = np.random.uniform(-10, 10, delta_x_size)
        delta = np.interp(t, delta_x, delta_y)
        t = t + delta
        spline = interpolate.BSpline(t, c, k, extrapolate=True)
        return spline(x)


# class Random_X_Spline_Shift(Augmenter):
#     def augment(self, X, apply_on="samples"):
#         """Add spline based x shift"""
#         x = np.arange(0, len(X[0]), 1)
#         delta = np.random.uniform(-0.5, 0.5, len(X[0]))
#         x = x + delta
#         print(x)
#         t, c, k = interpolate.splrep(x, X[0], s=0, k=3)
#         spline = interpolate.BSpline(t, c, k, extrapolate=True)
#         return spline(x)


class Random_Spline_Addition(Augmenter):
    def augment(self, X, apply_on="samples"):
        """Add spline noise on y"""
        nfreq = len(X)
        x_range_real = np.arange(0, nfreq, 1)
        interval_width = np.max(X) / 80
        half_interval_width = interval_width / 2
        baseline = random.uniform(-half_interval_width, half_interval_width)
        interval_min = -half_interval_width + baseline
        interval_max = half_interval_width + baseline

        nb_spline_points = 40
        x_points = np.linspace(0, nfreq, nb_spline_points)
        y_points = [
            random.uniform(interval_min, interval_max) for i in range(nb_spline_points)
        ]

        x = np.asarray(x_points)
        x.sort()
        y = np.asarray(y_points)
        t, c, k = interpolate.splrep(x, y, s=0, k=3)
        spline = interpolate.BSpline(t, c, k, extrapolate=False)
        distor = spline(x_range_real)
        y_distor = X + np.reshape(distor, (-1, 1))
        # y_distor[y_distor < 0] = 0
        return np.array(y_distor)


# Kind of equivalent to gaussian preproprocessing
class Monotonous_Spline_Simplification(Augmenter):
    def augment(self, X, apply_on="samples"):
        """Select regularly spaced points along x_axis and adjust a spline"""
        nfreq = len(X[0])
        x_range_real = np.arange(0, nfreq, 1)
        nb_points = 60

        ctrl_points = np.unique(
            np.concatenate(([0], random.sample(range(nfreq), nb_points), [nfreq - 1]))
        )
        ctrl_points.sort()
        print(len(ctrl_points))
        x = x_range_real[ctrl_points]
        y = X[0][ctrl_points]
        t, c, k = interpolate.splrep(x, y, s=0, k=3)
        spline = interpolate.BSpline(t, c, k, extrapolate=False)

        return spline(x_range_real)


class Dependent_Spline_Simplification(Augmenter):
    def augment(self, X, apply_on="samples"):
        """Select regularly spaced points ON the X and adjust a spline"""
        nfreq = len(X[0])
        x0 = np.linspace(0, np.max(X[0]), nfreq)
        res = X_length(x0, X[0])
        nb_segments = 10
        x_samples = []
        y_samples = []

        print(res, nfreq, nb_segments)

        for s in range(1, nb_segments):
            length = X_length(x0, X[0])[0] / nb_segments
            print(">>>", length)
            # cumulative_length = np.cumsum(np.repeat(l,nb_segments))
            n_l = s * length
            print(">>>", n_l)
            test = res[2]
            print(">>>", test)
            toto = interval_selection(n_l, test)
            print("----", toto)

            P = segment_pt_coord(
                x1=x0[toto[1]],
                y1=X[toto[1]],
                x2=x0[toto[0]],
                y2=X[toto[0]],
                fracL=res[1][toto[1]] % length,
                L=res[1][toto[1]],
            )

            x_samples.append(P[0])
            y_samples.append(P[1])

        x = np.array(x_samples)
        x = np.concatenate(([0], x, [np.max(x0)]))
        y = np.array(y_samples)
        y = np.concatenate(([X[0][0]], y, [X[0][nfreq - 1]]))
        # print(x)
        t, c, k = interpolate.splrep(x, y, s=0, k=3)
        xmin, xmax = x.min(), x.max()
        xx = np.linspace(xmin, xmax, nfreq)
        spline = interpolate.BSpline(t, c, k, extrapolate=False)

        return spline(xx)
