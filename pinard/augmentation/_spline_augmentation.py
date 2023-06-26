import random

import numpy as np
import scipy.interpolate as interpolate

from .augmenter import Augmenter


def segment_length(x1, y1, x2, y2):
    """
    Compute the length of a line segment given its coordinates.

    Parameters
    ----------
    x1 : float
        x-coordinate of the first point.
    y1 : float
        y-coordinate of the first point.
    x2 : float
        x-coordinate of the second point.
    y2 : float
        y-coordinate of the second point.

    Returns
    -------
    float
        Length of the line segment.
    """

    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


v_segment_length = np.vectorize(segment_length)


def X_length(x, y):
    """
    Compute the total length, segment lengths, and cumulative segment lengths of a curve given its coordinates.

    Parameters
    ----------
    x : ndarray
        Array of x-coordinates of the curve.
    y : ndarray
        Array of y-coordinates of the curve.

    Returns
    -------
    tuple
        A tuple containing the total length, segment lengths, and cumulative segment lengths of the curve.
    """

    x1 = x[:-1]
    y1 = y[:-1]
    x2 = x[1:]
    y2 = y[1:]

    SpecLen_seg = v_segment_length(x1, y1, x2, y2)
    SpecLen = np.sum(SpecLen_seg)
    SpecLen_seg_cumsum = np.cumsum(SpecLen_seg)
    return SpecLen, SpecLen_seg, SpecLen_seg_cumsum


def segment_pt_coord(x1, y1, x2, y2, fracL, L):
    """
    Compute the coordinates of a point on a line segment given the fraction of its length.

    Parameters
    ----------
    x1 : float
        x-coordinate of the first point of the line segment.
    y1 : float
        y-coordinate of the first point of the line segment.
    x2 : float
        x-coordinate of the second point of the line segment.
    y2 : float
        y-coordinate of the second point of the line segment.
    fracL : float
        Fraction of the length of the line segment.
    L : float
        Length of the line segment.

    Returns
    -------
    tuple
        A tuple containing the x and y coordinates of the point on the line segment.
    """

    propL = fracL / L
    xp = x1 + propL * (x2 - x1)
    yp = y1 + propL * (y2 - y1)
    return xp, yp


def interval_selection(n_l, CumVect):
    """
    Select the interval indices that bound a given value in an array.

    Parameters
    ----------
    n_l : float
        Value to be bounded.
    CumVect : ndarray
        Cumulative array of values.

    Returns
    -------
    tuple
        A tuple containing the minimum and maximum indices of the bounding interval.
    """

    i1 = np.where(n_l <= CumVect)
    i2 = np.where(n_l >= CumVect)
    return np.min(i1), np.max(i2)


class Random_X_Spline_Deformation(Augmenter):
    def augment(self, X, apply_on="samples"):
        """
        Randomly modify the x-coordinate based on subsampled spline.

        Parameters
        ----------
        X : ndarray
            Input data.
        apply_on : str, optional
            Apply augmentation on "samples" or "features" (default: "samples").

        Returns
        -------
        ndarray
            Augmented data.
        """

        x = np.arange(0, len(X[0]), 1)
        t, c, k = interpolate.splrep(x, X[0], s=0, k=3)
        delta_x_size = int(np.around(len(t) / 20))
        delta_x = np.linspace(np.min(x), np.max(x), delta_x_size)
        delta_y = np.random.uniform(-10, 10, delta_x_size)
        delta = np.interp(t, delta_x, delta_y)
        t = t + delta
        spline = interpolate.BSpline(t, c, k, extrapolate=True)
        return spline(x)


class Random_Spline_Addition(Augmenter):
    def augment(self, X, apply_on="samples"):
        """
        Add spline noise on y-coordinate.

        Parameters
        ----------
        X : ndarray
            Input data.
        apply_on : str, optional
            Apply augmentation on "samples" or "features" (default: "samples").

        Returns
        -------
        ndarray
            Augmented data.
        """

        nfreq = len(X)
        x_range_real = np.arange(0, nfreq, 1)
        interval_width = np.max(X) / 80
        half_interval_width = interval_width / 2
        baseline = random.uniform(-half_interval_width, half_interval_width)
        interval_min = -half_interval_width + baseline
        interval_max = half_interval_width + baseline

        nb_spline_points = 40
        x_points = np.linspace(0, nfreq, nb_spline_points)
        y_points = [random.uniform(interval_min, interval_max) for _ in range(nb_spline_points)]

        x = np.asarray(x_points)
        x.sort()
        y = np.asarray(y_points)
        t, c, k = interpolate.splrep(x, y, s=0, k=3)
        spline = interpolate.BSpline(t, c, k, extrapolate=False)
        distor = spline(x_range_real)
        y_distor = X + np.reshape(distor, (-1, 1))
        return np.array(y_distor)


class Monotonous_Spline_Simplification(Augmenter):
    def augment(self, X, apply_on="samples"):
        """
        Select regularly spaced points along the x-axis and adjust a spline.

        Parameters
        ----------
        X : ndarray
            Input data.
        apply_on : str, optional
            Apply augmentation on "samples" or "features" (default: "samples").

        Returns
        -------
        ndarray
            Augmented data.
        """

        nfreq = len(X[0])
        x_range_real = np.arange(0, nfreq, 1)
        nb_points = 60

        ctrl_points = np.unique(np.concatenate(([0], random.sample(range(nfreq), nb_points), [nfreq - 1])))
        ctrl_points.sort()
        x = x_range_real[ctrl_points]
        y = X[0][ctrl_points]
        t, c, k = interpolate.splrep(x, y, s=0, k=3)
        spline = interpolate.BSpline(t, c, k, extrapolate=False)

        return spline(x_range_real)


class Dependent_Spline_Simplification(Augmenter):
    def augment(self, X, apply_on="samples"):
        """
        Select regularly spaced points on the x-axis and adjust a spline.

        Parameters
        ----------
        X : ndarray
            Input data.
        apply_on : str, optional
            Apply augmentation on "samples" or "features" (default: "samples").

        Returns
        -------
        ndarray
            Augmented data.
        """

        nfreq = len(X[0])
        x0 = np.linspace(0, np.max(X[0]), nfreq)
        res = X_length(x0, X[0])
        nb_segments = 10
        x_samples = []
        y_samples = []

        for s in range(1, nb_segments):
            length = X_length(x0, X[0])[0] / nb_segments
            n_l = s * length
            test = res[2]
            toto = interval_selection(n_l, test)

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
        t, c, k = interpolate.splrep(x, y, s=0, k=3)
        xmin, xmax = x.min(), x.max()
        xx = np.linspace(xmin, xmax, nfreq)
        spline = interpolate.BSpline(t, c, k, extrapolate=False)

        return spline(xx)
