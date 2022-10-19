import numpy as np
import scipy.interpolate as interpolate

from augmenter import Augmenter


def spectrum_length(x, y):
    x1 = x[range((len(x) - 1))]
    y1 = y[range((len(y) - 1))]
    x2 = x[range(1, len(x))]
    y2 = y[range(1, len(y))]
    SpecLen_seg = v_segment_length(x1, y1, x2, y2)
    SpecLen = np.sum(SpecLen_seg)
    SpecLen_seg_cumsum = np.cumsum(SpecLen_seg)
    return (SpecLen, SpecLen_seg, SpecLen_seg_cumsum)


def segment_length(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


v_segment_length = np.vectorize(segment_length)


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
    def augment(self, spectrum, y):
        """Random modification of x based on subsampled spline"""
        x = np.arange(0, len(spectrum), 1)
        t, c, k = interpolate.splrep(x, spectrum, s=0, k=3)

        delta_x_size = int(np.around(len(t) / 20))
        delta_x = np.linspace(np.min(x), np.max(x), delta_x_size)
        delta_y = np.random.uniform(-10, 10, delta_x_size)
        delta = np.interp(t, delta_x, delta_y)
        t = t + delta
        spline = interpolate.BSpline(t, c, k, extrapolate=True)
        return spline(x), y


class Random_X_Spline_Shift(Augmenter):
    def augment(self, spectrum, y):
        """Add spline based x shift"""
        x = np.arange(0, len(spectrum), 1)
        delta = random.uniform(-10, 10)
        x = x + delta
        t, c, k = interpolate.splrep(x, spectrum, s=0, k=3)
        spline = interpolate.BSpline(t, c, k, extrapolate=False)
        return spline(x), y


class Monotonous_Spline_Simplification(Augmenter):
    def augment(self, spectrum, y):
        """Select regularly spaced points along x_axis and adjust a spline"""
        nfreq = len(spectrum)
        x_range_real = np.arange(0, nfreq, 1)
        nb_points = 60

        ctrl_points = np.unique(
            np.concatenate(([0], random.sample(range(nfreq), nb_points), [nfreq - 1]))
        )
        ctrl_points.sort()
        x = x_range_real[ctrl_points]
        y = spectrum[ctrl_points]
        t, c, k = interpolate.splrep(x, y, s=0, k=3)
        spline = interpolate.BSpline(t, c, k, extrapolate=False)

        return spline(x_range_real), y


class Dependent_Spline_Simplification(Augmenter):
    def augment(self, spectrum, y):
        """Select regularly spaced points ON the spectrum and adjust a spline"""
        nfreq = len(spectrum)
        x0 = np.linspace(0, np.max(spectrum), nfreq)
        res = spectrum_length(x0, spectrum)
        nb_segments = 10
        x_samples = []
        y_samples = []

        for s in range(1, nb_segments):
            l = spectrum_length(x0, spectrum)[0] / nb_segments
            # cumulative_length = np.cumsum(np.repeat(l,nb_segments))
            n_l = s * l
            test = res[2]
            toto = interval_selection(n_l, test)
            P = segment_pt_coord(
                x1=x0[toto[1]],
                y1=spectrum[toto[1]],
                x2=x0[toto[0]],
                y2=spectrum[toto[0]],
                fracL=res[1][toto[1]] % l,
                L=res[1][toto[1]],
            )

            x_samples.append(P[0])
            y_samples.append(P[1])

        x = np.array(x_samples)
        x = np.concatenate(([0], x, [np.max(x0)]))
        y = np.array(y_samples)
        y = np.concatenate(([spectrum[0]], y, [spectrum[nfreq - 1]]))
        # print(x)
        t, c, k = interpolate.splrep(x, y, s=0, k=3)
        xmin, xmax = x.min(), x.max()
        xx = np.linspace(xmin, xmax, nfreq)
        spline = interpolate.BSpline(t, c, k, extrapolate=False)

        return spline(xx), y


class Random_Spline_Addition(Augmenter):
    def augment(self, spectrum, y):
        """Add spline noise on y"""
        nfreq = len(spectrum)
        x_range_real = np.arange(0, nfreq, 1)
        interval_width = np.max(spectrum) / 80
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
        y_distor = spectrum + distor
        # y_distor[y_distor < 0] = 0
        return np.array(y_distor), y
