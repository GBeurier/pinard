import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.interpolate as interpolate


def segment_length(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


v_segment_length = np.vectorize(segment_length)


def X_length(x, y):
    x1 = x[range((len(x) - 1))]
    y1 = y[range((len(y) - 1))]
    x2 = x[range(1, len(x))]
    y2 = y[range(1, len(y))]
    y1 = np.reshape(np.array(y1), (-1, 1))
    y2 = np.reshape(np.array(y2), (-1, 1))

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


class Dependent_Spline_Simplification():
    def augment(self, X, apply_on="samples"):
        """Select regularly spaced points ON the X and adjust a spline"""
        nfreq = len(X)
        x0 = np.linspace(0, np.max(X), nfreq)
        res = X_length(x0, X)
        nb_segments = 10
        x_samples = []
        y_samples = []

        print(res, nfreq, nb_segments)

        for s in range(1, nb_segments):
            length = X_length(x0, X)[0] / nb_segments
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
        y = np.concatenate(([X[0]], y, [X[nfreq - 1]]))
        # print(x)
        t, c, k = interpolate.splrep(x, y, s=0, k=3)
        xmin, xmax = x.min(), x.max()
        xx = np.linspace(xmin, xmax, nfreq)
        spline = interpolate.BSpline(t, c, k, extrapolate=False)

        return spline(xx)


x_fname = "test_augmentation.csv"
x_df = pd.read_csv(x_fname, sep=";", header=None)
x = x_df.astype(np.float32).values
y = np.reshape(x[:, 0], (-1, 1))
x = x[:, 1:]


x_axis = np.arange(0, len(x[0]), 1)
augmenter = Dependent_Spline_Simplification()
y_vals = augmenter.augment(x[0])
print(x.shape, y_vals.shape)
plt.plot(x_axis, x[0], color="blue")
plt.plot(x_axis, y_vals, color="red")
plt.show()
