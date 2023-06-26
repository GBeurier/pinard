import numpy as np
# import random
import operator
# from ast import operator

from .augmenter import Augmenter


def angle_p(x, xI, yI, p1, p2):
    """
    Helper function to calculate the angle for rotation and translation.

    Parameters
    ----------
    x : float
        Input value.
    xI : float
        Reference point.
    yI : float
        Initial value.
    p1 : float
        Slope 1.
    p2 : float
        Slope 2.

    Returns
    -------
    float
        Calculated angle.
    """
    mask = x <= xI
    return np.where(mask, p1 * (x - xI) + yI, p2 * (x - xI) + yI)


v_angle_p = np.vectorize(angle_p)


class Rotate_Translate(Augmenter):
    """
    Class for rotating and translating data augmentation.

    Parameters
    ----------
    apply_on : str, optional
        Apply augmentation on "samples" or "global" data. Default is "samples".
    random_state : int or None, optional
        Random seed for reproducibility. Default is None.
    copy : bool, optional
        If True, creates a copy of the input data. Default is True.
    p_range : int, optional
        Range for generating random slope values. Default is 2.
    y_factor : int, optional
        Scaling factor for the initial value. Default is 3.
    """

    def __init__(self, apply_on="samples", random_state=None, *, copy=True, p_range=2, y_factor=3):
        self.p_range = p_range
        self.y_factor = y_factor
        super().__init__(apply_on, random_state, copy=copy)

    def augment(self, X, apply_on="samples"):
        """
        Augment the data by rotating and translating the signal.

        Parameters
        ----------
        X : ndarray
            Input data to be augmented.
        apply_on : str, optional
            Apply augmentation on "samples" or "global" data. Default is "samples".

        Returns
        -------
        ndarray
            Augmented data.
        """
        def deformation(x):
            x_range = np.linspace(0, 1, x.shape[-1])
            p2 = np.random.uniform(-self.p_range, self.p_range)
            p1 = np.random.uniform(-self.p_range, self.p_range)
            xI = np.random.uniform(0, 1)
            yI = np.random.uniform(0, np.max(x) / self.y_factor)
            distor = v_angle_p(x_range, xI, yI, p1, p2)
            return distor

        if apply_on == "samples":
            increment = np.array([deformation(x) * np.std(x) for x in X])
        elif apply_on == "global":
            increment = deformation(X) * np.std(X)
        else:
            raise ValueError("Rotation transform can only be applied on samples or globally.")

        new_X = X + increment

        return new_X


class Random_X_Operation(Augmenter):
    """
    Class for applying random operation on data augmentation.

    Parameters
    ----------
    apply_on : str, optional
        Apply augmentation on "features" or "samples" data. Default is "features".
    random_state : int or None, optional
        Random seed for reproducibility. Default is None.
    copy : bool, optional
        If True, creates a copy of the input data. Default is True.
    operator_func : function, optional
        Operator function to be applied. Default is operator.mul.
    operator_range : tuple, optional
        Range for generating random values for the operator. Default is (0.97, 1.03).
    """

    def __init__(self, apply_on="features", random_state=None, *, copy=True, operator_func=operator.mul, operator_range=(0.97, 1.03)):
        self.operator_func = operator_func
        self.operator_range = operator_range
        super().__init__(apply_on, random_state, copy=copy)

    def augment(self, X, apply_on="samples"):
        """
        Augment the data by applying random operation.

        Parameters
        ----------
        X : ndarray
            Input data to be augmented.
        apply_on : str, optional
            Apply augmentation on "features" or "samples" data. Default is "features".

        Returns
        -------
        ndarray
            Augmented data.
        """
        min_val = self.operator_range[0]
        interval = self.operator_range[1] - self.operator_range[0]

        if apply_on == "samples":
            increment = np.random.rand(len(X)) * interval + min_val
        elif apply_on == "features":
            increment = np.random.rand(*X.shape) * interval + min_val
        else:
            increment = np.random.uniform(*self.operator_range)

        new_X = self.operator_func(X, increment)

        return new_X
