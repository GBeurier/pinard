import random as rd

import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.utils.validation import _num_samples

from ..sklearn._utils import _validate_shuffle_split


def shuffle_sampling(data, test_size, *, random_state=None):
    split_model = ShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    return next(split_model.split(data))


def systematic_circular_sampling(data, y, test_size, random_state):
    """
    FONCTION d'échantillonnage : ``systematic_sampling``
    --------
    Permet d'effectuer un échantillonnage non-aléatoire, basé sur la méthode
    d'échantillonnage systématique circulaire.
    Note
    ----
    Le point de départ, et le nombre de rotations sont tirés aléatoirement.
    Paramètres
    ----------
    * ``size`` : Int/Float
            La quantité d'échantillons à prélever, peut être exprimé soit en nombre, soit en proportion.
    * ``data`` : DataFrame
            DataFrame contenant les échantillons.
    * ``random_state`` : Int, default=None
            Valeur de la seed, pour la reproductibilité des résultats.
    Return
    ------
    * ``Tuple`` : (List[int], List[int])
            Retourne un split entre le train_set et le test_set.
    Exemple
    -------
    >>> index_test = systematic_sampling(0.2, data, 1)
    >>> print(sorted(index_test))
    [3, 8, ..., 53, 58, ..., 101, 106]
    """
    if y is None:
        raise ValueError("Y data are required to use systematic circular sampling")

    if random_state is not None:
        rd.seed(random_state)

    n_samples = _num_samples(data)
    n_train, n_test = _validate_shuffle_split(n_samples, test_size, None)

    ordered_idx = np.argsort(y[:, 0], axis=0)
    rotated_idx = np.roll(ordered_idx, rd.randint(0, n_samples))

    step = n_samples / n_train
    indices = [round(step * i) for i in range(n_train)]

    index_train = rotated_idx[indices]
    index_test = np.delete(rotated_idx, indices)
    return (index_train, index_test)
