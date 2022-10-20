import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.utils.validation import _num_samples

from ..sklearn._utils import _validate_shuffle_split


# TODO refactor for perf
def _max_min_distance_split(distance, train_size):
    """
    FUNCTION : ``max_min_distance_split``
    --------
    Sample set split method based on maximun minimun distance, which is the core 
    of Kennard Stone method.
    Parameters
    ----------
    * ``distance`` : Ndarray
            Semi-positive real symmetric matrix of a certain distance metric.
    * ``train_size`` : Int
            Should be greater than 2.
    Returns
    -------
    * ``Tuple`` : (List of int, List of int)
            Index of selected spetrums as train data, index is zero-based.
             Index of remaining spectrums as test data, index is zero-based.
    Example
    -------
    >>> index_train, index_test = max_min_distance_split(distance, train_size)
    >>> print(index_test[0:3])
    [6, 22, 33, 39]
    """

    index_train = np.array([]).astype(np.int32)
    index_test = np.arange(distance.shape[0]).astype(np.int32)

    # First select 2 farthest points
    first_2pts = np.unravel_index(np.argmax(distance), distance.shape)
    index_train = np.append(index_train, first_2pts[0])
    index_train = np.append(index_train, first_2pts[1])
    # Remove the first 2 points from the remaining list
    index_test = np.delete(index_test, np.argwhere(index_test == first_2pts[0]))
    index_test = np.delete(index_test, np.argwhere(index_test == first_2pts[1]))

    for i in range(train_size - 2):
        # Find the maximum minimum distance
        select_distance = distance[index_train, :]
        min_distance = select_distance[:, index_test]
        min_distance = np.min(min_distance, axis=0)
        max_min_distance = np.max(min_distance)

        # Select the first point (in case that several distances are the same, choose 
        # the first one)
        points = np.argwhere(select_distance == max_min_distance)[:, 1]
        for point in points:
            if point in index_train:
                pass
            else:
                index_train = np.append(index_train, point)
                index_test = np.delete(index_test, np.argwhere(index_test == point))
                break

    return (index_train, index_test)


def ks_sampling(
    data, test_size, *, random_state=None, pca_components=None, metric="euclidean"
):
    """
    FUNCTION of sampling : ``ks_sampling``
    --------
    Samples data using the kennard_stone method.
    Parameters
    ----------
    * ``size`` : Float/int
            Size of the test_set.
    * ``data`` : DataFrame
            Dataset used to get a train_set and a test_set.
    * ``pca`` : Int/Float, default=None
            Value to perform ``PCA``.
    * ``metric`` : Str, default="euclidean"
            The distance metric to use, by default 'euclidean'.
            See scipy.spatial.distance.cdist for more infomation.
    Return
    ------
    * ``Tuple`` : (List of int, List of int)
            Index of selected spetrums as train data, index is zero-based.
             Index of remaining spectrums as test data, index is zero-based.
    * ``Exceptation`` : ValueError
            Return error of type ``ValueError`` if train sample size isn't at least 2.
    Example
    -------
    >>> index_train, index_test = ks_sampling(0.2, data, None, "euclidean")
    >>> print(index_test[0:4])
    [22, 23, 33, 66]
    References
    --------
    Kennard, R. W., & Stone, L. A. (1969). Computer aided design of experiments.
    Technometrics, 11(1), 137-148. (https://www.jstor.org/stable/1266770)
    """

    n_samples = _num_samples(data)
    n_train, n_test = _validate_shuffle_split(n_samples, test_size, None)

    if pca_components is not None:
        pca = PCA(pca_components, random_state=random_state)
        data = pca.fit_transform(data)

    if n_train > 2:
        distance = cdist(data, data, metric=metric)
        return _max_min_distance_split(distance, n_train)
    else:
        raise ValueError("Train sample size should be at least 2.")


def spxy_sampling(
    data, y, test_size, *, random_state=None, pca_components=None, metric="euclidean"
):
    """
    FUNCTION of sampling : ``spxy_sampling``
    -------
    Samples data using the spxy method.
    Parameters
    ----------
    * ``size`` : Float/int
            Size of the test_set.
    * ``features`` : DataFrame
            Features used to get a train_set and a test_set.
    * ``labels`` : DataFrame
            Labels used to get a train_set and a test_set.
    * ``pca`` : Int/Float, default=None
            Value to perform ``PCA``.
    * ``metric`` : Str, default="euclidean"
            The distance metric to use, by default 'euclidean'.
            See scipy.spatial.distance.cdist for more infomation.
    Returns
    -------
    * ``Tuple`` : (List of int, List of int)
            Index of selected spetrums as train data, index is zero-based.
             Index of remaining spectrums as test data, index is zero-based.
    * ``Exceptation`` : ValueError
            Return error of type ``ValueError`` if train sample size isn't at least 2.
    Example
    -------
    >>> index_train, index_test = spxy_sampling(0.2, data, None, "euclidean")
    >>> print(index_test[0:4])
    [6, 22, 33, 39]
    References
    ---------
    Galvao et al. (2005). A method for calibration and validation subset partitioning.
    Talanta, 67(4), 736-740. (https://www.sciencedirect.com/science/article/pii/S003991400500192X)
    Li, Wenze, et al. "HSPXY: A hybrid‐correlation and diversity‐distances based data partition 
    method." Journal of Chemometrics 33.4 (2019): e3109.
    """

    if y is None:
        raise ValueError("Y data are required to use spxy sampling")

    n_samples = _num_samples(data)
    n_train, n_test = _validate_shuffle_split(n_samples, test_size, None)

    if pca_components is not None:
        pca = PCA(pca_components, random_state=random_state)
        data = pca.fit_transform(data)

    # Create Samples
    if n_train > 2:
        distance_features = cdist(data, data, metric=metric)
        distance_features = distance_features / distance_features.max()
        distance_labels = cdist(y, y, metric=metric)
        distance_labels = distance_labels / distance_labels.max()
        distance = distance_features + distance_labels
        return _max_min_distance_split(distance, n_train)
    else:
        raise ValueError("Train sample size should be at least 2.")


#  if train_size > 2:
#         yvalues = yvalues.reshape(yvalues.shape[0], -1)
#         distance_spectra = cdist(spectra, spectra, metric=metric, *args, **kwargs)
#         distance_y = cdist(yvalues, yvalues, metric=metric, *args, **kwargs)
#         distance_spectra = distance_spectra / distance_spectra.max()
#         distance_y = distance_y / distance_y.max()

#         distance = distance_spectra + distance_y
#         select_pts, remaining_pts = max_min_distance_split(distance, train_size)
#     else:
#         raise ValueError("train sample size should be at least 2")


# # =============================================
# # || FONCTION : ks_sampling_train_test_split ||
# # =============================================


# def ks_sampling_train_test_split(
#     data, test_size, pca_components=None, metric="euclidean"
# ):
#     """
#     FONCTION de splitting : ``ks_sampling_train_test_split``
#     --------
#     Permet le splitting d'un jeu de données, en quatre parties : x_train, x_test, y_train, y_test
#             x_train, y_train : Données d'entraînement
#             x_test, y_test   : Données de test
#     Paramètres
#     ----------
#     * ``features`` : DataFrame
#             DataFrame contenant les variables.
#     * ``labels`` : DataFrame
#             DataFrame contenant les étiquettes à prédire.
#     * ``test_size`` : Int/Float
#             Pourcentage d'échantillons que l'on souhaite avoir dans le test_set.
#     * ``pca`` : Int/Float, default=None
#             Nombre de composantes principales pour effectuer la ``PCA``.
#              Effectue une ``PCA`` si l'argument est non-nul.
#     * ``metric`` : Str, default="euclidean"
#             Métrique de distance à utiliser pour les calculs.
#              Voir scipy.spatial.distance.cdist pour plus d'informations.
#     Return
#     ------
#     * ``Tuple`` : (DataFrame, DataFrame, DataFrame, DataFrame)
#             Retourne un tuple de quatre éléments contenant des DataFrames.
#     """

#     return ks_sampling(test_size, features, pca, metric)


# # ===============================================
# # || FONCTION : spxy_sampling_train_test_split ||
# # ===============================================
# def spxy_sampling_train_test_split(
#     features: pd.DataFrame,
#     labels: pd.DataFrame,
#     test_size: tp.Union[int, float],
#     pca: tp.Union[int, float] = None,
#     metric: str = "euclidean",
# ) -> tp.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#     """
#     FONCTION de splitting : ``spxy_sampling_train_test_split``
#     --------
#     Permet le splitting d'un jeu de données, en quatre parties : ``x_train``, ``x_test``, ``y_train``, ``y_test``
#             x_train, y_train : Données d'entraînement
#             x_test, y_test   : Données de test
#     Paramètres
#     ----------
#     * ``features`` : DataFrame
#             DataFrame contenant les variables.
#     * ``labels`` : DataFrame
#             DataFrame contenant les étiquettes à prédire.
#     * ``test_size`` : Int/Float
#             Pourcentage d'échantillons que l'on souhaite avoir dans le test_set.
#     * ``pca`` : Int/Float, default=None
#             Nombre de composantes principales pour effectuer la ``PCA``.
#              Effectue une ``PCA`` si l'argument est non-nul.
#     * ``metric`` : Str, default="euclidean"
#             Métrique de distance à utiliser pour les calculs.
#              Voir scipy.spatial.distance.cdist pour plus d'informations.
#     Return
#     ------
#     * ``Tuple`` : (DataFrame, DataFrame, DataFrame, DataFrame)
#             Retourne un tuple de quatre éléments contenant des dataFrames.
#     Exemple
#     -------
#     >>> x_train, x_test, y_train, y_test = spxy_sampling_train_test_split(features, labels, 0.2, None, "euclidean")
#     >>> print(type(x_train))
#     <class 'pandas.core.frame.DataFrame'>
#     >>> print(len(features))
#     108
#     >>> print(len(x_train))
#     86
#     >>> print(len(x_test))
#     22
#     """
#     # Vérification des entrées
#     gt.input_verification(features, labels)
#     # Application de la pca + récupération des index du train_set, et du test_set
#     index_train, index_test = spxy_sampling(test_size, features, labels, pca, metric)
#     # Récupération des lignes des datasets, selon leurs index
#     return gt.get_train_test_tuple(features, labels, index_train, index_test)
