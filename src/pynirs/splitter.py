from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.model_selection import KFold as sk_KFold

# class KFold(_BaseKFold):
#     def __init__(self, n_splits = 5, **kwargs):
#         super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
#         del self.shuffle
#         del self.random_state
    
#     def _iter_test_indices(self, X=None, y=None, groups=None):
#         n_samples = _num_samples(X)

#         _ks = _KennardStone()
#         indices = _ks._get_indexes(X)

#         n_splits = self.n_splits
#         fold_sizes = np.full(n_splits, n_samples // n_splits, dtype = int)
#         fold_sizes[:n_samples % n_splits] += 1
#         current = 0
#         for fold_size in fold_sizes:
#             start, stop = current, current + fold_size
#             yield indices[start:stop]
#             current = stop



# def train_test_split(*arrays, test_size=None, train_size=None, **kwargs):
#     pass


# import numpy as np
# from sklearn.model_selection import train_test_split, KFold
# from scipy.spatial.distance import cdist
# # from kennard_stone import train_test_split, KFold
# a
# def random_split(spectra, test_size=0.25, random_state=None, shuffle=True, stratify=None):
#     """implement random_split by using sklearn.model_selection.train_test_split function. See
#     http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
#     for more infomation.
#     """
#     return train_test_split(
#         spectra,
#         test_size=test_size,
#         random_state=random_state,
#         shuffle=shuffle,
#         stratify=stratify)


# def kennardstone(spectra, test_size=0.25, metric='euclidean', *args, **kwargs):
#     """Kennard Stone Sample Split method
#     Parameters
#     ----------
#     spectra: ndarray, shape of i x j
#         i spectrums and j variables (wavelength/wavenumber/ramam shift and so on)
#     test_size : float, int
#         if float, then round(i x (1-test_size)) spectrums are selected as test data, by default 0.25
#         if int, then test_size is directly used as test data size
#     metric : str, optional
#         The distance metric to use, by default 'euclidean'
#         See scipy.spatial.distance.cdist for more infomation
#     Returns
#     -------
#     select_pts: list
#         index of selected spetrums as train data, index is zero based
#     remaining_pts: list
#         index of remaining spectrums as test data, index is zero based
#     References
#     --------
#     Kennard, R. W., & Stone, L. A. (1969). Computer aided design of experiments.
#     Technometrics, 11(1), 137-148. (https://www.jstor.org/stable/1266770)
#     """

#     if test_size < 1:
#         train_size = round(spectra.shape[0] * (1 - test_size))
#     else:
#         train_size = spectra.shape[0] - round(test_size)

#     if train_size > 2:
#         distance = cdist(spectra, spectra, metric=metric, *args, **kwargs)
#         select_pts, remaining_pts = max_min_distance_split(distance, train_size)
#     else:
#         raise ValueError("train sample size should be at least 2")

#     return select_pts, remaining_pts


# def spxy(spectra, yvalues, test_size=0.25, metric='euclidean', *args, **kwargs):
#     """SPXY Sample Split method
#     Parameters
#     ----------
#     spectra: ndarray, shape of i x j
#         i spectrums and j variables (wavelength/wavenumber/ramam shift and so on)
#     test_size : float, int
#         if float, then round(i x (1-test_size)) spectrums are selected as test data, by default 0.25
#         if int, then test_size is directly used as test data size
#     metric : str, optional
#         The distance metric to use, by default 'euclidean'
#         See scipy.spatial.distance.cdist for more infomation
#     Returns
#     -------
#     select_pts: list
#         index of selected spetrums as train data, index is zero based
#     remaining_pts: list
#         index of remaining spectrums as test data, index is zero based
#     References
#     ---------
#     Galvao et al. (2005). A method for calibration and validation subset partitioning.
#     Talanta, 67(4), 736-740. (https://www.sciencedirect.com/science/article/pii/S003991400500192X)
    
#     Li, Wenze, et al. "HSPXY: A hybrid‐correlation and diversity‐distances based data partition method." Journal of Chemometrics 33.4 (2019): e3109. 
#     """

#     if test_size < 1:
#         train_size = round(spectra.shape[0] * (1 - test_size))
#     else:
#         train_size = spectra.shape[0] - round(test_size)

#     if train_size > 2:
#         yvalues = yvalues.reshape(yvalues.shape[0], -1)
#         distance_spectra = cdist(spectra, spectra, metric=metric, *args, **kwargs)
#         distance_y = cdist(yvalues, yvalues, metric=metric, *args, **kwargs)
#         distance_spectra = distance_spectra / distance_spectra.max()
#         distance_y = distance_y / distance_y.max()

#         distance = distance_spectra + distance_y
#         select_pts, remaining_pts = max_min_distance_split(distance, train_size)
#     else:
#         raise ValueError("train sample size should be at least 2")

#     return select_pts, remaining_pts


# def max_min_distance_split(distance, train_size):
#     """sample set split method based on maximun minimun distance, which is the core of Kennard Stone
#     method
#     Parameters
#     ----------
#     distance : distance matrix
#         semi-positive real symmetric matrix of a certain distance metric
#     train_size : train data sample size
#         should be greater than 2
#     Returns
#     -------
#     select_pts: list
#         index of selected spetrums as train data, index is zero-based
#     remaining_pts: list
#         index of remaining spectrums as test data, index is zero-based
#     """

#     select_pts = []
#     remaining_pts = [x for x in range(distance.shape[0])]

#     # first select 2 farthest points
#     first_2pts = np.unravel_index(np.argmax(distance), distance.shape)
#     select_pts.append(first_2pts[0])
#     select_pts.append(first_2pts[1])

#     # remove the first 2 points from the remaining list
#     remaining_pts.remove(first_2pts[0])
#     remaining_pts.remove(first_2pts[1])

#     for i in range(train_size - 2):
#         # find the maximum minimum distance
#         select_distance = distance[select_pts, :]
#         min_distance = select_distance[:, remaining_pts]
#         min_distance = np.min(min_distance, axis=0)
#         max_min_distance = np.max(min_distance)

#         # select the first point (in case that several distances are the same, choose the first one)
#         points = np.argwhere(select_distance == max_min_distance)[:, 1].tolist()
#         for point in points:
#             if point in select_pts:
#                 pass
#             else:
#                 select_pts.append(point)
#                 remaining_pts.remove(point)
#                 break
#     return select_pts, remaining_pts