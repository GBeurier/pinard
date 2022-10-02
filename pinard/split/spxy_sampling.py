#!/bin/python3
#-*- coding : utf-8 -*-

# =============================
# || Importation des modules ||
# =============================

import get_tts as gt
import pandas as pd
import typing as tp
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

# ==============================
# || FUNCTION : spxy_sampling ||
# ==============================

def spxy_sampling(size: tp.Union[int, float],
				  features: pd.DataFrame,
				  labels: pd.DataFrame,
				  pca: tp.Union[int, float]=None,
				  metric: str='euclidean') -> tp.Tuple[tp.List[int], tp.List[int]]:

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
	
	Li, Wenze, et al. "HSPXY: A hybrid‐correlation and diversity‐distances based data partition method." Journal of Chemometrics 33.4 (2019): e3109. 
	"""

	# If test_size is a float 

	len_df, n_sample = gt.subsample_size(labels, size, get_length=True)

	n_sample = len_df - n_sample

	# Perform pca if specified

	if pca is not None:
	
		acp = PCA(pca)

		features = acp.fit_transform(features)

	else:

		features = features.to_numpy()

	labels = labels.to_numpy().reshape(len_df, -1)

	# Create Samples 

	if n_sample > 2:

		distance_features = cdist(features, features, metric=metric)
		distance_features = distance_features / distance_features.max()
		distance_labels = cdist(labels, labels, metric=metric)
		distance_labels = distance_labels / distance_labels.max()
		distance = distance_features + distance_labels

		return gt.max_min_distance_split(distance, n_sample)

	else:

		raise ValueError("Train sample size should be at least 2.")

# ===============================================
# || FONCTION : spxy_sampling_train_test_split ||
# ===============================================

def spxy_sampling_train_test_split(features: pd.DataFrame,
								   labels: pd.DataFrame,
								   test_size: tp.Union[int, float],
								   pca: tp.Union[int, float]=None,
								   metric: str="euclidean") -> tp.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

	"""
	FONCTION de splitting : ``spxy_sampling_train_test_split``
	--------
	
	Permet le splitting d'un jeu de données, en quatre parties : ``x_train``, ``x_test``, ``y_train``, ``y_test``

		x_train, y_train : Données d'entraînement
		x_test, y_test   : Données de test

	Paramètres
	----------

	* ``features`` : DataFrame

		DataFrame contenant les variables.

	* ``labels`` : DataFrame

		DataFrame contenant les étiquettes à prédire.

	* ``test_size`` : Int/Float

		Pourcentage d'échantillons que l'on souhaite avoir dans le test_set.

	* ``pca`` : Int/Float, default=None

		Nombre de composantes principales pour effectuer la ``PCA``. 
		 Effectue une ``PCA`` si l'argument est non-nul.

	* ``metric`` : Str, default="euclidean"

		Métrique de distance à utiliser pour les calculs.
		 Voir scipy.spatial.distance.cdist pour plus d'informations.

	Return
	------

	* ``Tuple`` : (DataFrame, DataFrame, DataFrame, DataFrame)

		Retourne un tuple de quatre éléments contenant des dataFrames.

	Exemple
	-------

	>>> x_train, x_test, y_train, y_test = spxy_sampling_train_test_split(features, labels, 0.2, None, "euclidean")
	>>> print(type(x_train))
	<class 'pandas.core.frame.DataFrame'>
	>>> print(len(features))
	108
	>>> print(len(x_train))
	86
	>>> print(len(x_test))
	22
	"""

	# Vérification des entrées

	gt.input_verification(features, labels)

	# Application de la pca + récupération des index du train_set, et du test_set
	
	index_train, index_test = spxy_sampling(test_size, features, labels, pca, metric)
	
	# Récupération des lignes des datasets, selon leurs index

	return gt.get_train_test_tuple(features, labels, index_train, index_test)