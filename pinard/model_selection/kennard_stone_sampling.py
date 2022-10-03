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

# ============================
# || FUNCTION : ks_sampling ||
# ============================

def ks_sampling(size: tp.Union[int, float],
				data: pd.DataFrame,
				pca: tp.Union[int, float]=None,
				metric: str="euclidean") -> tp.Tuple[tp.List[int], tp.List[int]]:

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
	
	# If test_size is a float 

	len_data, n_sample = gt.subsample_size(data, size, get_length=True)

	n_sample = len_data - n_sample

	# Perform pca if specified

	if pca is not None:
	
		acp = PCA(pca)

		data = acp.fit_transform(data)

	else:

		data = data.to_numpy()

	# Create samples

	if n_sample > 2:

		distance = cdist(data, data, metric=metric)

		return gt.max_min_distance_split(distance, n_sample)

	else:

		raise ValueError("Train sample size should be at least 2.")

# =============================================
# || FONCTION : ks_sampling_train_test_split ||
# =============================================

def ks_sampling_train_test_split(features: pd.DataFrame,
								 labels: pd.DataFrame,
								 test_size: tp.Union[int, float],
								 pca: tp.Union[int, float]=None,
								 metric: str="euclidean") -> tp.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

	"""
	FONCTION de splitting : ``ks_sampling_train_test_split``
	--------
	
	Permet le splitting d'un jeu de données, en quatre parties : x_train, x_test, y_train, y_test

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
		
		Retourne un tuple de quatre éléments contenant des DataFrames.

	Exemple
	-------

	>>> x_train, x_test, y_train, y_test = ks_sampling_train_test_split(features, labels, 0.2, None, "euclidean")
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
		
	index_train, index_test = ks_sampling(test_size, features, pca, metric) 
	
	# Récupération des lignes des datasets, selon leurs index

	return gt.get_train_test_tuple(features, labels, index_train, index_test)