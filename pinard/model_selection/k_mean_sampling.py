#!/bin/python3
#-*- coding : utf-8 -*-

# =============================
# || Importation des modules ||
# =============================

import get_tts as gt
import numpy as np
import pandas as pd
import typing as tp
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ===============================
# || FONCTION : kmean_sampling ||
# ===============================

def kmean_sampling(size: tp.Union[int, float],
				   data: pd.DataFrame,
				   pca: tp.Union[int, float]=None,
				   random_state: tp.Union[int, None]=None) -> tp.Tuple[tp.List[int], tp.List[int]]:

	"""
	FONCTION d'échantillonnage : ``kmean_sampling``
	--------

	Permet le tirage d'échantillons dans un dataset, chaque échantillon tiré est le plus proche voisin d'un centroid initialisé en amont.

	Paramètres
	----------

	* ``size`` : Int/Float
		
		Quantité d'échantillons à prélever dans ``data`` pour le test_set.

	* ``data`` : Int
		
		Dataset dans lequel on prélève les échantillons.

	* ``pca`` : Int/Float, default=None

		Nombre de composantes principales pour effectuer la ``PCA``. Effectue une ``PCA`` si l'argument est non-nul.

	* ``random_state`` : Int, default=None

		Valeur de la seed, pour la reproductibilité des résultats.

	Return
	------

	* ``Tuple`` : (List[int], List[int])
		
		Retourne un split entre le train_set et le test_set. 

	Exemple
	-------

	>>> index_test = kmean_sampling(20, data, None, random_state=0)
	>>> print(sorted(index_test))
	[9, 10, ..., 43, 47, ..., 101, 104]
	"""
	
	# Transfromation des données selon la PCA

	if pca is not None:

		acp = PCA(pca, random_state=random_state)

		data = pd.DataFrame(data=acp.fit_transform(data), columns=[i for i in range(acp.transform(data).shape[1])])

	# Définition de l'effectif dans le train_set

	len_data, n_sample = gt.subsample_size(data, size, get_length=True)
	
	n_sample = len_data - n_sample

	index_train = [-1]*(n_sample)
	id_tr = 0

	# Génération des centroids

	kmean = KMeans(n_sample, random_state=random_state)
	kmean.fit(data)

	tab_centroids = kmean.cluster_centers_

	# Pour chacun des centroids
	
	data = data.to_numpy()

	for centroid in tab_centroids:

		# Calculer les distances qui séparent chaque points du centroid

		tab_temp = cdist(data, [centroid], metric="euclidean").tolist()

		# Ajout du point le plus proche de centroid dans le tableau d'index

		index_train[id_tr] = tab_temp.index(min(tab_temp))
		id_tr+=1
	
	# Récupération du train_set et le test_set

	index_train = np.unique(index_train).tolist()

	index_test = [id for id in range(len_data) if id not in index_train]

	return (index_train, index_test)

# ================================================
# || FONCTION : kmean_sampling_train_test_split ||
# ================================================

def kmean_sampling_train_test_split(features: pd.DataFrame,
									labels: pd.DataFrame,
									test_size: tp.Union[int, float],
									pca: tp.Union[int, float]=None,
									random_state: tp.Union[int, None]=None) -> tp.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

	"""
	FONCTION de splitting : ``kmean_sampling_train_test_split``
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

		Nombre de composantes principales pour effectuer la ``PCA``. Effectue une ``PCA`` si l'argument est non-nul.

	* ``random_state`` : Int, default=None

		Valeur de la seed, pour la reproductibilité des résultats.

	Return
	------

	* ``Tuple`` : (DataFrame, DataFrame, DataFrame, DataFrame)

		Retourne un tuple de quatre éléments contenant des DataFrames

	Exemple
	-------

	>>> x_train, x_test, y_train, y_test = kmean_sampling_train_test_split(features, labels, 0.2, None, None)
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

	# Récupération des index du test_set et du train_set, à l'aide de la fonction d'échantillonnage

	index_train, index_test = kmean_sampling(test_size, features, pca, random_state)

	# Récupération des lignes des datasets, selon leurs index

	return gt.get_train_test_tuple(features, labels, index_train, index_test)