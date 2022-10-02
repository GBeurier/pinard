#!/bin/python3
#-*- coding : utf-8 -*-

# =============================
# || Importation des modules ||
# =============================

import get_tts as gt
import pandas as pd
import random as rd
import typing as tp

# ================================
# || FONCTION : random_sampling ||
# ================================

def random_sampling(size: tp.Union[int, float],
					data: pd.DataFrame,
					random_state: tp.Union[int, None]=None) -> tp.Tuple[tp.List[int], tp.List[int]]: 

	"""
	FONCTION d'échantillonnage : ``random_sampling``
	--------

	Permet le tirage aléatoire d'échantillons dans un dataset.

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

	>>> index_train, index_test = random_sampling(0.2, data, 1)
	>>> print(sorted(index_test))
	[3, ..., 55, ..., 106]
	"""

	# Mise en place de la seed
	
	if random_state is not None:

		rd.seed(random_state)

	# Définition du nombre d'échantillons à prélever

	len_data, n_sample = gt.subsample_size(data, size, get_length=True)

	# Mise en place des tableau d'index

	tab_select = [True if i < n_sample else False for i in range(len_data)]

	rd.shuffle(tab_select)

	index_train = [-1]*(len_data-n_sample)
	id_tr = 0

	index_test = [-1]*n_sample
	id_ts = 0

	# Récupération des échantillons jusqu'obtention du nombre voulu
 
	for i in range(len_data):

		if tab_select[i]:

			index_test[id_ts] = i
			id_ts+=1

		else:

			index_train[id_tr] = i
			id_tr+=1

	return (index_train, index_test)

# =================================================
# || FONCTION : random_sampling_train_test_split ||
# =================================================

def random_sampling_train_test_split(features: pd.DataFrame,
									 labels: pd.DataFrame,
									 test_size: tp.Union[int, float],
									 random_state: tp.Union[int, None]=None) -> tp.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

	"""
	FONCTION de splitting : ``random_sampling_train_test_split``
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

		La quantité d'échantillons à prélever, peut être exprimé soit en nombre, soit en proportion.

	* ``random_state`` : Int, default=None

		Valeur de la seed, pour la reproductibilité des résultats.

	Return
	------

	* ``Tuple`` : (DataFrame, DataFrame, DataFrame, DataFrame)

		Retourne un tuple de quatre éléments contenant des dataFrames.

	Exemple
	-------

	>>> x_train, x_test, y_train, y_test = random_sampling_train_test_split(features, labels, 0.2, None)
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
	
	# Récupération des index du test_set, et du train_set, à l'aide de la fonction d'échantillonnage

	index_train, index_test = random_sampling(test_size, labels, random_state)

	# Récupération des lignes des datasets, selon leurs index

	return gt.get_train_test_tuple(features, labels, index_train, index_test)