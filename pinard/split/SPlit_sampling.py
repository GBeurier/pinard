#!/bin/python3
#-*- coding : utf-8 -*-

# =============================
# || Importation des modules ||
# =============================

import get_tts as gt
import pandas as pd
import random as rd
import typing as tp
from twinning import twin

# ===============================
# || FONCTION : split_sampling ||
# ===============================

def split_sampling(size: tp.Union[int, float],
				   data: pd.DataFrame,
				   random_state: tp.Union[int, None]=None) -> tp.Tuple[tp.List[int], tp.List[int]]:

	"""
	FONCTION d'échantillonnage : ``split_sampling``
	--------

	Permet le tirage non-aléatoire d'échantillons dans un dataset, selon la méthode des supports points.

	Paramètres
	----------

	* ``size`` : Int/Float
		
		Pourcentage d'échantillons à prélever.

	* ``data`` : DataFrame
		
		Dataset dans lequel on prélève les échantillons.

	* ``random_state`` : Int, default=None
		
		Valeur de la seed, pour la reproductibilité des résultats.

	Return
	------

	* ``Tuple`` : Tuple(List[Int], List[Int])
		
		Retourne un split entre le train_set et le test_set. 

	Exemple
	-------

	>>> index_train, index_test = split_sampling(0.2, data, None)
	>>> print(sorted(index_test))
	[1, 5, ..., 49, 55, ..., 100, 105]
	"""

	# Récupération de la quantité d'échantillons pour le test_set + la longueur de data

	len_df, n_sample = gt.subsample_size(data, size, get_length=True)

	# Au cas où on dépasse la valeur de la seed autorisée
	
	if random_state not in range(len_df) :

		random_state = rd.randint(0, len_df-1)

	n_sample/=len_df 
	
	# Définition de l'inverse du ratio de splitting

	r = round(1 / n_sample)
	
	# Mise en place des tableaux d'index

	index_test = twin(data.to_numpy(), r, u1=random_state).tolist()

	index_train = [i for i in range(len_df) if i not in index_test]
	
	return (index_train, index_test)

# ================================================
# || FONCTION : split_sampling_train_test_split ||
# ================================================

def split_sampling_train_test_split(features: pd.DataFrame,
									labels: pd.DataFrame,
									test_size: tp.Union[int, float],
									random_state: tp.Union[int, None]=None) -> tp.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

	"""
	FONCTION de splitting : ``split_sampling_train_test_split``
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

	* ``test_size`` : Float

		Pourcentage d'échantillons que l'on souhaite avoir dans le test_set.

	* ``random_state`` : Int, default=None
		
		Valeur de la seed, pour la reproductibilité des résultats.

	Return
	------

	* ``Tuple`` : (DataFrame, DataFrame, DataFrame, DataFrame)

		Retourne un tuple de quatre éléments contenant des DataFrames

	Exemple
	-------

	>>> x_train, x_test, y_train, y_test = split_sampling_train_test_split(features, labels, 0.2, 1)
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
	
	index_train, index_test = split_sampling(test_size, features, random_state)

	# Récupération des lignes des dataset, selon leur index
	
	return gt.get_train_test_tuple(features, labels, index_train, index_test)