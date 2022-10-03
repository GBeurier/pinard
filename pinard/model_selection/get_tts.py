#!/bin/python3
#-*- coding : utf-8 -*-

# =============================
# || Importation des modules ||
# =============================

import numpy as np
import pandas as pd
import typing as tp

# ===================================
# || FONCTION : input_verification ||
# ===================================   

def input_verification(features: pd.DataFrame, labels: pd.DataFrame) -> None:

	"""
	FONCTION de vérification : ``input_verification``
	--------
	
	Permet de vérifier si les données en entrée sont bien des objets de type : <class 'pandas.core.frame.DataFrame'>. 

	Paramètres
	----------

	* ``features`` : DataFrame

		DataFrame contenant les variables.

	* ``labels`` : DataFrame

		DataFrame contenant les étiquettes à prédire.

	Return
	------

	* ``Exceptation`` : TypeError

		Retourne une erreur de type ``TypeError`` si les entrées ne sont pas des dataframes.

	Exemple
	-------

	>>> name = "dataframe.csv"
	>>> df = pd.read_csv(name)
	>>> ...
	>>> features = df.iloc[:, 0:df.shape[1]-1]
	>>> print(type(features))
	<class 'pandas.core.frame.DataFrame'>
	>>> ...
	>>> labels = df.iloc[:, df.shape[1]-1:df.shape[1]]
	>>> print(type(labels))
	<class 'pandas.core.frame.DataFrame'>
	>>> ...
	>>> input_verifiaction(list(features), list(labels))
	TypeError: "Parameters features and labels have to be objects of type : <class 'pandas.core.frame.DataFrame'>."
	"""

	if not isinstance(features, pd.DataFrame) or not isinstance(labels, pd.DataFrame):

		raise TypeError("Arguments 'features' and 'labels' must be objects of type : <class 'pandas.core.frame.DataFrame'>.")

# ===============================
# || FONCTION : subsample_size ||
# ===============================

def subsample_size(data: pd.DataFrame, size: tp.Union[int, float], get_length: bool=True) -> tp.Union[tp.Tuple[int, int], int]:
	
	"""
	FONCTION : ``subsample_size``
	--------
	
	Fonction renvoyant le nombre d'échantillons à prélever.

	Paramètres
	----------

	* ``data`` : DataFrame

		DataFrame contenant les échantillons.

	* ``size`` : Int/Float

		La quantité d'échantillons à prélever, peut être exprimé soit en nombre, soit en proportion.

	* ``get_length`` : Bool, default=True

		Définie si on récupère la longueur de ``data``.

	Return
	------

	* ``Tuple`` : (int, int)

		Retourne le tuple (len(data), nombre d'échantillon)

	* ``Int`` : int

		Retourne le nombre d'échantillon à prélever.

	* ``Exceptation`` : ValueError

		Retourne une erreur de type ``ValueError`` si le nombre à prélever est négative, ou s'il vaut zéro,
		 ou s'il dépasse la quantité d'échantillons disponibles.

	Exemple
	-------

	>>> len_data, resample_size = subsample_size(0.2, data, True)
	>>> print(len_data)
	100
	>>> print(resample_size)
	20
	"""	

	# Si la quantité est donné en proportion

	if size < 1.0 :

		if get_length:

			len_data = len(data)

			return (len_data, round(len_data * size))

		else:

			return round(len(data) * (size))

	# Si la quantité est donnée en nombre

	elif size > 1.0 :

		if get_length:

			return (len(data), round(size))

		else:
			
			return round(size)
	
	# Renvoyer une erreur dans tout les autres cas

	else :

		raise ValueError("Size must be positive, between (0 ; len(data)].")

# =======================================
# || FUNCTION : max_min_distance_split ||
# =======================================

def max_min_distance_split(distance: np.ndarray, train_size: int) -> tp.Tuple[tp.List[int], tp.List[int]]:
	
	"""
	FUNCTION : ``max_min_distance_split``
	--------

	Sample set split method based on maximun minimun distance, which is the core of Kennard Stone
	 method.

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

	index_train = []
	index_test = [x for x in range(distance.shape[0])]

	# First select 2 farthest points

	first_2pts = np.unravel_index(np.argmax(distance), distance.shape)
	index_train.append(first_2pts[0])
	index_train.append(first_2pts[1])

	# Remove the first 2 points from the remaining list

	index_test.remove(first_2pts[0])
	index_test.remove(first_2pts[1])

	for i in range(train_size - 2):

		# Find the maximum minimum distance

		select_distance = distance[index_train, :]
		min_distance = select_distance[:, index_test]
		min_distance = np.min(min_distance, axis=0)
		max_min_distance = np.max(min_distance)

		# Select the first point (in case that several distances are the same, choose the first one)

		points = np.argwhere(select_distance == max_min_distance)[:, 1].tolist()

		for point in points:

			if point in index_train:
				pass

			else:

				index_train.append(point)
				index_test.remove(point)
				break

	return (index_train, index_test)

# =====================================
# || FONCTION : get_train_test_tuple ||
# =====================================

def get_train_test_tuple(features: pd.DataFrame,
						 labels: pd.DataFrame,
						 index_train: tp.List[int],
						 index_test: tp.List[int]) -> tp.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

	"""
	FONCTION de partitionnement : ``get_train_test_tuple``
	--------
	
	Permet le splitting d'un jeu de données, en quatre parties : ``x_train``, ``x_test``, ``y_train``, ``y_test``.

	Paramètres
	----------

	* ``features`` : DataFrame

		DataFrame contenant les variables.

	* ``labels`` : DataFrame

		DataFrame contenant les étiquettes à prédire.

	* ``index_train`` : List of int

		Tableau contenant les indices pour le train_set.

	* ``index_test`` : List of int

		Tableau contenant les indices pour le test_set.

	Return
	------

	* ``Tuple`` : (DataFrame, DataFrame, DataFrame, DataFrame)

		Retourne un tuple de quatre éléments contenant des dataFrames.

	Exemples
	--------

	>>> print(type(features))
	<class 'pandas.core.frame.DataFrame'>
	>>> x_train, x_test, y_train, y_test = get_train_test_tuple(features, labels, index_train, index_test)
	>>> print(type(x_train))
	<class 'pandas.core.frame.DataFrame'>
	"""

	# Commencer par récupérer les jeux de test et d'entraînement

	x_train, y_train = features.iloc[index_train], labels.iloc[index_train]
	x_test, y_test = features.iloc[index_test], labels.iloc[index_test]

	# Trier selon les labels

	train = pd.concat([x_train, y_train], axis=1).sort_values(by=labels.columns[0]).reset_index(drop=True)
	test = pd.concat([x_test, y_test], axis=1).sort_values(by=labels.columns[0]).reset_index(drop=True)

	# Reséparer les jeux de test et d'entraînement

	shape_trainning_var = features.shape[1]

	x_train, y_train = train.iloc[:, 0:shape_trainning_var], train.iloc[:, shape_trainning_var:]
	x_test, y_test = test.iloc[:, 0:shape_trainning_var], test.iloc[:, shape_trainning_var:]

	return (x_train, x_test, y_train, y_test)