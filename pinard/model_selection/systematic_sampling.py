#!/bin/python3
#-*- coding : utf-8 -*-

# =============================
# || Importation des modules ||
# =============================

import get_tts as gt
import pandas as pd
import random as rd
import typing as tp

# =========================
# || FONCTION : rotation ||
# =========================

def rotation(tab: tp.List[int], len_tab: int) -> tp.List[int]:

	"""
	FONCTION : ``rotation`` 
	--------

	Permet la rotation d'un tableau.

	Paramètres
	----------

	* ``tab`` : List[int]
	
		Tableau à rotationner.

	* ``len_tab`` : Int
	
		Longueur du tableau à rotationner.

	Return
	------

	* ``List`` : List[int]
	
		Retourne un tableau ayant subi une rotation.

	Exemples
	--------

	>>> import numpy as np
	>>> tab = np.arange(0, 10, 1)
	>>> print(tab)
	[0, 1, ..., 9]
	>>> tab = rotation(tab, 10)
	>>> print(tab)
	[9, 0, ..., 8]
	>>> ...
	>>> a = "Bonjour"
	>>> a = rotation(a, 7)
	>>> print(a)
	['r', 'B', 'o', 'n', 'j', 'o', 'u']
	"""

	# Retour du tableau après rotation

	return [tab[len_tab-1] if i == 0 else tab[i-1] for i in range(len_tab)]

# =========================
# || FONCTION : n_rotate ||
# =========================

def n_rotate(tab: tp.List[int], n_rotation: int, len_tab: int) -> tp.List[int]:
	
	"""
	FONCTION : ``n_rotate``
	--------

	Permet d'effectuer n rotations d'un tableau.

	Paramètres
	----------

	* ``tab`` : List[int]
	
		Tableau à rotationner.

	* ``n_rotation`` : Int
	
		Nombre de rotations à effectuer.

	* ``len_tab`` : Int
	
		Longueur du tableau à rotationner.

	Return
	------

	* ``List`` : List[Int]
	
		Retourne un tableau ayant subi une(ou plusieurs) rotation(s).

	Exemples
	--------

	>>> import numpy as np
	>>> tab = np.arange(0, 10, 1)
	>>> print(tab)
	[0, 1, ..., 9]
	>>> tab = n_rotate(tab, 2, 10)
	>>> print(tab)
	[8, 9, ..., 7]
	>>> ...
	>>> a = "Bonjour"
	>>> a = rotation(a, 2, 7)
	>>> print(a)
	['u', 'r', 'B', 'o', 'n', 'j', 'o']
	"""
	
	# Application des n rotations

	for _ in range(n_rotation):
	
		tab = rotation(tab, len_tab)
	
	return tab

# ====================================
# || FONCTION : systematic_sampling ||
# ====================================

def systematic_sampling(size: tp.Union[int, float],
						data: pd.DataFrame,
						random_state: tp.Union[int, None]=None) -> tp.Tuple[tp.List[int], tp.List[int]] :

	"""
	FONCTION d'échantillonnage : ``systematic_sampling``
	--------

	Permet d'effectuer un échantillonnage non-aléatoire, basé sur la méthode d'échantillonnage systématique circulaire.
	
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
	
	# Mettre en place une graine aléatoire, si elle a été spécifié

	if random_state is not None :

		rd.seed(random_state)

	# Récupérer le nom de la colonne des labels

	col_name = data.columns[0]

	# Récupérer les index après les avoir trier en fonction d'une colonne

	len_data, n_sample = gt.subsample_size(data, size, get_length=True)

	tab_index = data.sort_values(by=col_name).reset_index(drop=False)["index"].tolist()

	# Rotationner le tableau n_rotation de fois

	n_rotation = rd.randint(0, len_data-1)

	tab_index = n_rotate(tab_index, n_rotation, len_data)

	# Sélectionner les éléments du test_set, en fonction du pas défini

	pas = round(len_data/n_sample)

	index_test = tab_index[0 : len_data : pas]

	# Récupérer les éléments du train_set

	index_train = [id for id in tab_index if id not in index_test]

	return (index_train, index_test)

# =====================================================
# || FONCTION : systematic_sampling_train_test_split ||
# =====================================================

def systematic_sampling_train_test_split(features: pd.DataFrame,
										 labels: pd.DataFrame,
										 test_size: tp.Union[int, float],
										 random_state: tp.Union[int, None]=None) -> tp.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

	"""
	FONCTION de splitting : ``systematic_sampling_train_test_split``
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
	
		La quantité d'échantillons à prélever, peut être exprimé soit en nombre, soit en proportion.

	* ``random_state`` : Int, default=None

		Valeur de la seed, pour la reproductibilité des résultats.

	Return
	------

	* ``Tuple`` : (DataFrame, DataFrame, DataFrame, DataFrame)
	
		Retourne un tuple de quatre éléments contenant des DataFrames

	Exemple
	-------

	>>> x_train, x_test, y_train, y_test = systematic_sampling_train_test_split(features, labels, 0.2, 1)
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

	index_train, index_test = systematic_sampling(test_size, labels, random_state)

	# Récupération des lignes des datasets, selon leurs index

	return gt.get_train_test_tuple(features, labels, index_train, index_test)