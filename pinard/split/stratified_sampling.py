#!/bin/python3
#-*- coding : utf-8 -*-

# =============================
# || Importation des modules ||
# =============================

import get_tts as gt
import numpy as np
import pandas as pd
import random as rd
import typing as tp

# ========================
# || set_table_quantile || 
# ========================

def set_table_quantile(data: pd.DataFrame, q_step: float) -> tp.List[float] :

	"""
	FONCTION : ``set_table_quantile``
	--------

	Permet d'obtenir le tableau des quantiles à utiliser pour créer des strates.

	Paramètres
	----------

	* ``data`` : DataFrame
		
		Données à utiliser pour l'obtention des strates.

	* ``q_step`` : float, default=0.1

		Valeur du quantile pour les strates, ``0.1 == division en déciles``.

	Return
	------

	* ``List`` : List[float]
		
		Retourne une liste contenant des quantiles.
		 Ils limitent les strates.

	Exemple
	-------

	>>> tab_strates = set_table_quantile(data, 0.1)
	>>> print(sorted(tab_strates))
	[10, ..., 30, ..., 100]
	>>> print(len(tab_strates))
	10
	"""

	# Pour chaque q_th, dépendant de q_step

		# Si q_th est inférieur à 1

			# Ajouter le quantile correspondant

		# Sinon

			# Ajouter le quantile pour q_th = 1.0
	
	return np.unique([np.quantile(data, q_th) if q_th < 1.0 else np.quantile(data, 1.0) for q_th in np.arange(q_step, (1+q_step), q_step)]).tolist()


def set_strate(data: pd.DataFrame, q_step: float=0.1) -> pd.DataFrame :

	"""retourne un dataframe contenant des strates"""

	# Récupérer le nom de la colonne des data

	col_name = data.columns[0]

	# Fusionner les features et les data, trier en fonction d'une colonne, changer les index en garder les index originaux

	data = data.sort_values(by=col_name).reset_index(drop=False)

	# Créer une colonne strate

	data = data.assign(strate=0)

	# Créer le tableau des quantiles

	tab_quantile = set_table_quantile(data, q_step)

	# Initialiser la strate + la limite de cette dernière

	strate = 0
	id = 0
	lim_strate =  tab_quantile[id]

	# Pour chaque éléments du dataframe

	for row in range(len(data)):

		# Si l'élément est dans l'intervalle de la strate actuelle, alors attribuer la strate à l'élément

		if data.at[row, (col_name)] < lim_strate:
			
			data.at[row, ("strate")] = strate

		# Sinon, passer à la strate suivante + ajouter l'élément dans la nouvelle strate

		else :
				
			strate+=1
			id+=1
			lim_strate = tab_quantile[id]
			
			data.at[row, ("strate")] = strate

	return data[["index", "strate"]]


def stratified_sampling(size: tp.Union[int, float],
						data: pd.DataFrame,
						q_step: float=0.1,
						random_state: tp.Union[int, None]=None) -> tp.Tuple[tp.List[int], tp.List[int]] :

	"""
	FONCTION : ``stratified_sampling``
	--------

	Permet le tirage d'échantillons dans un dataset, en fonction des strates.

	Paramètres
	----------

	* ``size`` : Int/Float

		La quantité d'échantillons à prélever, peut être exprimé soit en nombre, soit en proportion.
		
	* ``data`` : DataFrame
		
		Données à utiliser pour l'obtention des strates.

	* ``q_step`` : float, default=0.1

		Valeur du quantile pour les strates, ``0.1 == division en déciles``.

	* ``random_state`` : Int, default=None

		Valeur de la seed, pour la reproductibilité des résultats.

	Return
	------

	* ``Tuple`` : (List[int], List[int])
		
		Retourne un split entre le train_set et le test_set. 

	Exemple
	-------

	>>> index_train, index_test = stratified_sampling(0.2, data, 0.1, None)
	>>> print(sorted(index_test))
	[3, ..., 55, ..., 106]
	"""
	# Mettre en place une graine aléatoire si elle a été spécifié

	if random_state is not None :

		rd.seed(random_state)

	# Créer le dataframe des strates

	data = set_strate(data, q_step)

	# Récupérer l'effectif des différentes strates + Initialiser le tableau d'index du test_set

	dic_strate = dict(data["strate"].value_counts())

	len_data, n_sample = gt.subsample_size(data, size, get_length=True)

	index_train = [-1]*(len_data-n_sample)
	len_train = len(index_train)
	id_tr = 0

	index_test = [-1]*n_sample
	len_test = len(index_test)
	id_ts = 0

	n_sample/=len_data

	# Pour chaque strates dans le dataframe

	for strate, strate_size in dic_strate.items():

		# Définir un nombre d'éléments maximum que l'on peut prendre dans une strate
		
		cpt_rbrs = round(n_sample*strate_size)

		if cpt_rbrs == 0:

			cpt_rbrs = 1

		tab_select = [True if i < cpt_rbrs else False for i in range(strate_size)]
		rd.shuffle(tab_select)
		id_slct = 0

		#  Pour chaque éléments du dataframe

		for row in range(len_data):

			# Sortir de la boucle si on a déjà le nombre d'éléments maximum

			if data.at[row, ("strate")] == strate:
		
				if tab_select[id_slct] and id_ts < len_test:

					index_test[id_ts] = data.at[row, ("index")]
					id_ts+=1
					id_slct+=1

				elif id_tr < len_train:

					index_train[id_tr] = data.at[row, ("index")]
					id_tr+=1
					id_slct+=1
					
	return (index_train, index_test)


def stratified_sampling_train_test_split(features: pd.DataFrame,
										 labels: pd.DataFrame,
										 test_size: tp.Union[int, float],
										 q_step: float=0.1,
										 random_state: tp.Union[int, None]=None) -> tp.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

	"""
	FONCTION de splitting : ``stratified_sampling_train_test_split``
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

	>>> x_train, x_test, y_train, y_test = stratified_sampling_train_test_split(features, labels, 0.2, 0.1, None)
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

	# Récupération des index du train_set + test_set

	index_train, index_test = stratified_sampling(test_size, labels, q_step, random_state)

	# Récupération des lignes des datasets, selon leurs index

	return gt.get_train_test_tuple(features, labels, index_train, index_test)