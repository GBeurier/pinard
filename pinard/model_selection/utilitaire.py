#!/bin/python3
#-*- coding : utf-8 -*-

# =============================
# || Importation des modules ||
# =============================

import get_tts as gt
import math_mod as mm
import numpy as np
import pandas as pd
import sampling_techniques as st
import typing as tp
from sklearn.decomposition import PCA
from statistics import mean, stdev
		
# ======================================
# || FONCTION : compute_dissimilarity ||
# ======================================

def compute_dissimilarity(train: pd.DataFrame, test: pd.DataFrame) -> float:

	"""
	FONCTION : ``compute_dissimilarity``
	--------

	Permet le calcul de la dissimilarité entre un train et un test set.

	Paramètres
	----------

	* ``train`` : DataFrame

		Dataframe contenant le jeu d'entraînement.

	* ``test`` : DataFrame

		Dataframe contenant le jeu de test.

	Return
	------

	* ``Float``
	
		Retourne la valeur de la dissimilarité. 

	Exemple
	-------

	>>> import sampling_techniques as st
	>>> import utilitaire as ut
	>>> x_train, x_test, y_train, y_test = st.sampling_train_test_split(features, labels, 0.2, "spxy")
	>>> print(ut.compute_dissimilarity(x_train, x_test))
	-0.0012258291244506836
	"""

	n_dims = test.shape[1]

	dic_train = {"mean" : [-1]*n_dims,
				 "std" : [-1]*n_dims,
				 "q_25" : [-1]*n_dims,
				 "q_50" : [-1]*n_dims,
				 "q_75" : [-1]*n_dims,
				 "q_100" :[-1]*n_dims}

	dic_test = {"mean" : [-1]*n_dims,
				"std" : [-1]*n_dims,
				"q_25" : [-1]*n_dims,
				"q_50" : [-1]*n_dims,
				"q_75" : [-1]*n_dims,
				"q_100" :[-1]*n_dims}

	# vec_train_mean = [-1]*n_dims
	# vec_train_std = [-1]*n_dims
	# vec_test_mean = [-1]*n_dims
	# vec_test_std = [-1]*n_dims

	for i in range(n_dims):

		train_tmp = train.iloc[:, i].values.tolist()
		test_tmp = test.iloc[:, i].values.tolist()

		dic_train["mean"][i] = mean(train_tmp)
		dic_test["mean"][i] = mean(test_tmp)

		dic_train["std"][i] = stdev(train_tmp)
		dic_test["std"][i] = stdev(test_tmp)

		dic_train["q_25"][i] = np.quantile(q=0.25, a=train_tmp)
		dic_test["q_25"][i] = np.quantile(q=0.25, a=test_tmp)

		dic_train["q_50"][i] = np.quantile(q=0.50, a=train_tmp)
		dic_test["q_50"][i] = np.quantile(q=0.50, a=test_tmp)

		dic_train["q_75"][i] = np.quantile(q=0.75, a=train_tmp)
		dic_test["q_75"][i] = np.quantile(q=0.75, a=test_tmp)

		dic_train["q_100"][i] = np.quantile(q=1.0, a=train_tmp)
		dic_test["q_100"][i] = np.quantile(q=1.0, a=test_tmp)

		# vec_train_mean[i] = mean(train.iloc[:, i].values.tolist())
		# vec_test_mean[i] = mean(test.iloc[:, i].values.tolist())
		# vec_train_std[i] = stdev(train.iloc[:, i].values.tolist())
		# vec_test_std[i] = stdev(test.iloc[:, i].values.tolist())

	diss_mean = mm.dist_func(dic_train["mean"], dic_test["mean"])
	diss_std = mm.dist_func(dic_train["std"], dic_test["std"])
	diss_q_25 = mm.dist_func(dic_train["q_25"], dic_test["q_25"])
	diss_q_50 = mm.dist_func(dic_train["q_50"], dic_test["q_50"])
	diss_q_75 = mm.dist_func(dic_train["q_75"], dic_test["q_75"])
	diss_q_100 = mm.dist_func(dic_train["q_100"], dic_test["q_100"])

	return mm.dist_func((diss_mean, diss_std, diss_q_25, diss_q_50, diss_q_75, diss_q_100), (0, 0, 0, 0, 0, 0))

# =========================
# || FONCTION : eval_dis ||
# =========================

def eval_dis(features: pd.DataFrame,
			 labels: pd.DataFrame,
			 test_size: tp.Union[int, float],
			 n_iter: int,
			 tab_tech: tp.List[str],
			 pca: tp.Union[int, None]=None,
			 metric: str="euclidean",
			 q_step: float=0.1) -> pd.DataFrame :

	"""
	FONCTION : ``eval_dis``
	--------

	Permet l'obtention dataframe contenant le score de dissimilarité en fonction de la technique de splitting et du random_state.

	Paramètres
	----------

	* ``features`` : DataFrame

		Dataframe contenant les features.

	* ``labels`` : DataFrame

		Dataframe contenant les labels.

	* ``test_size`` : Int/Float

		Quantité d'échantillons pour le test_set.

	* ``n_iter`` : Int

		Le nombre de configurations que l'on teste.

	* ``tab_tech`` : List[string]

		Tableau contenant les techniques de splitting à utiliser.

	* ``pca`` : Int/Float, default=None

		Nombre de composantes principales pour une éventuelle `PCA`.
		 Cette option n'est utile que pour les techniques `k_mean`, `kennard_stone`, et `spxy`.

	* ``metric`` : Str, default="euclidean"

		Metrique pour les calculs de distances. Utile pour les méthodes : `kennard_stone`, `spxy`.
		 Pour plus d'informations, voir ``scipy.spatial.distance.cdist``.

	* ``q_step`` : Float, default=0.1

		Valeur du quantile à utiliser pour la méthode `stratified`.

	Return
	------

	* ``DataFrame`` : DataFrame
	
		Retourne un dataframe contenant le score de dissimilarité en fonction de la technique de splitting et du random_state.

	Exemple
	-------

	>>> df_res = eval_dis(features=features, labels=labels, test_size=0.2, n_iter=1, tab_tech=["k_mean"])
	>>> print(type(df_res))
	<class 'pandas.core.frame.DataFrame'>
	"""

	# Définition du dictionnaire de dissimilarité

	size_tab = (n_iter*(len(tab_tech)-2))+2

	tab_01 = [-1]*size_tab
	tab_02 = [-1]*size_tab
	tab_03 = [-1]*size_tab
	id = 0

	# Pour chacune des techniques 

	for tech in tab_tech:

		# Tester ``n_iter`` configurations

		for random_state in range(n_iter):

			# Kennard_stone et spxy sont totalement déterministes, il est donc inutile de les itérer plus d'une fois

			if (tech == 'kennard_stone' or tech == 'spxy') and random_state != 0:

				break
			
			# Splitting des données

			x_train, x_test, y_train, y_test = st.sampling_train_test_split(features=features,
																			labels=labels,
																			test_size=test_size,
																			tech=tech,
																			random_state=random_state,
																			pca=pca,
																			metric=metric,
																			q_step=q_step)

			# Ajout dans le dictionnaire du score de dissimilarité

			tab_01[id] = tech
			tab_02[id] = random_state
			tab_03[id] = compute_dissimilarity(y_train, y_test)
			id+=1

	tab_01 = pd.DataFrame(data=tab_01, columns=["Technique"])
	tab_02 = pd.DataFrame(data=tab_02, columns=["Random_state"])
	tab_03 = pd.DataFrame(data=tab_03, columns=["Score"])

	return pd.concat([tab_01, tab_02, tab_03], axis=1)
	
# ============================
# || FONCTION : new_line_df ||
# ============================

def new_line_df(len_col: int) -> pd.DataFrame:

	"""
	FONCTION : ``new_line_df``
	--------

	Permet la création d'un dataframe à ``len_col`` colonnes + 1 colonne ``tech`` + 1 colonne ``random_state``. 
	 Note: cette fonction n'est utile que pour la fonction ``"splitting_state_to_csv"``.

	Params
	------

	* ``len_col`` : Int

		Nombre de colonnes à créer.

	return
	------

	* ``DataFrame`` : DataFrame

		Retourne un dataframe.

	Exemple
	-------

	>>> new_line = new_line_df(3)
	>>> print(type(new_line))
	<class 'pandas.core.frame.DataFrame'>
	>>> print(new_line.shape)
	(1, 3)
	"""
	
	# Création d'une ligne à "len_col" colonnes

	df = pd.DataFrame(data=[[False]*len_col], columns=[i for i in range(len_col)])

	# Ajout des colonnes "tech" et "random_state"

	df = df.assign(Technique=None)
	df = df.assign(Random_state=0)

	return df

# =======================================
# || FONCTION : splitting_state_to_csv ||
# =======================================

def splitting_state_to_csv(features: pd.DataFrame,
						   labels: pd.DataFrame,
						   test_size: tp.Union[int, float],
						   n_iter: int,
						   tab_tech: tp.List[str],
						   pca: tp.Union[int, None]=None,
						   metric: str="euclidean",
						   q_step: float=0.1,
						   save: str="splitting_state.csv") -> pd.DataFrame:

	"""
	FONCTION : ``splitting_state_to_csv``
	--------

	Permet de sauvegarder les différents splits entre les jeux de train, et de test, dans un fichier csv. 
	 Retourne également un dataframe.

	Params
	------

	* ``features`` : DataFrame

		Dataframe contenant les features.

	* ``labels`` : DataFrame

		Dataframe contenant les labels.

	* ``test_size`` : Int/Float

		Quantité d'échantillons pour le test_set.

	* ``n_iter`` : Int

		Le nombre de configurations que l'on teste.

	* ``tab_tech`` : List[string]

		Tableau contenant les techniques de splitting à utiliser.

	* ``pca`` : Int/Float, default=None

		Nombre de composantes principales pour une éventuelle `PCA`.
		 Cette option n'est utile que pour les techniques `k_mean`, `kennard_stone`, et `spxy`.

	* ``metric`` : Str, default="euclidean"

		Metrique pour les calculs de distances. Utile pour les méthodes : `kennard_stone`, `spxy`.
		 Pour plus d'informations, voir ``scipy.spatial.distance.cdist``.

	* ``q_step`` : Float, default=0.1

		Valeur du quantile à utiliser pour la méthode `stratified`.
	
	* ``save`` : bool, default=True

		Si on sauvegarde ou pas les résultats.

	return
	------

	* ``DataFrame`` : DataFrame

		Retourne un dataframe dont les colonnes jusqu'à ``len(features) - 2 `` sont les index des échantillons.
		 Et les deux dernières colonnes sont(respectivement) la techniques de splitting, et les random_state.

	Exemple
	-------

	>>> df = splitting_state_to_csv(features, labels, n_iter, test_size, tab_tech, n_cluster, pca)
	>>> print(type(df))
	<class 'pandas.core.frame.DataFrame'>
	"""

	# Récupérer le futur nombre de colonnes

	len_col = len(features)

	# Initier la première ligne du dataframe 

	df = new_line_df(len_col)

	# Itérateur d'indice

	i = 0

	# Pour chacunes des techniques de splitting

	for tech in tab_tech:
		
		# Faire ``n_iter`` configurations

		for random_state in range(n_iter):


			if (tech == "kennard_stone" or tech == "spxy") and random_state != 0:
				
				# On ne les itèrent qu'une fois, car ils donnent toujours les mêmes résultats

				break
			
			# Splitting des données
			
			index_train, index_test = st.sampling_index(features=features,
														labels=labels,
														test_size=test_size,
														tech=tech,
														random_state=random_state,
														pca=pca,
														q_step=q_step,
														metric=metric)

			# Ajouter des nouvelles lignes dans le dataframes jusqu'à obtenir le nombre voulu

			if i == 0:

				df.iloc[i, index_test] = True
				df.loc[i, ("Random_state")] = random_state
				df.loc[i, ("Technique")] = tech

			else :

				df_tmp = new_line_df(len_col)
				df_tmp.iloc[0, index_test] = True
				df_tmp.loc[0, ("Random_state")] = random_state
				df_tmp.loc[0, ("Technique")] = tech

				df = pd.concat([df, df_tmp], axis=0, ignore_index=True)

			# Incrémenter l'itérateur

			i+=1
	
	# Si l'on souhaite sauvegarder les résultats
	
	if save is not None:

		df.to_csv(save)

	return df

# ==================================
# || FONCTION : transform_results ||
# ==================================

def transform_results(dic_res: tp.Dict[tp.Tuple[tp.Any, tp.Any], tp.Any],
					  col_name_01: str="Technique",
					  col_name_02: str="Random_state",
					  col_name_03: str="Score") -> pd.DataFrame :

	"""
	FONCTION : ``transform_results``
	--------

	Permet l'obtention d'un dataframe, d'un dictionnaire de la forme :

		``dict({(a, b) : c})``

	Paramètres
	----------

	* ``dic_res`` : Dictionnary of type ``{(a, b) : c}``

		Dictionnaire dont les clés sont des tuples, et les valeurs sont quelconques.

	* ``col_name_01`` : Str, default=technique

		Nom de la colonne 1 du dataframe.

	* ``col_name_02`` : Str, default="r_state"

		Nom de la colonne 2 du dataframe.

	* ``col_name_03`` : Str, default="scoring"

		Nom de la colone 3 du dataframe.

	Returns
	-------

	* ``DataFrame`` : DataFrame
	
		Retourne un dataframe de ``len(dic_res)`` lignes et trois colonnes : [``"a"``, ``"b"``, ``"c"``].

	Exemples
	--------

	>>> df = transform_results(dic_res=dic_res, col_name_01="technique", col_name_02="r_state", col_name_03="scoring")
	>>> print(type(df))
	<class 'pandas.core.frame.DataFrame'>
	"""
	# Définition des tableaux

	len_dic = len(dic_res)

	tab_tech = [-1]*len_dic
	tab_r_state = [-1]*len_dic
	tab_score = [-1]*len_dic

	# Itération du dictionnaire

	i = 0

	for keys, values in dic_res.items():

		tab_tech[i] = keys[0]
		tab_r_state[i] = keys[1]
		tab_score[i] = values
		i+=1

	df_tech = pd.DataFrame(data=tab_tech, columns=[col_name_01])
	df_r_state = pd.DataFrame(data=tab_r_state, columns=[col_name_02])
	df_score = pd.DataFrame(data=tab_score, columns=[col_name_03])

	return pd.concat([df_tech, df_r_state, df_score], axis=1)

# =============================
# || FONCTION : get_df_diss ||
# =============================

def get_df_diss(df: pd.DataFrame, features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame :

	"""
	FONCTION : ``get_df_diss``
	--------

	Permet l'obtention d'un dataframe contenant le score de dissimilarité en fonction de la technique de splitting et du random_state.

	Params
	------

	* ``df`` : DataFrame

		Dataframe contenant les splittings entre le train, et le test set, selon la technique, et selon le random_state.

	* ``features`` : DataFrame
		
		DataFrame contenant les variables.

	* ``labels`` : DataFrame
		
		DataFrame contenant les étiquettes à prédire.

	return
	------

	* ``DataFrame`` : DataFrame

		Retourne un dataframe contenant le score de dissimilarité en fonction de la technique de splitting et du random_state.

	Exemple
	-------

	>>> df_res = get_df_diss(df=df)
	>>> print(type(df_res))
	<class 'pandas.core.frame.DataFrame'>
	"""
	
	# Définition du dictionnaire de dissimilarité

	dic_diss = {}

	tab_id = df.columns[0:-2].values.tolist()

	# Récupération des colonnes de type int

	tab_id = [int(i) for i in tab_id if i]

	# Pour chaque lignes(= splitting)

	for line in range(len(df)):

		# Récupérer les éléments du test_set

		index_test = [index for index in tab_id if df.iat[line, index] == True]

		# Récupérer les éléments du train_set

		index_train = [index for index in tab_id if index not in index_test]

		# Splitter selon les index obtenus

		x_train, x_test, y_train, y_test = gt.get_train_test_tuple(features, labels, index_train, index_test)

		# Calculer et ajouter dans le dictionnaire le score de dissimilarité

		dic_diss[(df.at[line, ("Technique")], df.at[line, ("Random_state")])] = compute_dissimilarity(x_train, x_test)

	return transform_results(dic_diss)