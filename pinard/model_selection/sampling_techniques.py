#!/bin/python3
#-*- coding : utf-8 -*-

# ============================= 
# || Importation des modules || 
# ============================= 

import pandas as pd
import typing as tp
from k_mean_sampling import kmean_sampling_train_test_split, kmean_sampling
from kennard_stone_sampling import ks_sampling_train_test_split, ks_sampling
from random_sampling import random_sampling_train_test_split, random_sampling
from SPlit_sampling import split_sampling_train_test_split, split_sampling
from spxy_sampling import spxy_sampling_train_test_split, spxy_sampling
from stratified_sampling import stratified_sampling_train_test_split, stratified_sampling
from systematic_sampling import systematic_sampling_train_test_split, systematic_sampling

# =============================== 
# || FONCTION : sampling_index ||
# =============================== 

def sampling_index(features: pd.DataFrame,
				   labels: pd.DataFrame,
				   test_size: tp.Union[int, float],
				   tech: str="random",
				   random_state: tp.Union[int, None]=None,
				   metric: str="euclidean",
				   pca: tp.Union[int, float]=None,
				   q_step: float=0.1) -> tp.Tuple[tp.List[int], tp.List[int]]:

	"""
	FONCTION d'échantillonnage : ``sampling_index``
	--------
	
	Permet l'obtention des index du train_set, et du test_set, après séparation selon une technique de splitting.

	Paramètres
	----------

	* ``features`` : DataFrame

		DataFrame contenant les variables.

	* ``labels`` : DataFrame

		DataFrame contenant les étiquettes à prédire.

	* ``test_size`` : Int/Float

		La quantité d'échantillons à prélever, peut être exprimé soit en nombre, soit en proportion.

	* ``tech`` : String, default="random"

		La technique d'échantillonnage à utiliser.

		 liste des techniques disponibles :

			["`k_mean`", "`kennard_stone`", "`random`", "`SPlit`", "`spxy`", "`Stratified`", "`systematic`"]

	* ``random_state`` : Int, default=None

		Valeur de la seed, pour la reproductibilité des résultats.

	* ``pca`` : Int/Float, default=None

		Nombre de composantes principales pour effectuer une `PCA`.
		 Cette option n'est disponible que pour : `k_mean`, `kennard_stone`, `spxy`.

	* ``metric`` : Str, default="euclidean"

		Metrique pour les calculs de distances. Utile pour les méthodes : `kennard_stone`, `spxy`.
		 Pour plus d'informations, voir ``scipy.spatial.distance.cdist``.

	* ``q_step`` : Float, default=0.1

		Valeur du quantile à utiliser pour la méthode `stratified`.

	Return
	------

	* ``Tuple`` : (List[int], List[int])

		Retourne un tuple de deux éléments contenant des listes d'entiers.
		 Ils contiennent les index du train_set, et du test_set, après séparation.

	* ``Exceptation`` : ValueError

		Retourne une erreur de type ``ValueError`` si la technique demandée n'est pas implémentée.

	Exemple
	-------

	>>> index_train, index_test = sampling_train_test_split(features, labels, 0.2, "k_mean")
	>>> print(type(index_train))
	<class 'list'>
	>>> print(len(features))
	108
	>>> print(len(index_train))
	86
	>>> print(len(index_test))
	22
	"""

	if tech == "k_mean":
		
		return kmean_sampling(test_size, features, pca, random_state)
		
	elif tech == "kennard_stone":
		
		return ks_sampling(test_size, features, pca, metric)
				
	elif tech == "random":

		return random_sampling(test_size, labels, random_state)

	elif tech == "SPlit":
		
		return split_sampling(test_size, features, random_state)

	elif tech == "spxy":
		
		return spxy_sampling(test_size, features, labels, pca, metric)

	elif tech == "stratified":

		return stratified_sampling(test_size, labels, q_step, random_state)
		
	elif tech == "systematic":
		
		return systematic_sampling(test_size, labels, random_state)

	else:

		raise ValueError("Argument 'tech' must be : ['k_mean' ; 'kennard_stone' ; 'random' ; 'SPlit' ; 'spxy' ; 'stratified' ; 'systematic'].")

# ========================================== 
# || FONCTION : sampling_train_test_split || 
# ========================================== 

def sampling_train_test_split(features: pd.DataFrame,
							  labels: pd.DataFrame,
							  test_size: tp.Union[int, float],
							  tech: str="random",
							  random_state: tp.Union[int, None]=None,
							  pca: tp.Union[int, float]=None,
							  metric: str="euclidean",
							  q_step: float=0.1) -> tp.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] :

	"""
	FONCTION de splitting : ``sampling_train_test_split``
	--------
	
	Permet le splitting d'un jeu de données, en quatre parties (`x_train`, `x_test`, `y_train`, `y_test`) selon la technique choisie.
	
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

	* ``tech`` : String, default="random"

		La technique d'échantillonnage à utiliser.

		 liste des techniques disponibles :

			["`k_mean`", "`kennard_stone`", "`random`", "`SPlit`", "`spxy`", "`Stratified`", "`systematic`"]

	* ``random_state`` : Int, default=None

		Valeur de la seed, pour la reproductibilité des résultats.

	* ``pca`` : Int/Float, default=None

		Nombre de composantes principales pour effectuer une `PCA`.
		 Cette option n'est disponible que pour : `k_mean`, `kennard_stone`, `spxy`.

	* ``metric`` : Str, default="euclidean"

		Metrique pour les calculs de distances. Utile pour les méthodes : `kennard_stone`, `spxy`.
		 Pour plus d'informations, voir scipy.spatial.distance.cdist.

	* ``q_step`` : Float, default=0.1

		Valeur du quantile à utiliser pour la méthode `stratified`.

	Return
	------

	* ``Tuple`` : (DataFrame, DataFrame, DataFrame, DataFrame)

		Retourne un tuple de quatre éléments contenant des DataFrames.

	* ``Exceptation`` : ValueError

		Retourne une erreur de type ``ValueError`` si la technique demandée n'est pas implémentée.

	Exemple
	-------

	>>> x_train, x_test, y_train, y_test = sampling_train_test_split(features, labels, 0.2, "k_mean")
	>>> print(type(x_train))
	<class 'pandas.core.frame.DataFrame'>
	>>> print(len(features))
	108
	>>> print(len(x_train))
	86
	>>> print(len(x_test))
	22
	"""
		
	if tech == "k_mean":
		
		return kmean_sampling_train_test_split(features, labels, test_size, pca, random_state)
		
	elif tech == "kennard_stone":
		
		return ks_sampling_train_test_split(features, labels, test_size, pca, metric)
				
	elif tech == "random":

		return random_sampling_train_test_split(features, labels, test_size, random_state)

	elif tech == "SPlit":
		
		return split_sampling_train_test_split(features, labels, test_size, random_state)

	elif tech == "spxy":
		
		return spxy_sampling_train_test_split(features, labels, test_size, pca, metric)

	elif tech == "stratified":

		return stratified_sampling_train_test_split(features, labels, test_size, q_step, random_state)
		
	elif tech == "systematic":
		
		return systematic_sampling_train_test_split(features, labels, test_size, random_state)

	else:

		raise ValueError("Argument 'tech' must be : ['k_mean' ; 'kennard_stone' ; 'random' ; 'SPlit' ; 'spxy' ; 'stratified' ; 'systematic'].")

# ===================================== 
# || FONCTION : train_val_test_split || 
# ===================================== 

def sampling_train_val_test_split(features: pd.DataFrame,
						 		  labels: pd.DataFrame,
						 		  val_size: tp.Union[int, float],
						 		  test_size: tp.Union[int, float],
						 		  tech: str="random",
						 		  random_state_test: tp.Union[int, None]=None,
						 		  random_state_val: tp.Union[int, None]=None,
						 		  pca: tp.Union[int, float]=None,
						 		  metric: str="euclidian",
						 		  q_step: float=0.1) -> tp.Union[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] :

	"""
	FONCTION de splitting : ``train_val_test_split``
	--------
	
	Permet le splitting d'un jeu de données, en six parties : x_train, x_val, x_test, y_train, y_val, y_test

		x_train, y_train : Données d'entraînement
		x_val, y_val     : Données de validation
		x_test, y_test   : Données de test

	Paramètres
	----------

	* ``features`` : DataFrame

		DataFrame contenant les variables.

	* ``labels`` : DataFrame

		DataFrame contenant les étiquettes à prédire.

	* ``val_size`` : Int/Float

		La quantité d'échantillons à prélever, dans le validation_set, peut être exprimé soit en nombre, soit en proportion. 

	* ``test_size`` : Int/Float

		La quantité d'échantillons à prélever, dans le test_set, peut être exprimé soit en nombre, soit en proportion.

	* ``tech`` : Str, default="random"

		La technique d'échantillonnage à utiliser.

		 liste des techniques disponibles :

			["`k_mean`", "`kennard_stone`", "`random`", "`SPlit`", "`spxy`", "`Stratified`", "`systematic`"]

	* ``random_state_test`` : Int, default=None

		Valeur de la seed, pour la reproductibilité du test_set.
		
	* ``random_state_val`` : Int, default=None

		Valeur de la seed, pour la reproductibilité du val_set.

	* ``pca`` : Int/Float, default=None

		Nombre de composantes principales pour effectuer une `PCA`.
		 Cette option n'est disponible que pour : `k_mean`, `kennard_stone`, `spxy`.

	* ``metric`` : Str, default="euclidean"

		Metrique pour les calculs de distances. Utile pour les méthodes : `kennard_stone`, `spxy`.
		 Pour plus d'informations, voir scipy.spatial.distance.cdist.

	* ``q_step`` : Float, default=0.1

		Valeur du quantile à utiliser pour la méthode `stratified`.

	Return
	------

	* ``Tuple`` : (DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame)

		Retourne un tuple de six éléments contenant des DataFrames.

	* ``Exceptation`` : ValueError

		Retourne une erreur de type ``ValueError`` si la technique demandée n'est pas implémentée.

	Exemple
	-------

	>>> x_train, x_test, x_val, y_train, y_val, y_test = train_val_test_split(features, labels, 20, 20, "k_mean")
	>>> print(type(x_train))
	<class 'pandas.core.frame.DataFrame'>
	>>> print(len(features))
	108
	>>> print(len(x_train))
	68
	>>> print(len(x_val))
	20
	>>> print(len(x_test))
	20
	"""

	if tech not in ["cluster", "k_mean", "kennard_stone", "outsider", "random", "SPlit", "spxy", "systematic"]:

		raise ValueError("Argument 'tech' must be : ['k_mean' ; 'kennard_stone' ; 'random' ; 'SPlit' ; 'spxy' ; 'stratified' ; 'systematic'].")
		
	if tech == "k_mean":
		
		x, x_test, y, y_test = kmean_sampling_train_test_split(features, labels, test_size, pca, random_state_test)
		x_train, x_val, y_train, y_val = kmean_sampling_train_test_split(x, y, val_size,  pca, random_state_val)
		
	elif tech == "kennard_stone":
		
		x, x_test, y, y_test = ks_sampling_train_test_split(features, labels, test_size, pca, metric)
		x_train, x_val, y_train, y_val = ks_sampling_train_test_split(x, y, val_size)

	if tech == "random":
		
		x, x_test, y, y_test = random_sampling_train_test_split(features, labels, test_size, random_state_test)
		x_train, x_val, y_train, y_val = random_sampling_train_test_split(x, y, val_size, random_state_val)

	elif tech == "SPlit":
		
		x, x_test, y, y_test = split_sampling_train_test_split(features, labels, test_size, random_state_test)
		x_train, x_val, y_train, y_val = split_sampling_train_test_split(x, y, val_size, random_state_val)

	elif tech == "spxy":

		x, x_test, y, y_test = spxy_sampling_train_test_split(features, labels, test_size, pca, metric)
		x_train, x_val, y_train, y_val = spxy_sampling_train_test_split(x, y, val_size, pca, metric)

	elif tech == "stratified":

		x, x_test, y, y_test = stratified_sampling_train_test_split(features, labels, test_size, q_step, random_state_test)
		x_train, x_val, y_train, y_val = stratified_sampling_train_test_split(x, y, val_size, q_step, random_state_val)

	elif tech == "systematic":
		
		x, x_test, y, y_test = systematic_sampling_train_test_split(features, labels, test_size, random_state_test)
		x_train, x_val, y_train, y_val = systematic_sampling_train_test_split(x, y, val_size, random_state_val)

	return (x_train, x_val, x_test, y_train, y_val, y_test)