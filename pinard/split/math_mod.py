#!/bin/python3
#-*- coding : utf-8 -*-

# =============================
# || Importation des modules ||
# =============================

import numpy as np
import typing as tp
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, median_absolute_error

# =============================
# || FONCTION : apppend_list ||
# =============================

def apppend_list(tab: list, n: tp.Any) -> list:

	"""
	FONCTION : ``apppend_list``
	--------
	
	Permet l'ajout d'un élément dans une liste. 

	Paramètres
	----------

	* ``tab`` : List of Any

		La list auquelle on va ajouter un nouvel élément.

	* ``n`` : Any

		Le nouvel élément à ajouter à la liste.

	Return
	------

	* ``List`` : List of Any

		La nouvelle liste avec le nouvel élément n.

	Exemple
	-------

	>>> tab = [1, 2, 3] 
	>>> tab = append_list(tab, 4)
	>>> print(tab)
	[1, 2, 3, 4]
	"""

	# Si la liste est vide, définir une longueur de 1, sinon, récupérer la valeur de l'ancienne liste

	if not len(tab):

		len_tab = 1

	else:

		len_tab = len(tab) + 1 

	# Créer la nouvelle liste

	new_tab = [-1]*len_tab

	# Ajouter le nouvel élément à la liste

	for i in range(len_tab):

		if i < len_tab-1:
			
			new_tab[i] = tab[i]

		else:

			new_tab[i] = n
			return new_tab

# ============================
# || FONCTION : concat_list ||
# ============================

def concat_list(tab_01: list, tab_02: list) -> list:

	"""
	FONCTION : ``concat_list``
	--------
	
	Permet la concaténation de deux listes.

	Paramètres
	----------

	* ``tab_01`` : List of Any

		La première liste.

	* ``tab_02`` : List of Any

		La seconde liste.

	Return
	------

	* ``List`` : List of Any

		La nouvelle liste, elle est la résultante de la concaténation des deux anciennes listes.

	Exemple
	-------

	>>> tab_01 = [1, 2, 3] 
	>>> tab_02 = [4, 5, 6]
	>>> new_tab = concat_list(tab_01, tab_02)
	>>> print(new_tab)
	[1, 2, 3, 4, 5, 6]
	"""

	# Création de la nouvelle liste

	len_tab_01 = len(tab_01)

	len_new_tab = (len_tab_01 + len(tab_02))
	
	new_tab = [-1]*len_new_tab
	id_tab_01 = 0
	id_tab_02 = 0

	# Concaténation des deux anciennes listes.

	for i in range(len_new_tab):

		if i < len_tab_01:

			new_tab[i] = tab_01[id_tab_01]
			id_tab_01+=1

		else:

			new_tab[i] = tab_02[id_tab_02]
			id_tab_02+=1

	return new_tab

# =======================
# || FONCTION : fusion || 
# =======================

def fusion(tab_01: list, tab_02: list) -> list:

	"""
	FONCTION : ``fusion``
	--------
	
	Fonction ``fusion`` pour le trie fusion.

	Paramètres
	----------

	* ``tab_01`` : List of Any

		La première liste.

	* ``tab_02`` : List of Any

		La seconde liste.

	Return
	------

	* ``List`` : List of Any

		La nouvelle liste trié.

	Exemple
	-------

	>>> tab_01 = [2]
	>>> tab_02 = [1]
	>>> new_tab = fusion(tab_01, tab_02)
	>>> print(new_tab)
	[1, 2]
	"""

	# Si l'une des listes est vide, la renvoyer car elle est déjà triée.

	len_tab_01 = len(tab_01)
	len_tab_02 = len(tab_02)

	if (not len_tab_01) or (not len_tab_02):

		return tab_01 or tab_02
	
	# Renvoyer la nouvelle liste triée.

	result = []
	id_tab_01, id_tab_02 = 0, 0

	while len(result) < (len_tab_01 + len_tab_02):

		if tab_01[id_tab_01] < tab_02[id_tab_02]:

			result.append(tab_01[id_tab_01])
			id_tab_01+=1

		else:

			result.append(tab_02[id_tab_02])
			id_tab_02+=1

		if (id_tab_01 == len_tab_01) or (id_tab_02 == len_tab_02):

			result.extend(tab_01[id_tab_01 : len_tab_01 : 1] or tab_02[id_tab_02 : len_tab_02 : 1])
			break
 
	return result

# ===========================
# || FONCTION : tri_fusion || 
# ===========================

def tri_fusion(tab: list) -> list:

	"""
	FONCTION : ``tri_fusion``
	--------
	
	Effectue le trie d'une liste selon la méthode du trie fusion.

	Paramètres
	----------

	* ``tab`` : List of Any

		La liste à trier.

	Return
	------

	* ``List`` : List of Any

		La nouvelle liste trié.

	Exemple
	-------

	>>> tab = [3, 2, 1]
	>>> tab = tri_fusion(tab)
	>>> print(tab)
	[1, 2, 3]
	"""

	# Si la liste ne contien qu'un élément, c'est qu'elle est déjà triée.

	len_tab = len(tab)

	if len_tab < 2:

		return tab
	
	# Casser la liste jusqu'obtenir des listes d'un élément, puis les réassembler dans le bonne ordre.

	milieu = round(len_tab/2)
	tab_01 = tri_fusion(tab[0 : milieu : 1])
	tab_02 = tri_fusion(tab[milieu : len_tab : 1])
 
	return fusion(tab_01, tab_02)

# =========================
# || FONCTION : abs_func || 
# =========================

def abs_func(n: tp.Union[int, float]) -> float:

	"""
	FONCTION : ``abs_func``
	--------

	Permet le calcul de la valeur absolue d'un nombre.

	Param
	-----

	* ``n`` : Int/Float

		Nombre dont on souhaite la valeur absolue.

	return
	------

	* ``Float`` : Float

		Retourne la valeur absolue du nombre.

	Exemple
	-------

	>>> abs_func((-2))
	2
	"""

	return (n**2)**0.5

# =========================
# || FONCTION : moy_func ||
# =========================

def moy_func(tab_values: tp.Union[tp.List[float], tp.List[int]]) -> float:

	"""
	FONCTION : ``moy_func``
	--------

	Permet le calcul de la moyenne d'un tableau de nombres.

	Param
	-----

	* ``tab_values`` : List[Float]/List[Int]

		Tableau de nombres.

	return
	------

	* ``Float`` : Float

		Retourne la moyenne du tableau.

	Exemple
	-------

	>>> moy_func([1., 2.])
	1.5
	"""

	return sum(tab_values)/len(tab_values)

# =========================
# || FONCTION : std_func ||
# =========================

def std_func(tab_values: tp.Union[tp.List[float], tp.List[int]], variance: bool=False) -> float:

	"""
	FONCTION : ``std_func``
	--------

	Permet le calcul d'écart-type d'un tableau de nombres.

	Param
	-----

	* ``tab_values`` : List[Float]/List[Int]

		Tableau de nombres.

	return
	------

	* ``Float`` : Float

		Retourne l'écart-type du tableau.

	Exemple
	-------

	>>> std_func([10., 20.])
	7.0710678118654755
	"""

	moy = moy_func(tab_values)

	if variance:

		return (sum([(n - moy)**2 for n in tab_values]) / (len(tab_values) - 1))

	else:

		return (sum([(n - moy)**2 for n in tab_values]) / (len(tab_values) - 1))**0.5

# ==========================
# || FONCTION : dist_func ||
# ==========================

def dist_func(point_01: tp.Tuple[tp.Union[float, int]],
			  point_02: tp.Tuple[tp.Union[float, int]]=None) -> float:

	"""
	FONCTION : ``dist_func``
	--------

	Permet le calcul de la distance euclidienne d'un point, par rapport à un autre point.

	Params
	------

	* ``point_01`` : Tuple[Float/Int]

		Premier point.

	* ``point_02`` : Tuple[Float/Int], default=None

		Second point.

	return
	------

	* ``Float`` : Float

		Retourne la distance la distance euclidienne d'un point, par rapport à un autre point.

	* ``Exceptation`` : ValueError

		Retourne une erreur de type ``ValueError`` si les deux points ne possèdent pas le même nombre de coordonnées.

	Exemple
	-------

	>>> print(dist_func((10, 20)))
	22.360679774997898
	"""

	len_point_01 = len(point_01)

	# S'il n'y a pas de second point, calculer la distance par rapport à l'origine

	if point_02 is None:

		point_02 = tuple(0 for _ in range(len_point_01))

		return sum([(point_01[i] - point_02[i])**2 for i in range(len_point_01)])**0.5

	# Sinon, calculer la distance entre les deux points.

	else:

		if len_point_01 == len(point_02):

			return sum([(point_01[i] - point_02[i])**2 for i in range(len_point_01)])**0.5
		
		else:

			raise ValueError("Size of point_01 must be equal to size of point_02.")

# ==========================
# || FONCTION : mape_func ||
# ==========================

def mape_func(y: tp.List[float], y_pred: tp.List[float]) -> float:

	"""
	FONCTION : ``mape_func``
	--------

	Permet le calcul de la ``mean absolute pourcentage error`` entre deux listes de floats.

	Params
	------

	* ``y`` : List[Float]

		Liste contenant les Valeurs réelles.

	* ``y_pred`` : List[float]

		Liste contenant les Valeurs prédites.

	return
	------

	* ``Float`` : Float

		Retourne la valeur de la "mape" entre les deux listes.

	Exemple
	-------

	>>> print(mape_func([1, 2, 3], [1.1, 2.2, 3.3]))
	0.10
	"""
	return moy_func([abs_func(y[i] - y_pred[i]) / y[i] for i in range(len(y)) if y[i] != 0])

# =========================
# || FONCTION : mde_func ||
# =========================

def mde_func(y: tp.List[float], y_pred: tp.List[float]) -> float:

	"""
	FONCTION : ``mde_func``
	--------

	Permet le calcul de la ``median absolute error`` entre deux listes de floats.

	Params
	------

	* ``y`` : List[Float]

		Liste contenant les Valeurs réelles.

	* ``y_pred`` : List[float]

		Liste contenant les Valeurs prédites.

	return
	------

	* ``Float`` : Float

		Retourne la valeur de la "mde" entre les deux listes.

	Exemple
	-------

	>>> print(mde_func([1, 2, 3], [1.1, 2.2, 3.3]))
	0.2
	"""

	# Trier le tableau des erreurs

	len_y = len(y)
	
	tab_errors = sorted([abs_func(y[i] - y_pred[i]) for i in range(len_y)])

	# Si la taille de la liste est paire

	if len_y % 2 == 0:

		borne_sup = round(len_y/2)

		borne_inf = borne_sup-1

		return (tab_errors[borne_inf] + tab_errors[borne_sup])/2

	# Si la taille de la liste est impaire

	else:

		mediane = round((len_y-1)/2)

		return tab_errors[mediane]

# =========================
# || FONCTION : mse_func ||
# =========================

def mse_func(y: tp.List[float], y_pred: tp.List[float]) -> float:

	"""
	FONCTION : ``mse_func``
	--------

	Permet le calcul de la ``mean squared error`` entre deux listes de floats.

	Params
	------

	* ``y`` : List[Float]

		Liste contenant les Valeurs réelles.

	* ``y_pred`` : List[float]

		Liste contenant les Valeurs prédites.

	return
	------

	* ``Float`` : Float

		Retourne la valeur de la "mse" entre les deux listes.

	Exemple
	-------

	>>> print(mse_func([1, 2, 3], [1.1, 2.2, 3.3]))
	0.04666666666666666
	"""
	return moy_func([(y[i] - y_pred[i])**2 for i in range(len(y))])

# ==========================
# || FONCTION : rmse_func ||
# ==========================

def rmse_func(y: tp.List[float], y_pred: tp.List[float]) -> float:

	"""
	FONCTION : ``rmse_func``
	--------

	Permet le calcul de la ``root mean squared error`` entre deux listes de floats.

	Params
	------

	* ``y`` : List[Float]

		Liste contenant les Valeurs réelles.

	* ``y_pred`` : List[float]

		Liste contenant les Valeurs prédites.

	return
	------

	* ``Float`` : Float

		Retourne la valeur de la "rmse" entre les deux listes.

	Exemple
	-------

	>>> print(rmse_func([1, 2, 3], [1.1, 2.2, 3.3]))
	0.21602468994692867
	"""
	return mse_func(y, y_pred)**0.5

# =========================
# || FONCTION : mae_func ||
# =========================

def mae_func(y: tp.List[float], y_pred: tp.List[float]) -> float:

	"""
	FONCTION : ``mae_func``
	--------

	Permet le calcul de la ``mean absolute error`` entre deux listes de floats.

	Params
	------

	* ``y`` : List[Float]

		Liste contenant les Valeurs réelles.

	* ``y_pred`` : List[float]

		Liste contenant les Valeurs prédites.

	return
	------

	* ``Float`` : Float

		Retourne la valeur de la "mae" entre les deux listes.

	Exemple
	-------

	>>> print(mae_func([1, 2, 3], [1.1, 2.2, 3.3]))
	0.2
	"""
	return moy_func([abs_func(y[i] - y_pred[i]) for i in range(len(y))])

# ========================
# || FONCTION : r2_func ||
# ========================

def r2_func(y: tp.List[float], y_pred: tp.List[float]) -> float:

	"""
	FONCTION : ``r2_func``
	--------

	Permet le calcul du coefficient de détermination ``R²`` entre deux listes de floats.

	Params
	------

	* ``y`` : List[Float]

		Liste contenant les Valeurs réelles.

	* ``y_pred`` : List[float]

		Liste contenant les Valeurs prédites.

	return
	------

	* ``Float`` : Float

		Retourne la valeur du "R²" entre les deux listes.

	Exemple
	-------

	>>> print(r2_func([1, 2, 3], [1.1, 2.2, 3.3]))
	0.93
	"""

	len_tab = len(y)

	moy_y = moy_func(y)

	erreur_quadratique = sum([(y[i] - y_pred[i])**2 for i in range(len_tab)])

	variance = sum([(y[i] - moy_y)**2 for i in range(len_tab)])

	return 1 - (erreur_quadratique/variance)

# ==============================
# || FONCTION : metric_choice ||
# ==============================

def metric_choice(y: tp.List[float], y_pred: tp.List[float], scoring: str="mae") -> float:

	"""
	FONCTION : ``metric_choice``
	--------

	Permet de calculer une métrique de scoring entre deux listes de floats.

	Paramètres
	----------

	* ``y`` : List[Float]

		Liste contenant les Valeurs réelles.

	* ``y_pred`` : List[Float]

		Liste contenant les Valeurs prédites.

	* ``scoring`` : String, default="mae"

		Le métrique de score à utiliser pour calculer le score.

		 Liste des métriques de score disponibles :

		 	['mae' ; 'mape' ; 'mde' ; 'mse' ; 'rmse' ; 'r2']

	Return
	------

	* ``Float`` : Float

		Retourne un score(selon la métrique choisie) entre des données réelles et des prédictions.

	Exemple
	-------

	>>> print(metric_choice([1, 2, 3], [1.1, 2.2, 3.3], scoring="mae"))
	0.2
	"""

	if scoring == "rmse":

		return mean_squared_error(y, y_pred, squared=False)

	elif scoring == "mse":

		return mean_squared_error(y, y_pred, squared=True)

	elif scoring == "mae":

		return mean_absolute_error(y, y_pred)

	elif scoring == "r2":

		return r2_score(y, y_pred)

	elif scoring == "mape":

		return mean_absolute_percentage_error(y, y_pred)

	elif scoring == "mde":

		return median_absolute_error(y, y_pred)

	else:

		raise ValueError("Argument 'scoring' must be : ['mae' ; 'mape' ; 'mde' ; 'mse' ; 'rmse' ; 'r2'].")

# ===================================
# || FONCTION : compute_score_perf ||
# ===================================

def compute_score_perf(y: tp.List[float], y_pred: tp.List[float], **kwargs) -> float:

	"""
	FONCTION : ``compute_score_perf``
	--------

	Permet de calculer un ``score de performance`` entre deux listes de floats.

	Paramètres
	----------

	* ``y`` : List[Float]

		Liste contenant les Valeurs réelles.

	* ``y_pred`` : List[Float]

		Liste contenant les Valeurs prédites.

	Return
	------

	* ``Float`` : Float

		Retourne un score de performance entre des données réelles et des prédictions.

	Exemple
	-------

	>>> print(compute_score_perf([1, 2, 3], [1.1, 2.2, 3.3]))
	-0.00422559
	"""

	if len(y) == 1:

		# Vecteur de coordonnées 

		vec_coord = [-1]*3
		id = 0

		# Chaque métriques devient une coordonnée du vecteur

		for metric in ["mae", "mde", "rmse"]:

			vec_coord[id] = metric_choice(y, y_pred, scoring=metric)
			id+=1

		vec_coord = tuple(vec_coord)

		# Retouner le score de performance

		return dist_func(vec_coord, (0, 0, 0))*(-1)

	else :

		# Vecteur de coordonnées 

		vec_coord = [-1]*4
		id = 0

		# Chaque métriques devient une coordonnée du vecteur

		for metric in ["mae", "mde", "rmse", "r2"]:

			vec_coord[id] = metric_choice(y, y_pred, scoring=metric)
			id+=1

		vec_coord = tuple(vec_coord)

		# Retouner le score de performance

		return abs_func(1 - dist_func(vec_coord, (0, 0, 0, 0)))*(-1)