#!/bin/python3
#-*- coding : utf-8 -*-

# =============================
# || Importation des modules ||
# =============================

import matplotlib.pyplot as plt
import pandas as pd
import typing as tp
import seaborn as sns
from sklearn.decomposition import PCA

# ==================================
# || FONCTION : plot_NIRS_spectra ||
# ==================================

def plot_NIRS_spectra(df_nirs: pd.DataFrame,				 
					  largeur: int=12,
					  hauteur: int=8,
					  title: str="",
					  x_lab: str="Longueurs d'ondes",
					  y_lab: str="Absorbance",
					  pad: int=15,
					  font_size: int=10,
					  grid: bool=False,
					  save: str=None,
					  xticks_color: str="k") -> None:

	"""
	FONCTION : plot_NIRS_spectra
	--------

	Permet l'obtention d'un visuel des spectres contenus dans un dataset.

	Paramètres
	----------

	* ``df_nirs`` : DataFrame

		Dataframe que l'on souhaite plotter.

	* ``largeur`` : Int, default=12

		Largeur de la figure.

	* ``hauteur`` : Int, default=8

		Hauteur de la figure.

	* ``title`` : String, defautl=""

		Titre du graphique.

	* ``x_lab`` : String, default="Longueurs d'ondes"

		Titre pour les abscisses.

	* ``y_lab`` : String, default="Absorbance"

		Titre pour les ordonnées.

	* ``pad`` : Int, default=15

		Contrôle l'espacement entre les titres et les axes.

	* ``font_size`` : Int, default=10

		Contrôle la taille des titres.

	* ``grid`` : Boolean, default=False

		Permet l'affichage d'une grille.

	* ``xticks_color`` : String, default="k"

		Permet la choisir la colorations des points de repère sur les abscisses.
		
	Return
	------

	* ``None``

		Affiche une figure contenant des spectres d'absorptions.

	Exemple
	-------

	>>> plot_NIRS_spectra(
			df_NIRS=features,
			title="Absorption en fonction des Longueur d'onde",
			largeur=12,
			hauteur=8,
			x_lab="Longueur d'onde",
			y_lab="Absorption",
			grid=False,
			xticks_color="black"):
	>>> ...
	<Figure size 864x576 with 1 Axes>
	"""

	# Transformation en objet numpy

	df_nirs = df_nirs.to_numpy()

	# Création d'un objet figure

	plt.figure(figsize=(largeur, hauteur))
	plt.title(title, pad=pad, fontsize=font_size+2.5)

	for i in range(len(df_nirs)):

		# Plotter les échantillons à la suite des autres

		plt.plot(df_nirs[i, :])

	# Finition de la figure

	plt.xlabel(x_lab, labelpad=pad, fontsize=font_size)
	plt.ylabel(y_lab, labelpad=pad, fontsize=font_size)
	plt.xticks(color=xticks_color)

	if grid:
		plt.grid()

	if save is not None:
		plt.savefig(save)

	plt.show()

# ======================================
# || FONCTION : plot_box_boxen_violin ||
# ======================================

def plot_box_boxen_violin(x: pd.DataFrame,
						  y: pd.DataFrame,
						  type_graph: str="box",
						  title: str="",
						  largeur: int=12,
						  hauteur: int=8,
						  grid: bool=False,
						  save: str=None) -> None:

	"""
	FONCTION : plot_box_boxen_violin
	--------

	Permet l'obtention de ``box``, ``boxen``, ``violin`` plot, à partir de données x et y.

	Paramètres
	----------

	* ``x`` : DataFrame
	
		DataFrame contenant les abscisses. 
		
	* ``y`` : DataFrame

		DataFrame contenant les ordonnées. 

	* ``type_graph`` : String, default="box"

		Type de graphique à plotter

		liste des graphiques disponibles :

			["`box`", "`boxen`", "`violin`"]

	* ``title`` : String, default=""

		Titre de la figure.

	* ``largeur`` : Int, default=12

		Largeur de la figure.

	* ``hauteur`` : Int, default=8

		Hauteur de la figure.

	* ``grid`` : Boolean, default="False"

		Permet l'affichage d'une grille.

	* ``save`` : Str, default=None

		Nom de la figure pour la sauvegarde. Ne sauvegarde la figure que si la valeur est différent de ``None``.

	Return
	------

	* ``None``

		Affiche, au choix, un box, boxen, violin plot illustrant la répartition des ordonnées en fonction des abscisses. 

	Exemple
	-------

	>>> plot_box_boxen_violin(
				x=df["x"],
				y=df["y"],
				type_graph="box",
				title="Boxplot de la répartition des scores en fonction de la techniques employée",
				largeur=14,
				hauteur=12,
				grid=False,
				save="test.png"):
	>>> ...
	<Figure size 1008x864 with 1 Axes>
	"""

	# Transformation en objet numpy

	y = y.to_numpy()
	x = x.to_numpy()

	# Création de la figure

	plt.figure(figsize=(largeur, hauteur))

	plt.title(title)

	# Choix du type de graph

	if type_graph == "box":
		sns.boxplot(y=y, x=x)

	elif type_graph == "boxen":
		sns.boxenplot(y=y, x=x)

	elif type_graph == "violin":
		sns.violinplot(y=y, x=x)

	else: 
		raise ValueError("Argument 'type_graph' must be 'box', 'boxen', 'violin'.")

	if grid:
		plt.grid()

	if save is not None:
		plt.savefig(save)
		
	plt.show()

# ======================================
# || FONCTION : plot_train_test_split ||
# ======================================

def plot_train_test_split(features: pd.DataFrame,
						  x_train: pd.DataFrame,
						  x_test: pd.DataFrame,
						  largeur: int=12,
						  hauteur: int=8,
						  title: str="",
						  train_color: tp.Tuple[str, str]=("c", "k"),
						  test_color: tp.Tuple[str, str]=("r", "k"),
						  grid: bool=False,
						  save: str=None) -> None:

	"""
	FONCTION : plot_train_test_split
	--------

	Permet l'obtention d'un visuel des spectres contenus dans un dataset.

	Paramètres
	----------

	* ``features`` : DataFrame

		Dataframe contenant les features.

	* ``x_train`` : DataFrame

		Dataframe contenant le train_set.

	* ``x_test`` : DataFrame

		Dataframe contenant le test_set.

	* ``largeur`` : Int, default=12

		Largeur de la figure.

	* ``hauteur`` : Int, default=8

		Hauteur de la figure.

	* ``title`` : String, defautl=""

		Titre du graphique.

	* ``train_color`` : Tuple(Str, Str), default=("c", "k")

		Couleur pour les échantillons du train_set.
		 Le second élément du tuple est la couleur de la bordure du point.

	* ``test_color`` : Tuple(Str, Str), default=("r", "k")

		Couleur pour les échantillons du test_set.
		 Le second élément du tuple est la couleur de la bordure du point.

	* ``grid`` : Boolean, default=False

		Permet l'affichage d'un grille.

	* ``save`` : Str, default=None

		Nom de la figure pour la sauvegarde. Ne sauvegarde la figure que si la valeur est différent de ``None``.

	Return
	------

	* ``None``

		Affiche une figure illustrant la répartitions entre les échantillons du train_set et du test_set, 
		 après une ``PCA`` si nécéssaire.

	Exemple
	-------

	>>> plot_train_test_split(
				features=features,
				x_train=x_train,
				x_test=x_test,
				largeur=16,
				hauteur=14,
				title="",
				train_color=("c", "k"),
				test_color=("r", "k"),
				grid=False,
				save="test.png"):
	>>> ...
	<Figure size 1152x1008 with 6 Axes>
	"""

	# Vérification du nombre de variables + transformation en objet numpy

	if 3 < features.shape[1] :

		pca = PCA(n_components=3).fit(features)

		train = pca.transform(x_train)
		test = pca.transform(x_test)

	else:

		train = x_train.to_numpy()
		test = x_test.to_numpy()

	features_shape = test.shape[1]

	# Création d'une figure

	plt.figure(figsize=(largeur, hauteur))
	plt.suptitle(title)

	sub_plot = 0

	for i in range(features_shape):

		for j in range(features_shape):

			if i != j:

				sub_plot+=1
				plt.subplot(3, 2, sub_plot)
				plt.xlabel("Feature{}".format(j+1), labelpad=5)
				plt.ylabel("Feature{}".format(i+1), labelpad=5)
				plt.scatter(train[:, j], train[:, i], color=train_color[0], edgecolors=train_color[1])
				plt.scatter(test[:, j], test[:, i], color=test_color[0], edgecolors=test_color[1])

	plt.subplots_adjust(wspace=0.3, hspace=0.5)
	
	if grid == True:
		plt.grid()

	if save is not None:
		plt.savefig(save)

	plt.show()

# ============================
# || FONCTION : plot_result ||
# ============================

def plot_result(labels: pd.DataFrame,
				x_train: pd.DataFrame,
				x_test: pd.DataFrame,
				y_train: pd.DataFrame,
				y_test: pd.DataFrame,
				estimator: object,
				largeur: int=12,
				hauteur: int=8,
				line_color: str="k",
				train_color: tp.Tuple[str, str]=("c", "k"),
				test_color: tp.Tuple[str, str]=("r", "k"),
				save: str=None) -> None:
	
	"""
	FONCTION : plot_result
	--------

	Permet l'obtention d'un visuel des résultats d'entraînement d'un modèle de machine learning.

	Paramètres
	----------

	* ``labels`` : DataFrame

		Dataframe contenant l'ensemble des labels(train_set + test_set). Paramètre nécessaire pour le traçage de la droite de régression.

	* ``x_train`` : DataFrame

		Dataframe des features du train_set.

	* ``x_test`` : DataFrame

		Dataframe des features du test_set.

	* ``y_train`` : DataFrame

		Dataframe des labels du train_set.

	* ``y_test`` : DataFrame

		Dataframe des labels du test_set.

	* ``estimator`` : Object

		Le modèle entrainé.

	* ``largeur`` : Int, default=12

		Largeur de la figure.

	* ``hauteur`` : Int, default=8

		Hauteur de la figure.

	* ``line_color`` : String, default="k"

		Couleur de la droite d'évaluation.

	* ``train_color`` : Tuple(Str, Str), default=("c", "k")

		Couleur des points du train_set.
		 Le second élément du tuple est la couleur de la bordure du point.

	* ``test_color`` : Tuple(Str, Str), default=("r", "k")

		Couleur des points du test_set.
		 Le second élément du tuple est la couleur de la bordure du point.

	* ``save`` : Str, default=None

		Nom de la figure pour la sauvegarde. Ne sauvegarde la figure que si la valeur est différent de ``None``.

	Return
	------

	* ``None``

		Affiche une figure illustrant les résultats d'entraînement d'un modèle.

	Exemple
	-------

	>>> plot_result(
			labels=labels,
			x_train=x_train,
			x_test=x_test,
			y_train=y_train,
			y_test=y_test,
			estimator=estimator,
			largeur=14,
			hauteur=12,
			line_color="k",
			train_color=("c", "k"),
			test_color=("r", "k")):
	>>> ...
	<Figure size 1008x864 with 4 Axes>
	"""
	# Transformation en objet numpy

	labels = labels.to_numpy()
	y_train = y_train.to_numpy()
	y_test = y_test.to_numpy()

	n_pred_var = labels.shape[1]
	sub_plot = 0

	# Création d'une figure

	plt.figure(figsize=(largeur, hauteur))
	plt.suptitle("Prédictions du modèle vs Valeurs réelles, en fonction des labels")

	# Plottage des résultats de l'entraînement sur le train_set

	for i in range(n_pred_var):

		max_labels = labels[:, i].max()
		inf = labels[:, i].min()-(0.2*max_labels)
		sup = max_labels+(0.2*max_labels)

		sub_plot+=1
		plt.subplot(2, n_pred_var, sub_plot)

		plt.title("Train : Prédictions {} vs Valeurs réelles {}".format(i+1, i+1))

		plt.grid()

		plt.xlabel("Prédictions {}".format(i+1))
		plt.ylabel("Valeurs réelles {}".format(i+1))
		plt.ylim(inf, sup)
		plt.xlim(inf, sup)

		plt.plot(labels[:, i], labels[:, i], color=line_color)

		pred = estimator.predict(x_train).reshape(-1, 1)

		plt.scatter(pred[:, i], y_train[:, i], color=train_color[0], edgecolors=train_color[1])

	# Plottage des résultats sur le test_set

	for i in range(n_pred_var):

		max_labels = labels[:, i].max()
		inf = labels[:, i].min()-(0.2*max_labels)
		sup = max_labels+(0.2*max_labels)

		sub_plot+=1
		plt.subplot(2, n_pred_var, sub_plot)

		plt.title("Test : Prédictions {} vs Valeurs réelles {}".format(i+1, i+1))

		plt.grid()

		plt.xlabel("Prédictions {}".format(i+1))
		plt.ylabel("Valeurs réelles {}".format(i+1))
		plt.xlim(inf, sup)
		plt.ylim(inf, sup)

		plt.plot(labels[:, i], labels[:, i], color=line_color)

		pred = estimator.predict(x_test).reshape(-1, 1)
		plt.scatter(pred[:, i], y_test[:, i], color=test_color[0], edgecolors=test_color[1])

	plt.subplots_adjust(wspace=0.3, hspace=0.3)

	if save is not None:
		plt.savefig(save)

	plt.show()