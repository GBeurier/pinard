---
title: 'Pinard: A Python package for Near Infrared Reflectance Spectroscopy'
tags:
  - Python
  - near infrared reflectance spectroscpy (NIRS)
  - machine learning
  - scikit-learn
  - tensorflow - keras
authors:
  - name: Grégory Beurier
    orcid: 0000-0002-5621-7079
    equal-contrib: true
    affiliation: "1"
  - name: Denis Cornet
    orcid: 0000-0001-9297-2680
    equal-contrib: true
    affiliation: "1"
  - name: Lauriane Rouan
    orcid: 0000-0002-0478-3634
    corresponding: true
    affiliation: "1"
affiliations:
 - name: AGAP, Univ Montpellier, CIRAD, INRA, and Montpellier SupAgro, Montpellier, France
   index: 1
date: 15 June 2023
bibliography: paper.bib

# Summary

NIRS measures the light reflected from a sample after irradiating it with wavelengths 
ranging from visible to shortwave infrared. This provides a signature of the physical 
and chemical characteristics of the sample. Thanks to its low cost NIRS has been widely 
used for determining chemical traits in various fields - pharmaceutical, agricultural, and
food sectors ([@Shepherd2007]; [@Biancolillo2018Chemometric];  [@Pasquini2018Near])
Although NIRS data are simple to acquire, they quickly generate a very large amount of 
information and this information must be processed to allow quality predictions for desired 
traits.
Pinard provides a set of python functionalities dedicated to the preprocessing and processing 
of NIRS data and allows the fast development of prediction models thanks to the extension of 
scikit-learn pipelines. Pinard has been used successfully in a number of scientific projects
([@Vasseur2022Perspective], [@Przybylska2023AraDiv])

# Statement of need

While NIRS data are simple to acquire and rapidly generate a very large amount of information,
they also require extensive post-processing, via chemiometric and multivariate statistical 
analyses. Usually, spectral information can be exploited through the development of calibration 
models relating spectra and reference trait data. For that, different statistical methods are 
commonly used to predict trait data from spectra, including partial least squares regression 
(PLSR; [@Wold1983Multivariate]), principal components analysis, and 2D 
correlation plots [@Darvishzadeh2008LAI]. However, (1) the performance of these methods, 
and especially PLSR, has been shown to vary significantly across samples depending on conditions
[@fu2020estimating] and (2) the strong statistical background of the methods used has limited
the strong dependence on statistical methods has limited the integration of methods developed 
outside platforms like R or Matlab.
In recent years, Machine Learning approaches have become widespread in multiple fields due 
to their better predictive performance and has been applied with success in NIRS 
([@Vasseur2022Perspective]; [@Zhang2022Review]; [@Le2020Application]). 
Machine Learning tools and methods are leaning heavily towards Python langage and therefore 
their adoption by the chemometry community is rather complex.
Pinard provides a way to solve this difficulty by providing all the traditional tools of NIRS
analysis (dedicated signal processing, dataset splitting methods, etc.) but compatible with
the [scikit-learn Pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html). Scikit-learn pipeline provides a elegant and efficient 
way to assemble several processing step and are compatibles with the most known Machine Learning
libraries ([scikit-learn](https://scikit-learn.org/stable/), [tensorflow](https://www.tensorflow.org), [pytorch](https://pytorch.org/), etc.).

# Acknowledgements

We acknowledge contributions from the Phenomen team at [CIRAD](https://www.cirad.fr).

# References