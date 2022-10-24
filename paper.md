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
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: "1"
  - name: Lauriane Rouan
    orcid: 0000-0002-0478-3634
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: "1"
affiliations:
 - name: AGAP, Univ Montpellier, CIRAD, INRA, and Montpellier SupAgro, Montpellier, France
   index: 1
date: October 2022
bibliography: paper.bib

# Summary

NIRS measures the light reflected from a sample after irradiating it with wavelengths 
ranging from visible to shortwave infrared. This provides a signature of the physical 
and chemical characteristics of the sample. Thanks to its low cost NIRS has been widely 
used for determining chemical traits in various fields - pharmaceutical, agricultural, and
food sectors (Shepherd and Walsh, 2007; Wójcicki, 2015; Biancolillo and Marini, 2018; 
Pasquini, 2018)
Although NIRS data are simple to acquire, they quickly generate a very large amount of 
information and this information must be processed to allow quality predictions for desired 
traits.
Pinard provides a set of python functionalities dedicated to the preprocessing and processing 
of NIRS data and allows the fast development of prediction models thanks to the extension of 
scikit-learn pipelines.

# Statement of need

While NIRS data are simple to acquire and rapidly generate a very large amount of information,
they also require extensive post-processing, via chemiometric and multivariate statistical 
analyses. Usually, spectral information can be exploited through the development of calibration 
models relating spectra and reference trait data. For that, different statistical methods are 
commonly used to predict trait data from spectra, including partial least squares regression 
(PLSR; (Wold et al., 1983)), principal components analysis (Dreccer et al., 2014), and 2D 
correlation plots (Darvishzadeh et al., 2008). However, (1) the performance of these methods, 
and especially PLSR, has been shown to vary significantly across samples depending on conditions
(Fu et al., 2020) and (2) the strong statistical background of the methods used has limited
the strong dependence on statistical methods has limited the integration of methods developed 
outside platforms like R or Matlab.
In recent years, Machine Learning approaches have become widespread in multiple fields due 
to their better predictive performance and has been applied with success in NIRS (Vasseur et al.
, 2021). Machine Learning tools and methods are leaning heavily towards Python langage and 
therefore their adoption by the chemometry community is rather complex.
Pinard provides a way to solve this difficulty by providing all the traditional tools of NIRS
analysis (dedicated signal processing, dataset splitting methods, etc.) but compatible with
the scikit-learn pipelines (REF). Scikit-learn pipeline provides a elegant and efficient way 
to assemble several processing step and are compatibles with the most known Machine Learning
libraries (scikit-learn, tensorflow, pytorch, etc.).

# Citations

# Figures

# Acknowledgements

# References