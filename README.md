Pinard is a python package that provides functionalities dedicated to the preprocessing and processing of NIRS data and allows the fast development of prediction models thanks to the extension of scikit-learn pipelines.

NIRS measures the light reflected from a sample after irradiating it with wavelengths ranging from visible to shortwave infrared. This provides a signature of the physical
and chemical characteristics of the sample. Thanks to its low cost NIRS has been widely used for determining chemical traits in various fields - pharmaceutical, agricultural, and food sectors (Shepherd and Walsh, 2007; Wójcicki, 2015; Biancolillo and Marini, 2018; Pasquini, 2018)
Although NIRS data are simple to acquire, they quickly generate a very large amount of information and this information must be processed to allow quality predictions for desired traits.
Pinard provides a set of python functionalities dedicated to the preprocessing and processing of NIRS data and allows the fast development of prediction models thanks to the extension of scikit-learn pipelines:

- Collection of spectra preprocessings: Baseline, StandardNormalVariate, RobustNormalVariate, SavitzkyGolay, Normalize, Detrend, MultiplicativeScatterCorrection, Derivate, Gaussian, Haar, Wavelet...,
- Collection of splitting methods based of spectra similarity metrics: Kennard Stone, SPXY, random sampling, stratified sampling, k-mean...,
- An extension of sklearn pipelines to provide 2D tensors to keras regressors.

Moreover, because Pinard extends scikit-learn, all scikit-learn features are natively available (split, regressor, etc.).

## Authors

Pinard is a python package developed at CIRAD (www.cirad.fr) by Grégory Beurier (beurier@cirad.fr) in collaboration with Denis Cornet (denis.cornet@cirad.fr) and Lauriane Rouan (lauriane.rouan@cirad.fr)

## Dependencies

https://github.com/avkl/twinning twinning
sklearn
numpy
pandas
scipy

## INSTALLATION

pinard is available with pip:

pip install pinard

## USAGE

see examples folder

## ROADMAP

- sklearn compatibility:
    - Extend sklearn pipeline to fully integrate data augmentation (x,y along the pipeline management)
    - Extend sklearn pipeline to integrate  validation data (required for Deep Learning tuning)
    - Add folds and iterable results to all splitting methods (cross validation / KFold compatibility)
- ease of use:
    - Extend model_selection helpers (metrics, methods, etc.)
    - Provide dedicated serialization methods to avoid compatibility problems between different frameworks (i.e. Keras + sklearn)
- data augmentation:
    - Auto-balance sample augmentation based on groups/classes/metric - augmentation count replaced by ratio/weight
    - Allow augmentation methods parameters