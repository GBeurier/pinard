# pynirs

pynirs is a python package developed at CIRAD (www.cirad.fr) by Grégory Beurier (beurier@cirad.fr), Denis Cornet (denis.cornet@cirad.fr) and Lauriane Rouan (lauriane.rouan@cirad.fr) to enhance Near Infrared Spectroscopy files processing.

It provides set of tools to load, filter, noise and manage sets of NIRS spectrum files. 

## INSTALLATION
tensorflow 2.6.2
cuda 11.2
python 3.7.9


pynirs is available with pip:

pip install pynirs

## USAGE

TODO

## TODO AND EXTENSIONS

### globally
- finish package structure / declarations
- enhance tests

### preprocessing
- thread preprocessing - Done 
    - multiprocessing with spawns to test
- compute distance beetween preprocessings. Keep only significative ones (mean / variance) - Done
- use previous result to design for the tree of preprocessing - Done
- validate scaling / normalization pattern (before / within / after preprocessing) - Done
- scaleY - Done
- save (xy_scalers and y_inverse_scalers) - Done

### sets
- save / load
- compressed representation to minimize memory print

### augmentation
- noises and data structures

### learning
- all