.. Pinard documentation master file, created by
   sphinx-quickstart on Mon Jun 26 16:18:34 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Pinard Documentation
====================

.. image:: ../pinard_logo.jpg
   :width: 25%
   :alt: Pinard logo


.. image:: https://img.shields.io/pypi/v/pinard.svg
   :target: https://pypi.python.org/pypi/pinard
   :alt: PyPI version


Pinard is a Python package designed to provide functionalities dedicated to the preprocessing and processing of NIRS (Near Infrared Spectroscopy) data. It offers a convenient way to develop prediction models by extending scikit-learn pipelines.

Features
========

Pinard includes the following features:

1. **Spectrum Preprocessings**: Pinard provides a collection of spectrum preprocessing techniques, including baseline correction, standard normal variate, robust normal variate, Savitzky-Golay filtering, normalization, detrending, multiplicative scatter correction, derivative, Gaussian filtering, Haar wavelet transformation, and more.

2. **Splitting Methods**: Pinard offers various splitting methods based on spectrum similarity metrics. These methods include Kennard Stone, SPXY, random sampling, stratified sampling, k-means, and more.

3. **Extension of scikit-learn Pipelines**: Pinard extends scikit-learn pipelines to support 2D tensors for keras regressors, enabling seamless integration of machine learning models.

4. **Compatibility with scikit-learn**: As Pinard extends scikit-learn, it natively inherits all the features provided by scikit-learn, such as data splitting, regression models, and more.

Installation
============

To install Pinard, you can use pip:

.. code-block:: bash

   pip install pinard

Usage
=====

Once Pinard is installed, you can import it in your Python code:

.. code-block:: python

   import pinard

For detailed usage instructions and examples, please refer to the official Pinard documentation and the provided examples.


Pinard Architecture
-------------------

The Pinard package provides modules to facilitate the preprocessing, processing, and modeling of NIRS (Near Infrared Spectroscopy) data. These modules offer functionalities for data augmentation, splitting and cross-validation, a wide range of preprocessing methods, and seamless integration with scikit-learn pipelines.

'Data Augmentation with "augmentation" Module'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. toctree::
   :maxdepth: 1

   augmentation

The "augmentation" module in Pinard focuses on data augmentation techniques. Data augmentation is crucial for enhancing the diversity and size of the training data, which helps improve the generalization and robustness of machine learning models. Pinard's "augmentation" module offers various methods to generate augmented samples from the existing data, enabling researchers and practitioners to effectively increase the dataset size and improve model performance.


Splitting and Cross-Validation with "model_selection" Module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. _model_selection:
.. toctree::
   :maxdepth: 1

   model_selection

The "model_selection" module in Pinard is dedicated to splitting and cross-validation techniques. It provides functionalities for splitting the NIRS data into training and testing sets based on various strategies such as Kennard Stone, SPXY, random sampling, stratified sampling, and k-means. Additionally, Pinard's "model_selection" module supports cross-validation methods for evaluating model performance, allowing users to perform robust and reliable model assessments.

Preprocessing Methods in "preprocessing" Module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. _preprocessing:
.. toctree::
   :maxdepth: 1

   preprocessing


The "preprocessing" module in Pinard offers a comprehensive collection of preprocessing methods specifically designed for NIRS data. These methods include baseline correction, standard normal variate, robust normal variate, Savitzky-Golay filtering, normalization, detrending, multiplicative scatter correction, derivative computation, Gaussian filtering, Haar wavelet transformation, and more. With the "preprocessing" module, users can efficiently preprocess their NIRS data and remove unwanted variations or noise before further analysis or modeling.

Seamless Integration with scikit-learn Pipelines via "sklearn" Module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. _sklearn:
.. toctree::
   :maxdepth: 1

   sklearn

Pinard seamlessly integrates with scikit-learn pipelines through the "sklearn" module. This module provides wrappers and dedicated code to interact with scikit-learn pipelines, enabling users to incorporate Pinard's preprocessing methods and NIRS-specific functionalities into their machine learning workflows. By leveraging the power of scikit-learn, users can benefit from the extensive range of tools and models available in scikit-learn while leveraging Pinard's specialized features for NIRS data processing and modeling.

Overall, the Pinard package offers a comprehensive suite of tools and modules that cater to the unique requirements of NIRS data analysis and modeling. From data augmentation to preprocessing and seamless integration with scikit-learn, Pinard provides a powerful and user-friendly environment for efficient NIRS data processing and predictive modeling.



Authors
=======

Pinard is developed at CIRAD (Centre de Coopération Internationale en Recherche Agronomique pour le Développement), a French research organization. The package is developed by Grégory Beurier (beurier@cirad.fr) in collaboration with Denis Cornet (denis.cornet@cirad.fr) and Lauriane Rouan (lauriane.rouan@cirad.fr).

For further inquiries or support, please contact the authors directly.

For more information about CIRAD, please visit their website: `www.cirad.fr <https://www.cirad.fr>`_.

Note
====

Pinard is an open-source project. Contributions, bug reports, and feature requests are welcome. You can find the Pinard project on GitHub: `github.com/cirad/pinard <https://github.com/cirad/pinard>`_.



API Overview
=============

.. toctree::
   :maxdepth: 1

   api


For detailed information about the classes, methods, and attributes provided by the Pinard package, please refer to the following API references:

- Pinard Augmentation API: :mod:`pinard.augmentation`
- Pinard Model Selection API: :mod:`pinard.model_selection`
- Pinard Preprocessing API: :mod:`pinard.preprocessing`
- Pinard scikit-learn Integration API: :mod:`pinard.sklearn`


Click on the links above to access the respective API reference documentation, which includes detailed explanations and usage examples for each module and its components.

Please note that the API reference provides in-depth technical information and is intended for users who want to explore the inner workings of the Pinard package. If you are new to Pinard or NIRS data analysis, we recommend starting with the user guide and examples provided in the Pinard documentation to get a better understanding of the package's capabilities and how to utilize them effectively.




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
