![alt text](https://github.com/gbeurier/pinard/blob/main/docs/pinard_logo.jpg?raw=true)
[![Documentation Status](https://readthedocs.org/projects/pinard/badge/?version=latest)](https://pinard.readthedocs.io/en/latest/?badge=latest)

Pinard is a python package that provides functionalities dedicated to the preprocessing and processing of NIRS data and allows the fast development of prediction models thanks to the extension of scikit-learn pipelines.

NIRS measures the light reflected from a sample after irradiating it with wavelengths ranging from visible to shortwave infrared. This provides a signature of the physical
and chemical characteristics of the sample. Thanks to its low cost NIRS has been widely used for determining chemical traits in various fields - pharmaceutical, agricultural, and food sectors (Shepherd and Walsh, 2007; Wójcicki, 2015; Biancolillo and Marini, 2018; Pasquini, 2018)
Although NIRS data are simple to acquire, they quickly generate a very large amount of information and this information must be processed to allow quality predictions for desired traits.
Pinard provides a set of python functionalities dedicated to the preprocessing and processing of NIRS data and allows the fast development of prediction models thanks to the extension of scikit-learn pipelines:

- Collection of spectra preprocessings: Baseline, StandardNormalVariate, RobustNormalVariate, SavitzkyGolay, Normalize, Detrend, MultiplicativeScatterCorrection, Derivate, Gaussian, Haar, Wavelet...,
- Collection of splitting methods based of spectra similarity metrics: Kennard Stone, SPXY, random sampling, stratified sampling, k-mean...,
- An extension of sklearn pipelines to provide 2D tensors to keras regressors.

Moreover, because Pinard extends scikit-learn, all scikit-learn features are natively available (split, regressor, etc.).

![](docs/pipeline.jpg)
*Illustrated End-to-End NIRS Analysis Pipeline using Pinard, Scikit-learn and Tensorflow: Data Input, Augmentation, Preprocessing, Training, Prediction, and Interpretation*


## Authors

Pinard is a python package developed at AGAP institute (https://umr-agap.cirad.fr/) by Grégory Beurier (beurier@cirad.fr) in collaboration with Denis Cornet (denis.cornet@cirad.fr) and Lauriane Rouan (lauriane.rouan@cirad.fr)


## INSTALLATION

Installing the pinard package can be done using the Python package installer pip. To install pinard, you need to open a terminal or command prompt and enter the following command:

pip install pinard

It is recommended to use virtual environments to avoid any conflicts with other packages. Once the installation is complete, you should be able to import the package and use its functions in your Python code.
It's also possible to use the package from Jupyter notebook or any other IDE, you just have to make sure the environment is the same as the one where you installed the package.

If you want to install from the source code, you can use the following command, after having downloaded or cloned the source code from the official repository (https://github.com/GBeurier/pinard):

pip install -r requirements.txt 

This command will install all the dependencies of the package, such as joblib, kennard-stone, numpy, pandas, pytest, PyWavelets, setuptools, scikit-learn, and scipy.

Then it's possible to install the package by running :

python setup.py install


## USAGE

### Basic usage
```python
x, y = utils.load_csv(xcal_csv, ycal_csv, x_hdr=0, y_hdr=0, remove_na=True) # Load data from CSV files, remove rows with missing values
train_index, test_index = train_test_split_idx(x, y=y, method="kennard_stone", metric="correlation" test_size=0.25, random_state=rd_seed) # Split data into training and test sets using the kennard_stone method and correlation metric, 25% of data is used for testing
X_train, y_train, X_test, y_test = x[train_index], y[train_index], x[test_index], y[test_index] # Assign data to training and test sets

# Declare preprocessing pipeline
preprocessing = [   ('id', pp.IdentityTransformer()), # Identity transformer, no change to the data
                    ('savgol', pp.SavitzkyGolay()), # Savitzky-Golay smoothing filter
                    ('derivate', pp.Derivate()), # Calculate the first derivative of the data
                    Pipeline([('_sg1',pp.SavitzkyGolay()),('_sg2',pp.SavitzkyGolay())]))] # nested pipeline to perform the Savitzky-Golay method twice for 2nd order preprocessing

# Declare complete pipeline
pipeline = Pipeline([
    ('scaler', MinMaxScaler()), # scaling the data
    ('preprocessing', FeatureUnion(preprocessing)), # preprocessing
    ('PLS',  sklearn.PLS()) # regressor
])

# Estimator including y values scaling
estimator = TransformedTargetRegressor(regressor = pipeline, transformer = MinMaxScaler())

# Training
estimator.fit(X_train, y_train)

# Predictions
Y_preds = estimator.predict(X_test) # make predictions on test data and assign to Y_preds variable

```

This code is a sample of how to use the pinard package to perform a chemical analysis regression. The package is designed to perform preprocessing and modeling of spectral data.

The first line loads the data from two CSV files, xcal_csv and ycal_csv, using the utils.load_csv function. The function takes several optional parameters, including x_hdr and y_hdr which specify the row number of the header for each file. The remove_na parameter is set to True, which means that any rows with missing values will be removed from the data.

The next line uses the train_test_split_idx function to split the data into training and test sets. The method parameter is set to "kennard_stone" and the metric parameter is set to "correlation". The test_size parameter is set to 0.25, which means that 25% of the data will be used for testing and 75% will be used for training. The random_state parameter is set to rd_seed, which ensures that the same split will be used each time the code is run.

The following lines declare a preprocessing pipeline using the Pipeline class. The pipeline consists of several steps that are performed in parrallel (see FeatureUnion after). The first transformer is the IdentityTransformer which simply returns the input data without any modification. The second transformer is the SavitzkyGolay method, which performs a Savitzky-Golay smoothing filter on the data. The third transformer is the Derivate method, which calculates the first derivative of the data. The last step is a nested pipeline that performs the SavitzkyGolay method twice, it is here for a 2nd order preprocessing.

The next block of code declares a complete pipeline that includes scaling the data using the MinMaxScaler method, performing the preprocessing steps defined above and concatening them (FeatureUnion), and then fitting a partial least squares (PLS) model to the data using the sklearn.PLS module.

After that, the code creates an estimator using the TransformedTargetRegressor class, which applies the specified transformer to the target variable (y) before passing it to the regressor (pipeline).

The code then fits the estimator to the training data using the fit method.

Finally, the code makes predictions on the test data using the predict method and assigns the output to the Y_preds variable.

It is worth noting that this code is just a sample of how the pinard package can be used for a specific type of analysis. The specific preprocessing steps, model, and parameters used may need to be adjusted depending on the specific data and analysis goals.

More complete examples can be found in examples folders and executed on google collab:
- https://colab.research.google.com/github/GBeurier/pinard/blob/main/examples/simple_pipelines.ipynb
- https://colab.research.google.com/github/GBeurier/pinard/blob/main/examples/stacking.ipynb


## ROADMAP

- Sklearn compatibility:
    - Extend sklearn pipeline to fully integrate data augmentation (x,y along the pipeline management)
    - Extend sklearn pipeline to integrate  validation data (required for Deep Learning tuning)
    - Add folds and iterable results to all splitting methods (cross validation / KFold compatibility)
- Ease of use:
    - Extend model_selection helpers (metrics, methods, etc.)
    - Provide dedicated serialization methods to avoid compatibility problems between different frameworks (i.e. Keras + sklearn)
- Data augmentation:
    - Auto-balance sample augmentation based on groups/classes/metric - augmentation count replaced by ratio/weight
    - Allow augmentation methods parameters