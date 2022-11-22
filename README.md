![alt text](https://github.com/gbeurier/pinard/blob/main/doc/logo_pinard.jpg?raw=true)

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


## INSTALLATION

pip install pinard

## USAGE

### Basic usage
```python
x, y = utils.load_csv(xcal_csv, ycal_csv, x_hdr=0, y_hdr=0, remove_na=True) # Load data
train_index, test_index = train_test_split_idx(x, y=y, method="kennard_stone", metric="correlation" test_size=0.25, random_state=rd_seed) # Get splitting indices
X_train, y_train, X_test, y_test = x[train_index], y[train_index], x[test_index], y[test_index]

# Declare preprocessing pipeline
preprocessing = [   ('id', pp.IdentityTransformer()),
                    ('savgol', pp.SavitzkyGolay()),
                    ('derivate', pp.Derivate()), 
                    Pipeline([('_sg1',pp.SavitzkyGolay()),('_sg2',pp.SavitzkyGolay())]))] # reification for 2nd order preprocessing

# Declare complete pipeline
pipeline = Pipeline([
    ('scaler', MinMaxScaler()), # scaling
    ('preprocessing', FeatureUnion(preprocessing)), # preprocessing
    ('PLS',  sklearn.PLS()) # regressor
])

# Estimator including y values scaling
estimator = TransformedTargetRegressor(regressor = pipeline, transformer = MinMaxScaler())

# Training
estimator.fit(X_train, y_train)

# Predictions
Y_preds = estimator.predict(X_test)

```

More complete examples can be found in examples folders and executed on google collab:
- https://colab.research.google.com/github/GBeurier/pinard/blob/main/examples/simple_pipelines.ipynb
- https://colab.research.google.com/github/GBeurier/pinard/blob/main/examples/stacking.ipynb

more examples to come soon...

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