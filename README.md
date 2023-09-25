# SOFAR

### Create virtual environment

Using Python 3.10.11 (>3.8 should work)

Dependencies:
* torch
* sklearn
* numpy
* pandas
* seaborn
* xgboost
* matplotlib
* tqdm
* jupyter
* pickle
* tigramite (see below)

### Installing Tigramite

Download files and follow instructions from https://github.com/jakobrunge/tigramite to install Tigramite in the base directory (needed for PCMCI).

`cd tigramite`

`python setup.py install`

This will install tigramite in your path.

To use just the ParCorr, CMIknn, and CMIsymb independence tests, only numpy/numba and scipy are required. For other independence tests more packages are required:

- GPDC: scikit-learn is required for Gaussian Process regression and dcor for distance correlation

- GPDCtorch: gpytorch is required for Gaussian Process regression

Note: Due to incompatibility issues between numba and numpy, we currently enforce soft dependencies on the versions.
