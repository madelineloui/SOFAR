# SOFAR: Satellite Onboard Fault Attribution and Response

Data-driven ML fault detection and attribution and satellite telemetry simulation tool (SatFaultSim)

Authors:
* Madeline Loui Anderson, MIT
* Kerri Cahoy, MIT
* Jeremy Muesing, Auria Space
* Kenneth Center, Auria Space

### Create virtual environment

Tested using Python 3.10.11, however >3.8 should work. 

Create virtual environment, for example using `python3.10 -m venv env`

Install dependencies with `pip install -r requirements.txt`, which will install the following dependencies:
* torch
* sklearn
* numpy
* pandas
* seaborn
* xgboost
* matplotlib
* tqdm
* jupyter
* tigramite (see below)

#### Installing Tigramite

Download files and follow instructions from https://github.com/jakobrunge/tigramite to install Tigramite in the base directory (needed for PCMCI).

`cd tigramite`

`python setup.py install`

This will install tigramite in your path.

To use just the ParCorr, CMIknn, and CMIsymb independence tests, only numpy/numba and scipy are required. For other independence tests more packages are required:

- GPDC: scikit-learn is required for Gaussian Process regression and dcor for distance correlation

- GPDCtorch: gpytorch is required for Gaussian Process regression

Note: Due to incompatibility issues between numba and numpy, we currently enforce soft dependencies on the versions.

### Usage

A. Data Simulation Tool
* Find data simulation code and example config (yaml) files for normal and faulting data in the `/data` directory
* Reference desired configuration file in `/data/simulated_data.py` and run the python file to generate data in csv format
* Use `-g` to plot the resulted simulated data and use `-f` to force faults

B. Representation Learners
* Code for representation learners are found in the `/algorithms` directory, which includes an autoencoder in `01_ae.iypnb`, PCMCI in `01_pcmci.ipynb`, Kalman Filter in `03_kf.ipynb`, Gaussian Mixture Model in `04_gmm.ipynb`, and LSTM in `05_lstm.ipynb`
* Follow directions in each notebook to pretrain each representation learning using normal (non-faulting data)
* Example pretrained models found in `/models/ensemble_models/`

C. Classifier
* Full inference pipeline and XGBoost classifier model found in `algorithms/ensemble_xgboost.ipynb`
* Example representation outputs using faulting data found in `/models/ensemble_outputs`
* Example trained XGboost model found in `models/xgboost`
* Evaluation found in `algorithms/ensemble_xgboost.ipynb`
* XGBoost visualization found in algorithms/xgboost_viz.py

### Citation

Code implementation of the following paper, published at SmallSat 2024:
```bibtex
@article{anderson2024ensemble,
  title={Ensemble Learning for Autonomous Onboard Satellite Fault Diagnosis With Validation Tool},
  author={Anderson, Madeline and Muesing, Jeremy and Cahoy, Kerri and Center, Kenneth},
  year={2024}
}
