# Time Series Forecasting: Machine Learning and Deep Learning with R & Python ----

# Lecture 5: Machine Learning Algorithms ----------------------------------
# Marco Zanotti

# Goals:
# - Linear Regression
# - Elastic Net
# - MARS
# - SVM
# - KNN
# - Random Forest
# - XGBoost, Light GBM, CAT Boost
# - Cubist
# - Neural Networks

# Challenges:
# - Challenge 2 - Testing New Forecasting Algorithms



# Packages ----------------------------------------------------------------

import os
import pickle

import numpy as np
import pandas as pd
import random
# import polars as pl
# import pytimetk as tk

from mlforecast import MLForecast
from statsforecast import StatsForecast
from statsforecast.utils import ConformalIntervals

from utilsforecast.evaluation import evaluate
from utilsforecast.losses import bias, mae, mape, mse, rmse

import python_extensions as pex

pd.set_option("display.max_rows", 3)
os.environ['NIXTLA_ID_AS_COL'] = '1'



# Data & Artifacts --------------------------------------------------------

with open('artifacts/Python/feature_engineering_artifacts_list.pkl', 'rb') as f:
    data_loaded = pickle.load(f)
data_prep_df = data_loaded['data_prep_df']
forecast_df = data_loaded['forecast_df']
feature_sets = data_loaded['feature_sets']
params = data_loaded['transform_params']

StatsForecast.plot(data_prep_df).show()

# * Forecast Horizon ------------------------------------------------------
horizon = 7 * 8 # 8 weeks

# * Prediction Intervals --------------------------------------------------

levels = [80, 95]
# Conformal intervals
intervals = ConformalIntervals(h = horizon, n_windows = 2)
# P.S. n_windows*h should be less than the count of data elements in your time series sequence.
# P.S. Also value of n_windows should be atleast 2 or more.

# * Cross-validation Plan -------------------------------------------------

pex.plot_cross_validation_plan(
    data_prep_df, freq = 'D', h = horizon, 
    n_windows = 1, step_size = 1, 
    engine = 'matplotlib'
)

# * External Regressors ---------------------------------------------------

y_df = pex.select_columns(data_prep_df)
y_xregs_df = data_prep_df



# Linear Regression -------------------------------------------------------

# - Baseline model for ML
from sklearn.linear_model import LinearRegression

# * Engines ---------------------------------------------------------------

models_lr = [
    LinearRegression()
]
mlf_lr = MLForecast(
    models = models_lr,
    freq = 'D', 
    num_threads = 1
)

# * Evaluation ------------------------------------------------------------

cv_res_lr = pex.calibrate_evaluate_plot(
    mlf_lr, data = data_prep_df.dropna(), 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_lr['cv_results']
cv_res_lr['accuracy_table']
cv_res_lr['plot'].show()

