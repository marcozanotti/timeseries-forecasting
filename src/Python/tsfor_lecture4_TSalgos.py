# Time Series Forecasting: Machine Learning and Deep Learning with R & Python ----

# Lecture 4: Time Series Algorithms ---------------------------------------
# Marco Zanotti

# Goals:
# - Baseline Models
# - SARIMAX
# - ETS
# - TBATS
# - STLM
# - PROPHET



# Packages ----------------------------------------------------------------

import os
import pickle

import numpy as np
import pandas as pd
# import polars as pl
#import pytimetk as tk

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

# with Nixtla's workflow there is no need to split data before 
# since validation is performed directly through the cross_validation method
# however it is always useful to visualize the validation plan 
# for that one can just cross-validate a naive model to obtain the 
# validation plan (cutoffs dates)
pex.plot_cross_validation_plan(
    data_prep_df, freq = 'D', h = horizon, 
    n_windows = 1, step_size = 1, 
    engine = 'matplotlib'
)

pex.plot_cross_validation_plan(
    data_prep_df, freq = 'D', h = horizon, 
    n_windows = 6, step_size = 14, 
    engine = 'matplotlib'
)



# S-NAIVE & AVERAGES ------------------------------------------------------

# Naive
# Seasonal Naive
# Averages
from statsforecast.models import (
    HistoricAverage,
    Naive, 
    SeasonalNaive,
    RandomWalkWithDrift,
    WindowAverage,
    SeasonalWindowAverage
)


# * Engines ---------------------------------------------------------------

models_baseline = [
    HistoricAverage(),
    Naive(),
    SeasonalNaive(season_length = 7),
    RandomWalkWithDrift(),
    WindowAverage(window_size = 7), 
    SeasonalWindowAverage(season_length = 7, window_size = 8)
]

# Instantiate StatsForecast class
sf_baseline = StatsForecast(
    models = models_baseline,
    freq = 'D', 
    n_jobs = -1,
)


# * Evaluation ------------------------------------------------------------

cv_res_baseline = pex.calibrate_evaluate_plot(
    sf_baseline, data_prep_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 2  
)

cv_res_baseline['cv_results']
cv_res_baseline['accuracy_table']
cv_res_baseline['plot'].show()


# * Refitting & Forecasting -----------------------------------------------

# if you need model parameters use .fit and .predict 
# otherwise use .forecast (optimized to run also on clusters)
# P.S. specify fitted = True to store the fitted values

# fit_res_baseline = sf_baseline.fit(
#     df = data_prep_df, prediction_intervals = intervals
# )
# preds_df_baseline = fit_res_baseline.predict(
#     h = horizon, level = levels
# )
fcst_df_baseline = sf_baseline.forecast(
    df = data_prep_df, h = horizon,
    prediction_intervals = intervals, 
    level = levels
)
fcst_df_baseline

sf_baseline.plot(
    data_prep_df, fcst_df_baseline,
    max_insample_length = horizon * 2,
    engine = 'plotly'
).show()

for nm in sf_baseline.models:
    sf_baseline.plot(
        data_prep_df, 
        fcst_df_baseline,
        models = [str(nm)], 
        level = levels,
        max_insample_length = horizon * 2,
        engine = 'plotly'
    ).show()



# S-ARIMA-X ---------------------------------------------------------------

# Seasonal Regression with ARIMA Errors and External Regressors
# yt = alpha * L(yt)^k +  beta L(yt)^s + et + gamma * L(et)^k + delta * L(xt)^k

# ARIMA is a simple algorithm that relies on Linear Regression
# Strengths:
# - Automated Differencing
# - Automated Parameter Search (auto_arima)
# - Single seasonality modeling included
# - Recursive Lag Forecasting
# Weaknesses:
# - Only single seasonality by default (XREGs can help go beyond single seasonality)
# - Becomes erratic with too many lags
# - Requires Expensive Parameter Search

from statsforecast.models import (
    ARIMA,
    AutoRegressive, 
    AutoARIMA
)


# * Engines ---------------------------------------------------------------

models_arima = [
    ARIMA(order = (1, 1, 1), season_length = 7, seasonal_order = (1, 1, 1)),
    AutoRegressive(lags = [1, 7, 14, 30]),
    AutoARIMA(season_length = 7)
]
sf_arima = StatsForecast(
    models = models_arima,
    freq = 'D', 
    n_jobs = -1,
)


# * Evaluation ------------------------------------------------------------

y_df = select_columns(data_prep_df)
y_xregs_arima_df = select_columns(data_prep_df, '(event)')

# ARIMA models without external regressors
cv_res_arima = pex.calibrate_evaluate_plot(
    class_object = sf_arima, 
    data = y_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_arima['cv_results']
cv_res_arima['accuracy_table']
cv_res_arima['plot'].show()

# ARIMA models with external regressors
cv_res_arima_xregs = pex.calibrate_evaluate_plot(
    class_object = sf_arima, 
    data = y_xregs_arima_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_arima_xregs['cv_results']
cv_res_arima_xregs['accuracy_table']
cv_res_arima_xregs['plot'].show()



# XX ------------------------------------------------

from statsforecast.models import (
    
)


# * Engines ---------------------------------------------------------------

models_xx = [
    
]
sf_xx = StatsForecast(
    models = models_xx,
    freq = 'D', 
    n_jobs = -1,
)


# * Evaluation ------------------------------------------------------------

cv_res_xx = pex.calibrate_evaluate_plot(
    class_object = sf_xx, data = data_prep_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_xx['cv_results']
cv_res_xx['accuracy_table']
cv_res_xx['plot'].show()



# XX ------------------------------------------------

from statsforecast.models import (
    
)


# * Engines ---------------------------------------------------------------

models_xx = [
    
]
sf_xx = StatsForecast(
    models = models_xx,
    freq = 'D', 
    n_jobs = -1,
)


# * Evaluation ------------------------------------------------------------

cv_res_xx = pex.calibrate_evaluate_plot(
    class_object = sf_xx, data = data_prep_df, 
    h = horizon, prediction_intervals = intervals, level = levels,
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_xx['cv_results']
cv_res_xx['accuracy_table']
cv_res_xx['plot'].show()
