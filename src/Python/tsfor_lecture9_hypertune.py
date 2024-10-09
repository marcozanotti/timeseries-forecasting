# Time Series Forecasting: Machine Learning and Deep Learning with R & Python ----

# Lecture 9: Hyperparameter Tuning ----------------------------------------
# Marco Zanotti

# Goals:
# - Sequential / Non-Sequential Models
# - Time Series / V-Fold Cross Validation
# - Grid Search / Random Search / Bayesian Optimization
# - Tuning with  ML and DL models

# Challenges:
# - Challenge 3 - Hyperparameter Tuning



# Packages ----------------------------------------------------------------

import os
import pickle

import numpy as np
import pandas as pd

from mlforecast import MLForecast
from mlforecast.lag_transforms import (
    RollingMean, ExponentiallyWeightedMean, ExpandingMean
)
from mlforecast.utils import PredictionIntervals
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import bias, mae, mape, mse, rmse
from utilsforecast.plotting import plot_series

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

plot_series(data_prep_df).show()

# * Forecast Horizon ------------------------------------------------------
horizon = 7 * 8 # 8 weeks

# * Prediction Intervals --------------------------------------------------

levels = [80, 95]
intervals = PredictionIntervals(h = horizon, n_windows = 2, method = 'conformal_distribution')


# * Train / Test Split ----------------------------------------------------

train_df = data_prep_df \
    .select_columns('(event)|(holiday)|(inter_)|(ds_)') \
    .head(n = -horizon)
test_df = data_prep_df \
    .select_columns('(event)|(holiday)|(inter_)|(ds_)') \
    .tail(n = horizon)
fcst_df = forecast_df \
    .select_columns('(event)|(holiday)|(inter_)|(ds_)')

pex.plot_cross_validation_plan(
    data_prep_df, freq = 'D', h = horizon, 
    n_windows = 1, step_size = 1, 
    engine = 'matplotlib'
)



# Model Types -------------------------------------------------------------

# Sequential Models Tuning
# - Sequential Model Definition:
#   - Creates Lags internally
#   - Predicts next H observations (recursive)
#   - All data must be sequential (ordered by date)
#   - Cannot use V-Fold Cross Validation / Must use Time Series Cross Validation
# - Examples of Sequential Models:
#   - ARIMA
#   - Exponential Smoothing
#   - NNETAR

# Non-Sequential Models Tuning
# - Non-Sequential Model Definition:
#   - Uses date features
#   - Lags Created Externally
#   - Spline can be modeled with random missing observations
#   - Can be Tuned using K-Fold Cross Validation
# - Examples:
#   - Machine Learning Algorithms that use Calendar Features (e.g. GLMNet, XGBoost)
#   - Prophet
# - IMPORTANT: Can use time_series_cv() or vfold_cv(). 
#   Usually better performance with vfold_cv().



# Validation Strategies ---------------------------------------------------

# Time-series cross-validation
pex.plot_cross_validation_plan(
    data_prep_df, freq = 'D', h = horizon, 
    n_windows = 12, step_size = 7, 
    engine = 'matplotlib'
)


# V-fold cross-validation



# Search Types ------------------------------------------------------------

# Grid Search
# Grid search is one of the most straightforward methods for hyperparameter 
# tuning. It systematically works through multiple combinations of parameter 
# options, evaluating each one to determine the best performance. While 
# this method guarantees finding the optimal parameters, it can be 
# computationally expensive and time-consuming, especially with a large 
# number of hyperparameters.

# Advantages of Grid Search
# - Exhaustive Search: Evaluates all possible combinations, ensuring the
#   best model is found.
# - Simplicity: Easy to implement and understand.
# Disadvantages of Grid Search
# - Time-Consuming: Can take a long time to complete, particularly with 
#   a large parameter space.
# - Scalability Issues: As the number of hyperparameters increases, the 
#   time required grows exponentially.

# Random Search
# Random search addresses some of the limitations of grid search by 
# randomly sampling from the hyperparameter space. This method can often 
# yield better results in less time, as it does not evaluate every combination.

# Benefits of Random Search
# - Efficiency: Can find good hyperparameters faster than grid search.
# - Flexibility: Allows for exploration of a wider range of values.

# Bayesian Optimization
# Bayesian optimization is a more sophisticated approach that builds 
# a probabilistic model of the function mapping hyperparameters to a 
# target objective. It uses this model to select the most promising 
# hyperparameters to evaluate next, balancing exploration and exploitation.

# Key Features of Bayesian Optimization
# - Model-Based: Utilizes a surrogate model to predict performance.
# - Adaptive Sampling: Focuses on areas of the hyperparameter space that 
#   are likely to yield better results.



# Tuning - ML Models ------------------------------------------------------

# * Non-Optimized Models --------------------------------------------------
from sklearn.linear_model import ElasticNet
from lightgbm import LGBMRegressor

models_mlf = [
    ElasticNet(l1_ratio = 0.5),
    LGBMRegressor(
        n_estimators = 100,
        learning_rate = 0.1,
        objective = 'rmse',
        random_state = 0
    )  
]

mlf = MLForecast(
    models = models_mlf,
    freq = 'D', 
    num_threads = 1,
    lags = [1, 2, 7, 14, 30],
    lag_transforms = {
        7: [
            RollingMean(window_size = 7),
            ExpandingMean()
        ],
        14: [
            RollingMean(window_size = 14)            
        ],
        30: [
            RollingMean(window_size = 30),
            RollingMean(window_size = 60), 
            RollingMean(window_size = 90),
            ExponentiallyWeightedMean(alpha = 0.3)
        ]
    }
)

mlf_fit = mlf.fit(
    df = train_df, 
    prediction_intervals = intervals,
    static_features = []
)
mlf_preds_df = mlf_fit.predict(
    h = horizon, 
    level = levels, 
    X_df = test_df.drop('y', axis = 1)
)

evaluate(
    df = mlf_preds_df \
        .merge(test_df.select_columns(), on = ['unique_id', 'ds']), 
    metrics = [bias, mae, mape, mse, rmse],
    agg_fn = 'mean'
)
plot_series(
    pd.concat([train_df, test_df]), mlf_preds_df,
    max_insample_length = horizon * 2,
    level = levels,
    engine = 'plotly'
).show()


# * Optimized Models ------------------------------------------------------

import optuna
from lightgbm import LGBMRegressor
from mlforecast.auto import (
    AutoMLForecast,
    AutoModel,
    AutoLightGBM,
    AutoElasticNet
)
optuna.logging.set_verbosity(optuna.logging.ERROR)

# * Default optimization --------------------------------------------------

# We have default search spaces for some models and we can define default 
# features to look for based on the length of the seasonal period of your 
# data. You just need to use the Auto-model couterpart.

# * Tuning model parameters -----------------------------------------------

# Otherwise, you can provide your own model with its search space to perform 
# the optimization. The search space should be a function that takes an 
# optuna trial and returns the model parameters. Then simply create the auto
# version of the desired model using the AutoModel function specifing 
# the model and the configuration.
def lgb_config(trial: optuna.Trial):
    return {
        'learning_rate': 0.05,
        'verbosity': -1,
        'num_leaves': trial.suggest_int('num_leaves', 2, 128, log = True),
        'objective': trial.suggest_categorical('objective', ['l1', 'l2', 'rmse']),
    }
my_lgb = AutoModel(
    model = LGBMRegressor(),
    config = lgb_config,
)

# * Tuning features -------------------------------------------------------

# The MLForecast class defines the features to build in its constructor. 
# You can tune the features by providing a function through the init_config 
# argument, which will take an optuna trial and produce a configuration to 
# pass to the MLForecast constructor. 
def init_config(trial: optuna.Trial):
    # lag_transforms = [
    #     ExponentiallyWeightedMean(alpha=0.3),
    #     RollingMean(window_size=24 * 7, min_samples=1),
    # ]
    # lag_to_transform = trial.suggest_categorical('lag_to_transform', [24, 48])
    lags = [7, 14, 30]
    lag_transforms = {
        7: [
            RollingMean(window_size = 7),
            ExpandingMean()
        ],
        14: [
            RollingMean(window_size = 14)            
        ],
        30: [
            RollingMean(window_size = 30),
            # RollingMean(window_size = 60), # too few data points for this transformation 
            # RollingMean(window_size = 90), # too few data points for this transformation
            ExponentiallyWeightedMean(alpha = 0.3)
        ]
    }
    date_features = []
    res = {
        'lags': lags, 
        'lag_transforms': lag_transforms,
        'date_features': date_features
    }
    return res 

# * Tuning fit parameters -------------------------------------------------

# The MLForecast.fit method takes some arguments that could improve the 
# forecasting performance of your models, such as dropna and static_features. 
# If you want to tune those you can provide a function to the fit_config argument.
def fit_config(trial: optuna.Trial):
    # if trial.suggest_int('use_id', 0, 1):
    #     static_features = ['unique_id']
    # else:
    #     static_features = None
    static_features = []
    prediction_intervals = PredictionIntervals(
        h = horizon, n_windows = 2, method = 'conformal_distribution'
    )
    dropna = True
    res = {
        'static_features': static_features, 
        'prediction_intervals': prediction_intervals,
        'dropna': dropna
    }
    return res

# * Engines ---------------------------------------------------------------
models_auto_mlf = {
    'elanet_default': AutoElasticNet(), 
    'lgbm_default': AutoLightGBM(),
    'my_lgbm': my_lgb
}
auto_mlf = AutoMLForecast(
    models = models_auto_mlf,
    freq = 'D',
    init_config = init_config,
    fit_config = fit_config,
    # season_length = 7, # not to use with init_config
)

# * Tuning ----------------------------------------------------------------

# Finally you can run the optimization process through the .fit method.
auto_mlf.fit(
    train_df,
    h = horizon,
    n_windows = 6, # 12 is not possible because too short time series (also for Conformal intervals)
    refit = 7, # step_size
    num_samples = 20 # number of trials to run
)

# * Extracting results ----------------------------------------------------

# There is one optimization process per model. This is because 
# different models can make use of different features. So after 
# the optimization process is done for each model the best 
# configuration is used to retrain the model using all of the data. 
# These final models are MLForecast objects and are saved in the 
# models_ attribute.

auto_mlf.results_['elanet_default'].best_params
auto_mlf.results_['lgbm_default'].best_params
auto_mlf.results_['my_lgbm'].best_params

auto_mlf.models_

# * Evaluation ------------------------------------------------------------

auto_mlf_preds_df = auto_mlf.predict(h = horizon, level = levels, X_df = test_df)
auto_mlf_preds_df

evaluate(
    df = auto_mlf_preds_df \
        .merge(test_df.select_columns(), on = ['unique_id', 'ds']), 
    metrics = [bias, mae, mape, mse, rmse],
    agg_fn = 'mean'
)
plot_series(
    pd.concat([train_df, test_df]), auto_mlf_preds_df,
    max_insample_length = horizon * 2,
    level = None,
    engine = 'plotly'
).show()



# Tuning - DL Models ------------------------------------------------------
