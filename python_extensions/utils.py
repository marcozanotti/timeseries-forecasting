
import pandas as pd
import numpy as np
import pandas_flavor as pf
from statsmodels.gam.api import BSplines
from mlforecast import MLForecast
from sklearn.linear_model import LinearRegression


# function to perform data standardization (mean 0, stdev 1)
def standardize(x):
    print(f'mean x = {np.mean(x)}, stdev x = {np.std(x)}')
    return (x - np.mean(x)) / np.std(x)

# function to invert standardization
def inv_standardize(x, mean, stdev):
    return (x * stdev) + mean

# function to perform data log-interval transformation
# log(((x + offset) - a)/(b - (x + offset)))
# a = lower bound, b = upper bound
def log_interval(x, lb = 0, ub = 'auto', offset = 1):
    if (ub == 'auto'):
        ub = np.max(x) * 1.10 
    print(f'lower bound = {lb}, upper bound = {ub}, offset = {offset}')
    return np.log(((x + offset) - lb)/(ub - (x + offset)))

# function to invert log-interval transformation
# (b-a)*(exp(x)) / (1 + exp(x)) + a - offset
def inv_log_interval(x, lb, ub, offset = 1):
    return (ub - lb) * (np.exp(x)) / (1 + np.exp(x)) + lb - offset

# function to add basis splines to the dataframe
@pf.register_dataframe_method
def augment_bsplines(data, column_name = 'ds_index_num', df = 5, degree = 3):
    
    bs = BSplines(
        data[column_name], 
        df = df, 
        degree = degree, 
        include_intercept = False
    )
    bs_df = pd.DataFrame(bs.basis)
    
    col_names = []
    for i in range(1, (len(bs_df.columns) + 1)):
        col_names.append(f'bspline_{i}_degree_{degree}')
    bs_df.columns = col_names
    bs_df.index = data.index

    data_splines = pd.concat([data, bs_df], axis = 1)

    return data_splines

# function to perform time series regression and plotting
@pf.register_dataframe_method
def plot_time_series_regression(data):
    
    # fit linear regression
    fcst = MLForecast(models = LinearRegression(), freq = 'D')
    fcst.fit(data, static_features = [], fitted = True)

    # extract fitted values
    data_fitted_values = fcst.forecast_fitted_values()

    # plot actual vs fitted
    p = data_fitted_values \
        .rename(columns = {'y': 'actual', 'LinearRegression': 'fitted'}) \
        .melt(id_vars = ['unique_id', 'ds']) \
        .plot_timeseries(
            date_column = 'ds', 
            value_column = 'value',
            color_column = 'variable',
            smooth = False
        )

    return p