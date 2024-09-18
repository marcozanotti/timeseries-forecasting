
import pandas as pd
import numpy as np
import pandas_flavor as pf
from mlforecast import MLForecast
from sklearn.linear_model import LinearRegression


# function to perform data standardization (mean 0, stdev 1)
def normalize(x):
    return (x - np.mean(x)) / np.std(x)

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