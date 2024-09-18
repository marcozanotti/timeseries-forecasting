# Time Series Forecasting: Machine Learning and Deep Learning with R & Python ----

# Lecture 2: Features Engineering & Recipes -------------------------------
# Marco Zanotti

# Goals:
# - Learn advanced features engineering workflows & techniques
# - Learn how to use recipes

# Challenges:
# - Challenge 1 - Feature Engineering



# Packages ----------------------------------------------------------------

import numpy as np
import pandas as pd
# import polars as pl
import pytimetk as tk

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

from utilsforecast.feature_engineering import fourier, trend, time_features, pipeline
from mlforecast import MLForecast
from sklearn.linear_model import LinearRegression

import python_extensions as pex

pd.set_option("display.max_rows", 3)



# Data --------------------------------------------------------------------

# Pandas
subscribers_df = pd.read_csv(
    "data/subscribers/subscribers.csv", 
    parse_dates=['optin_time']
)
analytics_df = pd.read_csv(
    "data/subscribers/analytics_hourly.csv", 
    parse_dates=['dateHour'],
    date_format='%Y%m%d%H'
)
events_df = pd.read_csv(
    "data/subscribers/events.csv", parse_dates=['event_date']
)

subscribers_df.glimpse()
analytics_df.glimpse()
events_df.glimpse()



# Features Engineering ----------------------------------------------------

# Pre-processing Data

# transform time series with log1p and standardization
# fix missing values at beginning of series
# clean anomaly on '2018-11-19'

# subscribers data
data_prep_df = subscribers_df \
    .assign(value = 1) \
    .rename(columns = {"optin_time": "date"}) \
    .summarize_by_time(
        date_column = 'date',
        value_column = 'value',
        freq = 'D',
        agg_func = 'sum'
    ) \
    .pad_by_time(
        date_column = 'date',
        freq = 'D',
        start_date = '2018-01-06'
    ) \
    .fillna(0) \
    .transform_columns(columns = 'value', transform_func = np.log1p) \
    .transform_columns(columns = 'value', transform_func = pex.normalize) \
    .filter_by_time(
        date_column = 'date', 
        start_date = '2018-07-03' 
    ) \
    .anomalize(
        date_column = 'date', value_column = 'value',
        method = 'twitter',
        iqr_alpha = 0.025,
        max_anomalies = 0.2,
        clean_alpha = 0.5,
        clean = 'min-max'
    ) \
    .assign(id = 'subscribers') \
    .reindex(columns = ['id', 'date', 'observed_clean']) \
    .rename(columns = {'observed_clean': 'value'})   
data_prep_df \
    .plot_timeseries(
        date_column = 'date',
        value_column = 'value',
        smooth = False
    )

# Nixtla's default format 
# 'unique_id', 'ds', 'y'
data_prep_df = data_prep_df \
    .rename(
        columns = {
            'id': 'unique_id',
            'date': 'ds',
            'value': 'y'
        }
    )
data_prep_df


# analytics data
analytics_prep_df = analytics_df \
    .rename(columns = {'dateHour': 'date'}) \
    .summarize_by_time(
        date_column = 'date',
        value_column = ['pageViews', 'organicSearches', 'sessions'],
        freq = 'D',
        agg_func = 'sum'
    ) \
    .pad_by_time(
        date_column = 'date',
        freq = 'D'
    ) \
    .fillna(0) \
    .transform_columns(
        columns = ['pageViews', 'organicSearches', 'sessions'], 
        transform_func = np.log1p
    ) \
    .transform_columns(
        columns = ['pageViews', 'organicSearches', 'sessions'], 
        transform_func = pex.normalize
    ) \
    .filter_by_time(
        date_column = 'date', 
        start_date = '2018-07-03' 
    )
analytics_prep_df \
    .melt(id_vars = 'date', value_vars = ['pageViews', 'organicSearches', 'sessions']) \
    .groupby('variable') \
    .plot_timeseries(
        date_column = 'date',
        value_column = 'value',
        color_column = 'variable',
        facet_nrow = 3,
        smooth = False
    )

# events data
events_prep_df = events_df \
    .assign(event = 1) \
    .rename(columns = {"event_date": "date"}) \
    .drop("id", axis = 1) \
    .summarize_by_time(
        date_column = 'date',
        value_column = 'event',
        freq = 'D',
        agg_func = 'sum'
    ) \
    .pad_by_time(
        date_column = 'date',
        freq = 'D'
    ) \
    .fillna(0)
events_prep_df['event'] = events_prep_df['event'].astype('int64')  
events_prep_df \
    .plot_timeseries(
        date_column = 'date',
        value_column = 'event',
        smooth = False
    )


# * Time-Based Features ---------------------------------------------------

# calendar
data_prep_df \
    .augment_timeseries_signature(date_column = 'ds')

# holidays
data_prep_df \
    .augment_holiday_signature(date_column = 'ds', country_name = 'US')


# * Trend-Based Features --------------------------------------------------

# linear trend
data_prep_df \
    .augment_timeseries_signature(date_column = 'ds') \
    .reindex(columns = ['unique_id', 'ds', 'y', 'ds_index_num']) \
    .plot_time_series_regression()

# nonlinear trend - basis splines


# nonlinear trend - natural splines




# * Seasonal Features -----------------------------------------------------

# weekly seasonality (one-hot-encoding)
data_prep_tmp = data_prep_df \
    .augment_timeseries_signature(date_column = 'ds') \
    .reindex(columns = ['unique_id', 'ds', 'y', 'ds_wday_lbl'])
data_prep_tmp = pd.get_dummies(
    data_prep_tmp, columns = ['ds_wday_lbl'], 
    drop_first = True, dtype = int
)
data_prep_tmp \
    .plot_time_series_regression()

# monthly seasonality
data_prep_tmp = data_prep_df \
    .augment_timeseries_signature(date_column = 'ds') \
    .reindex(columns = ['unique_id', 'ds', 'y', 'ds_month_lbl'])
data_prep_tmp = pd.get_dummies(
    data_prep_tmp, columns = ['ds_month_lbl'], 
    drop_first = True, dtype = int
)
data_prep_tmp \
    .plot_time_series_regression()


# * Interaction Features --------------------------------------------------

# day of week * week 2 of the month
data_prep_tmp = data_prep_df \
    .augment_timeseries_signature(date_column = 'ds') \
    .reindex(columns = ['unique_id', 'ds', 'y', 'ds_mweek', 'ds_wday'])
data_prep_tmp = pd.get_dummies(
    data_prep_tmp, columns = ['ds_mweek'], 
    drop_first = True, dtype = int
) \
    .reindex(columns = ['unique_id', 'ds', 'y', 'ds_wday', 'ds_mweek_2']) \
    .assign(wday_times_week2 = lambda x: x['ds_wday'] * x['ds_mweek_2'])
data_prep_tmp \
    .plot_time_series_regression()


# * Rolling Average Features ----------------------------------------------

data_prep_df \
    .augment_rolling(
        date_column = 'ds',
        value_column = 'y',
        window_func = 'mean',
        window = [7, 14, 30, 90]
    ) \
    .dropna(axis = 0) \
    .plot_time_series_regression()

# Exponential Weighted Moving Average
data_prep_df \
    .augment_ewm(
        date_column = 'ds',
        value_column = 'y',
        window_func = 'mean',
        alpha = 0.1
    ) \
    .augment_ewm(
        date_column = 'ds',
        value_column = 'y',
        window_func = 'mean',
        alpha = 0.2
    ) \
    .augment_ewm(
        date_column = 'ds',
        value_column = 'y',
        window_func = 'mean',
        alpha = 0.3
    ) \
    .plot_time_series_regression()


# * Lag Features ----------------------------------------------------------

plot_acf(data_prep_df["y"], lags = 50)
plot_pacf(data_prep_df["y"], lags = 50)

data_prep_df \
    .augment_lags(
        date_column = 'ds', 
        value_column = 'y', 
        lags = [1, 7, 14, 30, 90]
    ) \
    .dropna(axis = 0) \
    .plot_time_series_regression()


# * Fourier Series Features -----------------------------------------------

data_prep_df \
    .augment_fourier(
        date_column = 'ds', 
        periods = [1, 7, 14, 30, 90], 
        max_order = 2
    ) \
    .plot_time_series_regression()


# * Wavelet Series Features -----------------------------------------------

tk.augment_wavelet(
    data_prep_df, 
    date_column = 'ds', 
    value_column = 'y',
    scales = [7, 14, 30, 90],  
    sample_rate = 7,
    method = 'bump'
) \
    .plot_time_series_regression()


# * External Regressor Features -------------------------------------------

# Event data features 
data_prep_df \
    .merge(events_prep_df, left_on = 'ds', right_on = 'date', how = 'left') \
    .drop('date', axis = 1) \
    .fillna(0) \
    .plot_time_series_regression()

# Analytics data features
data_prep_df \
    .merge(analytics_prep_df, left_on = 'ds', right_on = 'date', how = 'left') \
    .drop('date', axis = 1) \
    .augment_lags(
        date_column = 'ds',
        value_column = ['pageViews', 'organicSearches', 'sessions'], 
        lags = [7, 42]
    ) \
    .dropna() \
    .plot_time_series_regression()



# Features Engineering Workflow -------------------------------------------

# * Pre-processing Data ---------------------------------------------------

# - Aggregate to daily frequency
# - Pad with zero values
# - Apply log1p transform
# - Apply standardization transform
# - Filter to kee only useful period
# - Clean anomalies

# subscribers data
subscribers_prep_df = subscribers_df \
    .assign(value = 1) \
    .rename(columns = {"optin_time": "date"}) \
    .summarize_by_time(
        date_column = 'date',
        value_column = 'value',
        freq = 'D',
        agg_func = 'sum'
    ) \
    .pad_by_time(
        date_column = 'date',
        freq = 'D',
        start_date = '2018-01-06'
    ) \
    .fillna(0) \
    .transform_columns(columns = 'value', transform_func = np.log1p) \
    .transform_columns(columns = 'value', transform_func = pex.normalize) \
    .filter_by_time(
        date_column = 'date', 
        start_date = '2018-07-03' 
    ) \
    .anomalize(
        date_column = 'date', value_column = 'value',
        method = 'twitter',
        iqr_alpha = 0.025,
        max_anomalies = 0.2,
        clean_alpha = 0.5,
        clean = 'min-max'
    ) \
    .assign(id = 'subscribers') \
    .reindex(columns = ['id', 'date', 'observed_clean']) \
    .rename(columns = {'observed_clean': 'value'})

# Nixtla's default format 
# 'unique_id', 'ds', 'y'
subscribers_prep_df = subscribers_prep_df \
    .rename(
        columns = {
            'id': 'unique_id',
            'date': 'ds',
            'value': 'y'
        }
    )
subscribers_prep_df

# events data
events_prep_df = events_df \
    .assign(event = 1) \
    .rename(columns = {"event_date": "date"}) \
    .drop("id", axis = 1) \
    .summarize_by_time(
        date_column = 'date',
        value_column = 'event',
        freq = 'D',
        agg_func = 'sum'
    ) \
    .pad_by_time(
        date_column = 'date',
        freq = 'D'
    ) \
    .fillna(0)
events_prep_df['event'] = events_prep_df['event'].astype('int64')  
events_prep_df


# * Creating Features -----------------------------------------------------

# - Extend to Future Window
# - Add any lags to full dataset
# - Add rolling features to full dataset
# - Add fourier terms to full dataset
# - Add calendar variable to full dataset (+ one-hot-encoding)
# - Add any external regressors to full dataset

horizon = 7 * 8 # 8 weeks
lag_periods = [7 * 8]
rolling_periods = [30, 60, 90]

data_prep_full_df = subscribers_prep_df \
    .future_frame(date_column = 'ds', length_out = horizon) \
    .augment_lags(
        date_column = 'ds', value_column = 'y', 
        lags = lag_periods
    ) \
    .augment_rolling(
        date_column = 'ds', value_column = 'y_lag_56',
        window_func = 'mean', window = rolling_periods
    ) \
    .augment_ewm(
        date_column = 'ds', value_column = 'y_lag_56',
        window_func = 'mean', alpha = 0.1
    ) \
    .augment_ewm(
        date_column = 'ds', value_column = 'y_lag_56',
        window_func = 'mean', alpha = 0.2
    ) \
    .augment_ewm(
        date_column = 'ds', value_column = 'y_lag_56',
        window_func = 'mean', alpha = 0.3
    ) \
    .augment_fourier(
        date_column = 'ds', 
        periods = [7, 14, 30, 90, 365], 
        max_order = 2
    ) \
    .augment_timeseries_signature(date_column = 'ds') \
    .drop(
        [
            'ds_year_iso', 'ds_yearstart', 'ds_yearend', 'ds_leapyear',
            'ds_quarteryear', 'ds_quarterstart', 'ds_quarterend', 
            'ds_monthstart', 'ds_monthend', 'ds_qday', 'ds_yday',
            'ds_hour', 'ds_minute', 'ds_second', 'ds_msecond', 'ds_nsecond', 'ds_am_pm'
        ], 
        axis = 1
    ) \
    .merge(events_prep_df, left_on = 'ds', right_on = 'date', how = 'left') \
    .drop('date', axis = 1) \
    .fillna({'event': 0})
data_prep_full_df


data_prep_full_dummy_df = pd.get_dummies(
    data_prep_full_df, columns = ['ds_wday_lbl', 'ds_month_lbl'], 
    drop_first = True, dtype = int
)
data_prep_full_dummy_df


# * Separate into Modelling & Forecast Data -------------------------------

data_prep_df = data_prep_full_dummy_df.head(n = -horizon)
forecast_df = data_prep_full_dummy_df.tail(n = horizon)


# * Save Artifacts --------------------------------------------------------

