# Time Series Forecasting: Machine Learning and Deep Learning with R & Python ----

# Lecture 1: Manipulation, Transformation & Visualization -----------------
# Marco Zanotti

# Goals:
# - Learn timetk data wrangling functionality
# - Commonly used time series transformations
# - Commonly used time series visualizations


# Packages ----------------------------------------------------------------

import numpy as np
import pandas as pd
# import polars as pl
import pytimetk as tk
# from datetime import datetime

pd.set_option("display.max_rows", 5)



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

# Polars
# subscribers_pl_df = pl.read_csv(
#     "data/subscribers/subscribers.csv", try_parse_dates=True
# )
# analytics_pl_df = pl.read_csv(
#     "data/subscribers/analytics_hourly.csv", try_parse_dates=True
# )
# events_pl_df = pl.read_csv(
#     "data/subscribers/events.csv", try_parse_dates=True
# )
# subscribers_pl_df.glimpse()



# Manipulation ------------------------------------------------------------

# * Summarize by Time -----------------------------------------------------

# - Apply commonly used aggregations
# - High-to-Low Frequency

# to daily
subscribers_df \
    .assign(value = 1) \
    .groupby(['member_rating']) \
    .summarize_by_time(
        date_column = 'optin_time',
        value_column = 'value',
        freq = 'D',
        agg_func = 'sum'
    )

# to weekly
subscribers_df \
    .assign(value = 1) \
    .summarize_by_time(
        date_column = 'optin_time',
        value_column = 'value',
        freq = 'W',
        agg_func = 'sum'
    )

# to monthly
subscribers_df \
    .assign(value = 1) \
    .summarize_by_time(
        date_column = 'optin_time',
        value_column = 'value',
        freq = 'M',
        agg_func = 'sum'
    )

# other aggregations
subscribers_df \
    .assign(value = 1) \
    .summarize_by_time(
        date_column = 'optin_time',
        value_column = 'value',
        freq = 'M',
        agg_func = ['mean', 'std', 'median']
    )


subscribers_daily_df = subscribers_df \
    .assign(value = 1) \
    .summarize_by_time(
        date_column = 'optin_time',
        value_column = 'value',
        freq = 'D',
        agg_func = 'sum'
    )

analytics_daily_df = analytics_df \
    .rename(columns={'dateHour' : 'date'}) \
    .summarize_by_time(
        date_column = 'date',
        value_column = ['pageViews', 'organicSearches', 'sessions'],
        freq = 'D',
        agg_func = 'sum'
    )


# * Pad by Time -----------------------------------------------------------

# - Filling in time series gaps
# - Low-to-High Frequency (un-aggregating)

# fill daily gaps
