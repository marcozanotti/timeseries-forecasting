# Time Series Forecasting: Machine Learning and Deep Learning with R & Python ----

# Lecture 7: Deep Learning Algorithms -------------------------------------
# Marco Zanotti

# Goals:
# - GluonTS / Torch
# - Deep AR
# - NBEATS
# - GP Forecaster
# - Deep State



# Packages ----------------------------------------------------------------

import os
import pickle
import re

import numpy as np
import pandas as pd
# import polars as pl
import pytimetk as tk


from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MQLoss
from neuralforecast.losses.pytorch import DistributionLoss

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

# In DL models prediction intervals are embedded in the model, usually
# through the choice of the loss function. 
help(MQLoss)
# Some DL models are probabilistic by construction.
help(DistributionLoss)

# * Cross-validation Plan -------------------------------------------------

pex.plot_cross_validation_plan(
    data_prep_df, freq = 'D', h = horizon, 
    n_windows = 1, step_size = 1, 
    engine = 'matplotlib'
)

# * External Regressors ---------------------------------------------------

# back-transform the y to use mlforecast full workflow
# but results are not directly comparable to those of TS because 
# the predictions are automatically back-transformed
# y_df = data_prep_df \
#     .select_columns() \
#     .back_transform_data(params = params)
# y_xregs_df = data_prep_df \
#     .select_columns(
#         regex = '(event)|(holiday)|(inter_)|(ds_mweek_2)|(_sin_)|(_cos_)'
#     ) \
#     .back_transform_data(params = params)
y_df = data_prep_df.select_columns()
y_xregs_df = data_prep_df



# MLP ---------------------------------------------------------------------

# One of the simplest neural architectures are Multi Layer Perceptrons 
# (MLP) composed of stacked Fully Connected Neural Networks trained with 
# backpropagation. Each node in the architecture is capable of modeling
#  non-linear relationships granted by their activation functions. Novel 
# activations like Rectified Linear Units (ReLU) have greatly improved 
# the ability to fit deeper networks overcoming gradient vanishing 
# problems that were associated with Sigmoid and TanH activations. 
# For the forecasting task the last layer is changed to follow a 
# auto-regression problem.

# https://nixtlaverse.nixtla.io/neuralforecast/models.mlp.html

# - Baseline model for DL
from neuralforecast.models import MLP

# * Engines ---------------------------------------------------------------

models_mlp = [
    MLP(
        h = horizon,
        input_size = 30,
        num_layers = 2,
        hidden_size = 128,
        max_steps = 10,
        loss = MQLoss(level = levels),
        random_seed = 0,
        alias = 'MLP' 
    ),
    MLP(
        h = horizon,
        input_size = 14,
        num_layers = 2,
        hidden_size = 128,
        max_steps = 10,
        loss = MQLoss(level = levels),
        futr_exog_list = ['y_lag_56', 'event'],
        hist_exog_list = ['y_lag_56', 'event'],
        random_seed = 0,
        alias = 'MLP_exog' 
    )
]

nf_mlp = NeuralForecast(
    models = models_mlp,
    freq = 'D'
)

# * Evaluation ------------------------------------------------------------

cv_res_mlp = pex.calibrate_evaluate_plot(
    object = nf_mlp, data = y_xregs_df.dropna(), 
    h = horizon, level = levels, loss = 'MQLoss',
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_mlp['cv_results']
cv_res_mlp['accuracy_table']
cv_res_mlp['plot'].show()

# * Refitting & Forecasting -----------------------------------------------

nf_mlp.fit(df = y_xregs_df.dropna()) 

preds_df_mlp = nf_mlp.predict(futr_df = forecast_df) \
    .rename(columns = lambda x: re.sub('-median', '', x))
preds_df_mlp

plot_series(
    data_prep_df, preds_df_mlp,
    max_insample_length = horizon * 2,
    level = levels,
    engine = 'plotly'
).show()



# RNN ---------------------------------------------------------------------

# Multi Layer Elman RNN (RNN), with MLP decoder. The network has tanh or 
# relu non-linearities, it is trained using ADAM stochastic gradient 
# descent. The network accepts static, historic and future exogenous data.

# https://nixtlaverse.nixtla.io/neuralforecast/models.rnn.html

# - Baseline model for DL in time series
from neuralforecast.models import RNN

# * Engines ---------------------------------------------------------------

# Feauture engineering is usually performed automatically by DL models. 
# If you want to add some specific feauture you have to manually create
# them and use the 'futr_exog_list', 'hist_exog_list' or 'stat_exog_list' 
# parameters within the engine.

models_rnn = [
    RNN(
        h = horizon,
        input_size = -1,
        inference_input_size = -1,
        encoder_n_layers = 2,
        encoder_hidden_size = 128,
        encoder_activation = 'relu',
        decoder_layers = 2,
        decoder_hidden_size = 128,
        max_steps = 100,
        loss = MQLoss(level = levels),
        random_seed = 0,
        alias = 'RNN'                
    ),
    RNN(
        h = horizon,
        input_size = -1,
        inference_input_size = -1,
        encoder_n_layers = 2,
        encoder_hidden_size = 128,
        encoder_activation = 'relu',
        decoder_layers = 2,
        decoder_hidden_size = 128,
        max_steps = 300,
        loss = MQLoss(level = levels),
        # scaler_type = 'robust',
        futr_exog_list = ['y_lag_56', 'event'],
        hist_exog_list = ['y_lag_56', 'event'],
        # stat_exog_list = [],
        random_seed = 0,
        alias = 'RNN_exog'                
    )
]

nf_rnn = NeuralForecast(
    models = models_rnn,
    freq = 'D'
)

# * Evaluation ------------------------------------------------------------

cv_res_rnn = pex.calibrate_evaluate_plot(
    object = nf_rnn, data = y_xregs_df.dropna(), 
    h = horizon, level = levels, loss = 'MQLoss',
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_rnn['cv_results']
cv_res_rnn['accuracy_table']
cv_res_rnn['plot'].show()

# * Refitting & Forecasting -----------------------------------------------

# ATTENTION: the .predict method have to be called after the .fit
# on the same NeuralForecast object. Do not store the fit.
# No need to provide the forecast horizon to the model because it 
# has been trained for that!

nf_rnn.fit(df = y_xregs_df.dropna()) 

preds_df_rnn = nf_rnn.predict(futr_df = forecast_df) \
    .rename(columns = lambda x: re.sub('-median', '', x))
preds_df_rnn

plot_series(
    data_prep_df, preds_df_rnn,
    max_insample_length = horizon * 2,
    level = levels,
    engine = 'plotly'
).show()



# DeepNPTS ----------------------------------------------------------------

# Deep Non-Parametric Time Series Forecaster (DeepNPTS) is a non-parametric 
# baseline model for time-series forecasting. This model generates 
# predictions by sampling from the empirical distribution according to a 
# tunable strategy. This strategy is learned by exploiting the information 
# across multiple related time series. This model provides a strong, 
# simple baseline for time series forecasting.

# ATTENTION: This implementation differs from the original work in that a
#  weighted sum of the empirical distribution is returned as forecast. 
# Therefore, it only supports point losses.

# https://nixtlaverse.nixtla.io/neuralforecast/models.deepnpts.html

# - Baseline non-parametric global model 
from neuralforecast.models import DeepNPTS

# * Engines ---------------------------------------------------------------

models_dnpts = [
    DeepNPTS(
        h = horizon,
        input_size = 30,
        n_layers = 2,
        hidden_size = 128,
        max_steps = 100,
        random_seed = 0,
        alias = 'DeepNPTS' 
    ),
    DeepNPTS(
        h = horizon,
        input_size = 30,
        n_layers = 2,
        hidden_size = 128,
        max_steps = 300,
        futr_exog_list = ['y_lag_56', 'event'],
        hist_exog_list = ['y_lag_56', 'event'],
        random_seed = 0,
        alias = 'DeepNPTS_exog' 
    )
]

nf_dnpts = NeuralForecast(
    models = models_dnpts,
    freq = 'D'
)

# * Evaluation ------------------------------------------------------------

cv_res_dnpts = pex.calibrate_evaluate_plot(
    object = nf_dnpts, data = y_xregs_df.dropna(), 
    h = horizon, level = None,
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_dnpts['cv_results']
cv_res_dnpts['accuracy_table']
cv_res_dnpts['plot'].show()



# LSTM & GRU --------------------------------------------------------------

# The Long Short-Term Memory Recurrent Neural Network (LSTM), uses a 
# multilayer LSTM encoder and an MLP decoder. It builds upon the LSTM-cell 
# that improves the exploding and vanishing gradients of classic RNN’s. 
# This network has been extensively used in sequential prediction tasks 
# like language modeling, phonetic labeling, and forecasting. 
# LSTM encoder, with MLP decoder. The network has tanh or relu 
# non-linearities, it is trained using ADAM stochastic gradient descent. 
# The network accepts static, historic and future exogenous data.

# Cho et. al proposed the Gated Recurrent Unit (GRU) to improve on 
# LSTM and Elman cells. The predictions at each time are given by a 
# MLP decoder. This architecture follows closely the original Multi 
# Layer Elman RNN with the main difference being its use of the GRU cells.
#  The network has tanh or relu non-linearities, it is trained using 
# ADAM stochastic gradient descent. The network accepts static, 
# historic and future exogenous data, flattens the inputs.

# https://nixtlaverse.nixtla.io/neuralforecast/models.lstm.html
# https://nixtlaverse.nixtla.io/neuralforecast/models.gru.html

from neuralforecast.models import LSTM, GRU

# * Engines ---------------------------------------------------------------

models_lstm = [
    LSTM(
        h = horizon,
        input_size = -1,
        inference_input_size = -1,
        encoder_n_layers = 2,
        encoder_hidden_size = 128,
        decoder_layers = 2,
        decoder_hidden_size = 128,
        max_steps = 100,
        loss = MQLoss(level = levels),
        random_seed = 0,
        alias = 'LSTM'                
    ),
    LSTM(
        h = horizon,
        input_size = -1,
        inference_input_size = -1,
        encoder_n_layers = 2,
        encoder_hidden_size = 128,
        decoder_layers = 2,
        decoder_hidden_size = 128,
        max_steps = 300,
        loss = MQLoss(level = levels),
        futr_exog_list = ['y_lag_56', 'event'],
        hist_exog_list = ['y_lag_56', 'event'],
        random_seed = 0,
        alias = 'LSTM_exog'                
    ),
    GRU(
        h = horizon,
        input_size = -1,
        inference_input_size = -1,
        encoder_n_layers = 2,
        encoder_hidden_size = 128,
        encoder_activation = 'relu',
        decoder_layers = 2,
        decoder_hidden_size = 128,
        max_steps = 100,
        loss = MQLoss(level = levels),
        random_seed = 0,
        alias = 'GRU'  
    ), 
    GRU(
        h = horizon,
        input_size = -1,
        inference_input_size = -1,
        encoder_n_layers = 2,
        encoder_hidden_size = 128,
        encoder_activation = 'relu',
        decoder_layers = 2,
        decoder_hidden_size = 128,
        max_steps = 300,
        loss = MQLoss(level = levels),
        futr_exog_list = ['y_lag_56', 'event'],
        hist_exog_list = ['y_lag_56', 'event'],
        random_seed = 0,
        alias = 'GRU_exog'  
    )
]

nf_lstm = NeuralForecast(
    models = models_lstm,
    freq = 'D'
)

# * Evaluation ------------------------------------------------------------

cv_res_lstm = pex.calibrate_evaluate_plot(
    object = nf_lstm, data = y_xregs_df.dropna(), 
    h = horizon, level = levels, loss = 'MQLoss',
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_lstm['cv_results']
cv_res_lstm['accuracy_table']
cv_res_lstm['plot'].show()



# Dilated RNN -------------------------------------------------------------

# The Dilated Recurrent Neural Network (DilatedRNN) addresses common 
# challenges of modeling long sequences like vanishing gradients, 
# computational efficiency, and improved model flexibility to model 
# complex relationships while maintaining its parsimony. The DilatedRNN 
# builds a deep stack of RNN layers using skip conditions on the temporal 
# and the network’s depth dimensions. The temporal dilated recurrent 
# skip connections offer the capability to focus on multi-resolution inputs.

# https://nixtlaverse.nixtla.io/neuralforecast/models.dilated_rnn.html

from neuralforecast.models import DilatedRNN

# * Engines ---------------------------------------------------------------

models_drnn = [
    DilatedRNN(
        h = horizon,
        input_size = -1,
        inference_input_size = -1,
        cell_type = 'RNN',
        encoder_hidden_size = 128,
        decoder_hidden_size = 128,
        max_steps = 100,
        loss = MQLoss(level = levels),
        random_seed = 0,
        alias = 'DRNN'                
    ),
    DilatedRNN(
        h = horizon,
        input_size = -1,
        inference_input_size = -1,
        cell_type = 'RNN',
        encoder_hidden_size = 128,
        decoder_hidden_size = 128,
        max_steps = 300,
        loss = MQLoss(level = levels),
        futr_exog_list = ['y_lag_56', 'event'],
        hist_exog_list = ['y_lag_56', 'event'],
        random_seed = 0,
        alias = 'DRNN_exog'                
    )
]

nf_drnn = NeuralForecast(
    models = models_drnn,
    freq = 'D'
)

# * Evaluation ------------------------------------------------------------

cv_res_drnn = pex.calibrate_evaluate_plot(
    object = nf_drnn, data = y_xregs_df.dropna(), 
    h = horizon, level = levels, loss = 'MQLoss',
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_drnn['cv_results']
cv_res_drnn['accuracy_table']
cv_res_drnn['plot'].show()



# TCN & BiTCN -------------------------------------------------------------

# For long time in deep learning, sequence modelling was synonymous with 
# recurrent networks, yet several papers have shown that simple convolutional
# architectures can outperform canonical recurrent networks like LSTMs by
# demonstrating longer effective memory. By skipping temporal connections
# the causal convolution filters can be applied to larger time spans while
# remaining computationally efficient.
# Temporal Convolution Network (TCN), with MLP decoder. The historical 
# encoder uses dilated skip connections to obtain efficient long memory, 
# while the rest of the architecture allows for future exogenous alignment.

# Bidirectional Temporal Convolutional Network (BiTCN) is a forecasting 
# architecture based on two temporal convolutional networks (TCNs). 
# The first network (‘forward’) encodes future covariates of the time 
# series, whereas the second network (‘backward’) encodes past 
# observations and covariates. This method allows to preserve the 
# temporal information of sequence data, and is computationally more 
# efficient than common RNN methods (LSTM, GRU, …). As compared to 
# Transformer-based methods, BiTCN has a lower space complexity, i.e. 
# it requires orders of magnitude less parameters. This model may be 
# a good choice if you seek a small model (small amount of trainable 
# parameters) with few hyperparameters to tune (only 2).

# https://nixtlaverse.nixtla.io/neuralforecast/models.tcn.html
# https://nixtlaverse.nixtla.io/neuralforecast/models.bitcn.html

from neuralforecast.models import TCN, BiTCN

# * Engines ---------------------------------------------------------------

models_tcn = [
    TCN(
        h = horizon,
        input_size = -1,
        inference_input_size = -1,
        kernel_size = 2, 
        dilations = [1, 7, 14],
        encoder_hidden_size = 128,
        decoder_layers = 2,
        decoder_hidden_size = 128,
        max_steps = 100,
        loss = MQLoss(level = levels),
        random_seed = 0,
        alias = 'TCN'                
    ),
    TCN(
        h = horizon,
        input_size = -1,
        inference_input_size = -1,
        kernel_size = 2, 
        dilations = [1, 7, 14],
        encoder_hidden_size = 128,
        decoder_layers = 2,
        decoder_hidden_size = 128,
        max_steps = 100,
        loss = MQLoss(level = levels),
        futr_exog_list = ['y_lag_56', 'event'],
        hist_exog_list = ['y_lag_56', 'event'],
        random_seed = 0,
        alias = 'TCN_exog'                
    ),
    BiTCN(
        h = horizon,
        input_size = 30,
        hidden_size = 128,
        max_steps = 100,
        loss = MQLoss(level = levels),
        random_seed = 0,
        alias = 'BiTCN'                
    ),
    BiTCN(
        h = horizon,
        input_size = 30,
        hidden_size = 128,
        max_steps = 100,
        loss = MQLoss(level = levels),
        futr_exog_list = ['y_lag_56', 'event'],
        hist_exog_list = ['y_lag_56', 'event'],
        random_seed = 0,
        alias = 'BiTCN_exog'                
    )
]

nf_tcn = NeuralForecast(
    models = models_tcn,
    freq = 'D'
)

# * Evaluation ------------------------------------------------------------

cv_res_tcn = pex.calibrate_evaluate_plot(
    object = nf_tcn, data = y_xregs_df.dropna(), 
    h = horizon, level = levels, loss = 'MQLoss',
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_tcn['cv_results']
cv_res_tcn['accuracy_table']
cv_res_tcn['plot'].show()



# NLinear & DLinear -------------------------------------------------------

# NLinear is a simple and fast yet accurate time series forecasting 
# model for long-horizon forecasting. The architecture aims to boost the 
# performance when there is a distribution shift in the dataset: 
# first subtracts the input by the last value of the sequence; 
# then, the input goes through a linear layer, and the subtracted part 
# is added back before making the final prediction.

# DLinear is a simple and fast yet accurate time series forecasting 
# model for long-horizon forecasting. The architecture has the following 
# distinctive features: 
# - Uses Autoformmer’s trend and seasonality decomposition. 
# - Simple linear layers for trend and seasonality component.

# https://nixtlaverse.nixtla.io/neuralforecast/models.nlinear.html
# https://nixtlaverse.nixtla.io/neuralforecast/models.dlinear.html

# - Benchmark models for Transformers
from neuralforecast.models import NLinear, DLinear

# * Engines ---------------------------------------------------------------

models_lin = [
    NLinear(
        h = horizon, 
        input_size = 365,
        max_steps = 100,
        loss = MQLoss(level = levels),
        random_seed = 0       
    ), 
    DLinear(
        h = horizon, 
        input_size = 365,
        max_steps = 100,
        moving_avg_window = 31,
        loss = MQLoss(level = levels),
        random_seed = 0
    )
]

nf_lin = NeuralForecast(
    models = models_lin,
    freq = 'D'
)

# * Evaluation ------------------------------------------------------------

cv_res_lin = pex.calibrate_evaluate_plot(
    object = nf_lin, data = y_xregs_df.dropna(), 
    h = horizon, level = levels, loss = 'MQLoss',
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_lin['cv_results']
cv_res_lin['accuracy_table']
cv_res_lin['plot'].show()



# DeepAR ------------------------------------------------------------------

# The DeepAR model produces probabilistic forecasts based on an 
# autoregressive recurrent neural network optimized on panel data using 
# cross-learning. DeepAR obtains its forecast distribution uses a Markov 
# Chain Monte Carlo sampler. 

# Given the sampling procedure during inference, DeepAR only supports 
# DistributionLoss as training loss. Note that DeepAR generates a 
# non-parametric forecast distribution using Monte Carlo. We use 
# this sampling procedure also during validation to make it closer 
# to the inference procedure. Therefore, only the MQLoss is available 
# for validation. Aditionally, Monte Carlo implies that historic 
# exogenous variables are not available for the model.

# https://nixtlaverse.nixtla.io/neuralforecast/models.deepar.html

from neuralforecast.models import DeepAR

# * Engines ---------------------------------------------------------------

models_deepar = [
    DeepAR(
        h = horizon,
        input_size = 48,
        lstm_n_layers = 2,
        lstm_hidden_size = 128,
        decoder_hidden_layers = 0,
        decoder_hidden_size = 0, 
        trajectory_samples = 200,
        max_steps = 100,
        loss = DistributionLoss(distribution = 'Normal', level = levels, return_params = False),
        random_seed = 0,
        alias = 'DeepAR'
    ),
    DeepAR(
        h = horizon,
        input_size = 48,
        lstm_n_layers = 2,
        lstm_hidden_size = 128,
        decoder_hidden_layers = 0,
        decoder_hidden_size = 0, 
        trajectory_samples = 200,
        max_steps = 300,
        loss = DistributionLoss(distribution = 'Normal', level = levels, return_params = False),
        futr_exog_list = ['y_lag_56', 'event'],
        random_seed = 0,
        alias = 'DeepAR_exog'
    )
]

nf_deepar = NeuralForecast(
    models = models_deepar,
    freq = 'D'
)

# * Evaluation ------------------------------------------------------------

cv_res_deepar = pex.calibrate_evaluate_plot(
    object = nf_deepar, data = y_xregs_df.dropna(), 
    h = horizon, level = levels, loss = 'DistributionLoss',
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_deepar['cv_results']
cv_res_deepar['accuracy_table']
cv_res_deepar['plot'].show()



# NBEATS & NHITS ----------------------------------------------------------

# The Neural Basis Expansion Analysis (NBEATS) is an MLP-based deep 
# neural architecture with backward and forward residual links. The 
# network has two variants: (1) in its interpretable configuration, 
# NBEATS sequentially projects the signal into polynomials and harmonic 
# basis to learn trend and seasonality components; (2) in its generic 
# configuration, it substitutes the polynomial and harmonic basis for 
# identity basis and larger network’s depth. This method proved 
# state-of-the-art performance on the M3, M4, and Tourism Competition 
# datasets, improving accuracy by 3% over the ESRNN M4 competition winner.

# The Neural Basis Expansion Analysis with Exogenous (NBEATSx), 
# incorporates projections to exogenous temporal variables available 
# at the time of the prediction.

# Long-horizon forecasting is challenging because of the volatility 
# of the predictions and the computational complexity. To solve this 
# problem we created the Neural Hierarchical Interpolation for Time 
# Series (NHITS). NHITS builds upon NBEATS and specializes its partial 
# outputs in the different frequencies of the time series through 
# hierarchical interpolation and multi-rate input processing. On the 
# long-horizon forecasting task NHITS improved accuracy by 25% on 
# AAAI’s best paper award the Informer, while being 50x faster.
# The model is composed of several MLPs with ReLU non-linearities. 
# Blocks are connected via doubly residual stacking principle with the 
# backcast and forecast outputs of the l-th block. Multi-rate input 
# pooling, hierarchical interpolation and backcast residual connections 
# together induce the specialization of the additive predictions in 
# different signal bands, reducing memory footprint and computational 
# time, thus improving the architecture parsimony and accuracy.

# https://nixtlaverse.nixtla.io/neuralforecast/models.nbeats.html
# https://nixtlaverse.nixtla.io/neuralforecast/models.nbeatsx.html

from neuralforecast.models import NBEATS, NBEATSx, NHITS

# * Engines ---------------------------------------------------------------

models_nbeats = [
    NBEATS(
        h = horizon, 
        input_size = 30,
        stack_types = ['identity', 'trend', 'seasonality'],
        loss = MQLoss(level = levels),
        max_steps = 100,
        random_seed = 0
    ), 
    NBEATSx(
        h = horizon, 
        input_size = 30,
        stack_types = ['identity', 'trend', 'seasonality'],
        loss = MQLoss(level = levels),
        max_steps = 300,
        futr_exog_list = ['y_lag_56', 'event'],
        hist_exog_list = ['y_lag_56', 'event'],
        random_seed = 0
    ), 
    NHITS(
        h = horizon, 
        input_size = 30,
        n_freq_downsample = [2, 1, 1],
        loss = MQLoss(level = levels),
        max_steps = 100,
        random_seed = 0, 
        alias = 'NHITS'
    ),
    NHITS(
        h = horizon, 
        input_size = 30,
        n_freq_downsample = [2, 1, 1],
        loss = MQLoss(level = levels),
        max_steps = 300,
        futr_exog_list = ['y_lag_56', 'event'],
        hist_exog_list = ['y_lag_56', 'event'],
        random_seed = 0, 
        alias = 'NHITS_exog'
    )
]

nf_nbeats = NeuralForecast(
    models = models_nbeats,
    freq = 'D'
)

# * Evaluation ------------------------------------------------------------

cv_res_nbeats = pex.calibrate_evaluate_plot(
    object = nf_nbeats, data = y_xregs_df.dropna(), 
    h = horizon, level = levels, loss = 'MQLoss',
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_nbeats['cv_results']
cv_res_nbeats['accuracy_table']
cv_res_nbeats['plot'].show()






# XX ---------------------------------------------------------------------

# Multi Layer Elman RNN (RNN), with MLP decoder. The network has tanh or 
# relu non-linearities, it is trained using ADAM stochastic gradient 
# descent. The network accepts static, historic and future exogenous data.

# https://nixtlaverse.nixtla.io/neuralforecast/models.rnn.html

from neuralforecast.models import XX

# * Engines ---------------------------------------------------------------

models_xx = [

]

nf_xx = NeuralForecast(
    models = models_xx,
    freq = 'D'
)

# * Evaluation ------------------------------------------------------------

cv_res_xx = pex.calibrate_evaluate_plot(
    object = nf_xx, data = y_xregs_df.dropna(), 
    h = horizon, level = levels, loss = 'MQLoss',
    engine = 'plotly', max_insample_length = horizon * 2  
)
cv_res_xx['cv_results']
cv_res_xx['accuracy_table']
cv_res_xx['plot'].show()

# * Refitting & Forecasting -----------------------------------------------

nf_xx.fit(df = y_xregs_df.dropna()) 

preds_df_xx = nf_xx.predict(futr_df = forecast_df) \
    .rename(columns = lambda x: re.sub('-median', '', x))
preds_df_xx

plot_series(
    data_prep_df, preds_df_xx,
    max_insample_length = horizon * 2,
    level = levels,
    engine = 'plotly'
).show()





# DEEP LEARNING
# - Pros:
#   - Create very powerful models by combining Machine Learning & Deep Learning
#   - Deep Learning is great for global modeling time series
# - Cons:
#   - Lower to train with respect to TS / ML algos
#   - More difficult to train
#   - Does not always factor with external regressors
#     - Solution 1: Run DL without. Run ML on the Residuals.
#     - Solution 2: Create an Ensemble with ML & DL