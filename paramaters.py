import os
import time
from tensorflow.keras.layers import LSTM
import yahoo_fin
import collections
from yahoo_fin import stock_info as si

# Window size or the sequence length (use to be 80)
N_STEPS = 50
# Lookup step, 1 is the next day
LOOKUP_STEP = 1

# test ratio size, 0.2 is 20%
TEST_SIZE = 0.08
# features to use
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]
# date now
date_now = time.strftime("%Y-%m-%d")

### model parameters

N_LAYERS = 3
# LSTM cell
CELL = LSTM
# 256 LSTM neurons
UNITS = 256
# 40% dropout
DROPOUT = 0.4
# whether to use bidirectional RNNs
BIDIRECTIONAL = False

### training parameters

# mean absolute error loss
# LOSS = "mae"
# huber loss
LOSS = "huber_loss"
OPTIMIZER = "adam"
BATCH_SIZE = 64
#was 400 was also 200
EPOCHS = 400

# Tesla stock market
ticker = "TSLA"
ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")

inflationtickerGBP = "GBP=X"
inflation_data_filename = os.path.join("data", f"{inflationtickerGBP}_{date_now}.csv")

# model name to save, making it as unique as possible based on parameters
model_name = f"{date_now}_{ticker}-{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"
if BIDIRECTIONAL:
    model_name += "-b"


