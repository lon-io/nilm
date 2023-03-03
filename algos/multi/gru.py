import tensorflow as tf
from tensorflow.keras.layers import GRU, Bidirectional, Conv1D, Dense
from tensorflow.keras.metrics import MeanAbsoluteError, RootMeanSquaredError
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def create_model(seq_len, learning_rate, clipvalue):
    model = Sequential()
    model.add(Conv1D(16, 4, activation="linear", input_shape=(seq_len, 1), padding="same", strides=1))
    model.add(Conv1D(8, 4, activation="linear", padding="same", strides=1))
    model.add(Bidirectional(GRU(64, return_sequences=True, stateful=False), merge_mode='concat'))
    model.add(Bidirectional(GRU(128, return_sequences=False, stateful=False), merge_mode='concat'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(3, activation='linear'))

    opt_adam = Adam(lr = learning_rate, clipvalue=clipvalue)

    model.compile(loss='mse', optimizer=opt_adam,metrics=[RootMeanSquaredError(), MeanAbsoluteError()])

    return model
