import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Reshape, Dropout
from tensorflow.keras.metrics import MeanAbsoluteError, RootMeanSquaredError
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from algos.callbacks import TimingCallback
from algos.norm import denormalize, normalize


def create_model(seq_len, learning_rate=1e-1, clipvalue=10.):
    # Adaptation of:
    # https://github.com/JackKelly/neuralnilm_prototype/blob/2119292e7d5c8a137797ad3c9abf9f37e7f749af/scripts/e567.py
    # https://github.com/OdysseasKr/neural-disaggregator/blob/master/DAE/daedisaggregator.py
    model = Sequential()

    model.add(Conv1D(8, 4, activation="linear", input_shape=(seq_len, 1), padding="same", strides=1))
    model.add(Flatten())

    model.add(Dropout(0.2))
    model.add(Dense((seq_len)*8, activation='relu'))

    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.2))
    model.add(Dense((seq_len)*8, activation='relu'))

    model.add(Dropout(0.2))

    model.add(Reshape(((seq_len), 8)))
    model.add(Conv1D(1, 4, activation="linear", padding="same", strides=1))

    optimizer = Adam(lr = learning_rate)
    model.compile(loss='mse', optimizer=optimizer,metrics=[RootMeanSquaredError(), MeanAbsoluteError()])

    return model

def get_train_data(df_train, df_val, feature, ref_norm, seq_len, seq_per_batch):
    x_train = df_train[['mains_active']].values
    x_val = df_val[['mains_active']].values
    y_train = df_train[[feature]].values
    y_val = df_val[[feature]].values

    x_train_norm = normalize(x_train, ref_norm)
    x_val_norm = normalize(x_val, ref_norm)
    y_train_norm = normalize(y_train, ref_norm)
    y_val_norm = normalize(y_val, ref_norm)

    print(x_train_norm.shape, y_train_norm.shape, x_val.shape, x_val_norm.shape)

    extra_train = seq_len - (len(x_train) % seq_len)

    x_train_norm = np.append(x_train_norm, np.zeros(extra_train))
    y_train_norm = np.append(y_train_norm, np.zeros(extra_train))

    extra_val = seq_len - (len(x_val) % seq_len)
    x_val_norm = np.append(x_val_norm,  np.zeros(extra_val))
    y_val_norm = np.append(y_val_norm,  np.zeros(extra_val))

    x_train = np.reshape(x_train_norm, (len(x_train_norm) // seq_len, seq_len, 1))
    y_train = np.reshape(y_train_norm, (len(y_train_norm) // seq_len, seq_len, 1))
    x_val = np.reshape(x_val_norm, (len(x_val_norm) // seq_len, seq_len, 1))
    y_val = np.reshape(y_val_norm, (len(y_val_norm) // seq_len, seq_len, 1))

    return x_train, y_train, x_val, y_val

def get_test_data(df_test, feature, ref_norm, seq_len, seq_per_batch):
    x_test = df_test[['mains_active']].values
    y_test = df_test[[feature]].values

    original_len = len(x_test)

    x_test = normalize(x_test, ref_norm)
    y_test = normalize(y_test, ref_norm)

    print('x_test.shape, y_test.shape')
    print(x_test.shape, y_test.shape)

    extra_test = seq_len - (len(x_test) % seq_len)

    x_test = np.append(x_test, np.zeros(extra_test))
    y_test = np.append(y_test, np.zeros(extra_test))

    x_test = np.reshape(x_test, (len(x_test) // seq_len, seq_len, 1))

    return x_test, y_test, original_len

def train(model, feature, df_train, df_val, ref_norm, seq_len, seq_per_batch, epochs, checkpoint_path):
    x_train, y_train, x_val, y_val = get_train_data(
        df_train=df_train, df_val=df_val, feature=feature, ref_norm=ref_norm, seq_len=seq_len, seq_per_batch=seq_per_batch)

    print(f'sequence_len {seq_len}')
    print(f'epochs {epochs}')
    print(f'Input initial shape {x_train.shape}; input final shape {x_train.shape}')
    print(f'Output initial shape {y_train.shape}; Output final shape {y_train.shape}')

    time_cb = TimingCallback()
    cp_cb = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 save_freq=500,
                                                 verbose=2)
    callbacks = [time_cb, cp_cb]

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=seq_per_batch, validation_data=(x_val, y_val), callbacks=callbacks, verbose=2)

    return model, history, sum(time_cb.logs)

def predict(model, feature, df_test, ref_norm, seq_len, seq_per_batch):
    x_test, y_test, original_len = get_test_data(df_test, feature=feature, ref_norm=ref_norm, seq_len=seq_len, seq_per_batch=seq_per_batch)

    y_pred = model.predict(x_test, batch_size=seq_per_batch)

    print('y_test.shape, y_pred.shape')
    print(y_test.shape, y_pred.shape)

    y_pred = np.reshape(y_pred, (y_test.shape[0]))[:original_len]
    y_test = y_test[:original_len]

    print('y_test.shape, y_pred.shape')
    print(y_test.shape, y_pred.shape)

    assert(y_test.shape == y_pred.shape)

    print('Normalized: y_test.mean, y_pred.mean')
    print(y_test.mean(), y_pred.mean())

    y_pred[y_pred < 0] = 0
    y_test = denormalize(y_test, ref_norm)
    y_pred = denormalize(y_pred, ref_norm)

    print('Denormalized: y_test.mean, y_pred.mean')
    print(y_test.mean(), y_pred.mean())

    return y_test, y_pred

def predict_from_base_model(x_test, seq_len, learning_rate, clipvalue):
    model = create_model(seq_len, learning_rate, clipvalue)

    y_pred = predict(model, x_test)

    return y_pred
