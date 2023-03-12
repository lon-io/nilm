import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional, Conv1D, Dense
from tensorflow.keras.metrics import MeanAbsoluteError, RootMeanSquaredError
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from algos.callbacks import TimingCallback
from algos.data_gen import Seq2PointDataGenerator
from algos.norm import denormalize, normalize


def create_model(seq_len, learning_rate, clipvalue):
    # Paper - https://dl.acm.org/doi/10.1145/2821650.2821672
    # Adapted from - https://github.com/OdysseasKr/neural-disaggregator/blob/master/RNN/rnndisaggregator.py#L348
    model = Sequential()

    model.add(Conv1D(16, 4, activation="linear", input_shape=(seq_len, 1), padding="same", strides=1))
    model.add(Bidirectional(LSTM(128, return_sequences=True, stateful=False), merge_mode='concat'))
    model.add(Bidirectional(LSTM(256, return_sequences=False, stateful=False), merge_mode='concat'))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(1, activation='linear'))

    opt_adam = Adam(lr = learning_rate, clipvalue=clipvalue)
    model.compile(loss='mse', optimizer=opt_adam,metrics=[RootMeanSquaredError(), MeanAbsoluteError()])

    return model

def get_train_data(df_train, df_val, feature, ref_norm, seq_len, seq_per_batch):
    x_train = df_train[['mains_active']].values
    x_val = df_val[['mains_active']].values
    y_train = df_train[[feature]].values
    y_val = df_val[[feature]].values

    x_train = normalize(x_train, ref_norm)
    x_val = normalize(x_val, ref_norm)
    y_train = normalize(y_train, ref_norm)
    y_val = normalize(y_val, ref_norm)

    print('x_train.shape, y_train.shape, x_val.shape, x_val.shape')
    print(x_train.shape, y_train.shape, x_val.shape, x_val.shape)

    train_data_generator = Seq2PointDataGenerator(x_train, y_train, seq_per_batch=seq_per_batch, seq_len=seq_len)
    val_data_generator = Seq2PointDataGenerator(x_val, y_val, seq_per_batch=seq_per_batch, seq_len=seq_len)

    x_train, y_train = train_data_generator.load_all()
    x_val, y_val = val_data_generator.load_all()

    return x_train, y_train, x_val, y_val

def get_test_data(df_test, feature, ref_norm, seq_len, seq_per_batch):
    x_test = df_test[['mains_active']].values
    y_test = df_test[[feature]].values

    x_test = normalize(x_test, ref_norm)
    y_test = normalize(y_test, ref_norm)

    print('x_test.shape, y_test.shape')
    print(x_test.shape, y_test.shape)

    test_data_generator = Seq2PointDataGenerator(x_test, y_test, seq_per_batch=seq_per_batch, seq_len=seq_len)
    x_test, y_test = test_data_generator.load_all()

    return x_test, y_test

def train(model, feature, df_train, df_val, ref_norm, seq_len, seq_per_batch, epochs, checkpoint_path):
    x_train, y_train, x_val, y_val = get_train_data(
        df_train=df_train, df_val=df_val, feature=feature, ref_norm=ref_norm, seq_len=seq_len, seq_per_batch=seq_per_batch)

    time_cb = TimingCallback()
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
    callbacks = [time_cb, cp_callback]

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=seq_per_batch, validation_data=(x_val, y_val), callbacks=callbacks)

    return model, history, sum(time_cb.logs)

def predict(model, feature, df_test, ref_norm, seq_len, seq_per_batch):
    x_test, y_test = get_test_data(df_test, feature=feature, ref_norm=ref_norm, seq_len=seq_len, seq_per_batch=seq_per_batch)

    y_pred = model.predict(x_test, batch_size=seq_per_batch)

    print('y_test.shape, y_pred.shape')
    print(y_test.shape, y_pred.shape)

    y_pred = y_pred.reshape(y_pred.shape[0])

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
