import warnings

from algos.lstm import create_model, predict, train
from algos.norm import get_ref_norm
from algos.processing import split_df_by_dates
from lib import analysis
from lib.chunk_reader import get_dates, get_merged_chunks, read_labels
from lib.logging import losses_to_file, metrics_to_file
from lib.plotting import (plot_ft_days, plot_model_history,
                          plot_prediction_windows)

# Constants
MAX_NO_ROWS = 1000000
LOAD_MODEL = False
SKIP_TRAINING = False
FEATURE = 'ft_fridge'
SEQ_PER_BATCH = 1024
SEQ_LEN = 32
N_STRIDES = 1
NORM_TYPE = 'max'
LEARNING_RATE = 1e-2
CLIP_VALUE = 10.
EPOCHS = 100
CHECKPOINT_PATH = f"lstm_training_{FEATURE}/cp.ckpt"


labels = read_labels(1)
for house in range(1, 2):
    print('House {}: '.format(house), labels[house], '\n')

ref_chunk_df = get_merged_chunks(1, 1)
print('Original DF shape', ref_chunk_df.shape)
print('---')
ref_chunk_df = ref_chunk_df[0:MAX_NO_ROWS]

dates = {}
dates[1] = get_dates(ref_chunk_df, 1)

rows_per_day = ref_chunk_df.loc[:dates[1][2]].shape[0]

plot_ft_days(ref_chunk_df, dates, 1, FEATURE, 2)

# Separate house 1 data into train, validation and test data
df_train, df_val, df_test = split_df_by_dates(
    ref_chunk_df, dates=dates, house=1)

ref_norm = get_ref_norm(df_train, NORM_TYPE)

model = create_model(
    seq_len=SEQ_LEN, learning_rate=LEARNING_RATE, clipvalue=CLIP_VALUE)
model.summary()

# Load and or Train
if LOAD_MODEL:
    model.load_weights(CHECKPOINT_PATH)
if SKIP_TRAINING:
    pass
else:
    model, history, time_spent = train(model, feature=FEATURE, df_train=df_train, df_val=df_val, ref_norm=ref_norm,
                                       seq_len=SEQ_LEN, seq_per_batch=SEQ_PER_BATCH, epochs=EPOCHS, checkpoint_path=CHECKPOINT_PATH)

if not SKIP_TRAINING:
    print('Time spent', time_spent)
    plot_model_history(history)


# Predict from test split
y_test, y_pred = predict(model, feature=FEATURE, df_test=df_test,
                         ref_norm=ref_norm, seq_len=SEQ_LEN, seq_per_batch=SEQ_PER_BATCH)

# Plot results
n_samples = int(rows_per_day * 0.2)
plot_prediction_windows(FEATURE, y_test, y_pred,
                        use_active=False, n_samples=n_samples)

# Log Metrics, params and loss
params = {}
params["algo"] = 'lstm'
params["feature"] = FEATURE
params["seq_len"] = SEQ_LEN
params["seq_per_batch"] = SEQ_PER_BATCH
params["strides"] = N_STRIDES
params["epochs"] = EPOCHS
params["train_rows"] = df_train.shape[0]
params["val_rows"] = df_val.shape[0]
params["test_rows"] = df_test.shape[0]
params["norm_value"] = ref_norm
params["norm_type"] = NORM_TYPE
params["learning_rate"] = LEARNING_RATE
params["loss"] = 'adam'
params["clipvalue"] = CLIP_VALUE

print(params)

results = analysis.classification_results(y_test, y_pred, on_threshold=10)

metrics = {}
metrics["mae"] = analysis.mse_loss(y_test, y_pred)
metrics["mse"] = analysis.mae_loss(y_test, y_pred)
metrics["f1"] = analysis.f1(results)
metrics["accuracy"] = analysis.accuracy(results)
metrics["precision"] = analysis.precision(results)
metrics["recall"] = analysis.recall(results)

print(metrics)

if not SKIP_TRAINING:
    losses_to_file(1, params, history.history)

metrics_to_file(1, params, metrics)
