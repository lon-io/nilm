import numpy as np
import matplotlib.pyplot as plt


# Plot first 10 days of device
def plot_ft_days(df, dates, house, label, n_days):
    days_series = df.loc[:dates[house][n_days]][label]
    print(days_series.shape)
    days_series_bins = days_series.values.reshape(-1, n_days)
    print(days_series_bins.shape)
    fig, axes = plt.subplots((n_days+1)//2,2, figsize=(24, n_days*2) )
    for i in range(days_series_bins.shape[-1]):
        series = days_series_bins[:,i]
        axes.flat[i].plot(series, alpha = 0.6)
        axes.flat[i].set_title(f'day {i+1}', fontsize = '15')
    plt.suptitle(f'First n_days for {label}', fontsize = '30')
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)

def plot_prediction_windows(label, y_test, y_pred, use_active = False, active_factor=2, n_samples = 32):
    y_test_mean = y_test.mean()
    n_plots_max = 16
    n_rows = n_samples
    n_plots = len(y_test) // n_rows
    y_test_bins = np.resize(y_test, (n_rows, n_plots))
    y_pred_bins = np.resize(y_pred, (n_rows, n_plots))
    fig, axes = plt.subplots((n_plots_max+1)//2,2, figsize=(24, n_plots_max*2) )
    plot_counter = 0
    for i in range(y_test_bins.shape[-1]):
        if plot_counter < n_plots_max:
            y_test_series = y_test_bins[:,i]
            y_pred_series = y_pred_bins[:,i]
            if use_active:
                window_mean = y_test_series.mean()
                if window_mean >= y_test_mean * active_factor:
                    axes.flat[plot_counter].plot(y_test_series, color = 'blue', alpha = 0.6, label = 'True value')
                    axes.flat[plot_counter].plot(y_pred_series, color = 'red', alpha = 0.6, label = 'Predicted value')
                    axes.flat[plot_counter].set_title(f'window {plot_counter+1}; mean {window_mean}', fontsize = '15')
                    plot_counter += 1
            else:
                axes.flat[plot_counter].plot(y_test_series, color = 'blue', alpha = 0.6, label = 'True value')
                axes.flat[plot_counter].plot(y_pred_series, color = 'red', alpha = 0.6, label = 'Predicted value')
                axes.flat[plot_counter].set_title(f'window {plot_counter+1}', fontsize = '15')
                plot_counter += 1
    plt.suptitle(f'Sample n_window predictions for {label}; data mean {y_test_mean} ', fontsize = '30')
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)

def plot_model_history(history):
    fig = plt.figure()
    fig.add_subplot(1,2,1)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='test loss')
    plt.legend()
    plt.grid(True)
    plt.xlabel('epoch')

    print(history.history.keys())
