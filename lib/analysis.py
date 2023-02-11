import numpy as np
from IPython.display import display

def print_heads_tails(df, h=True,t=True):
    print(f'House data has shape: {df.shape}')
    if h:
        display(df.head(2))

    print()

    if t:
        display(df.tail(2))

def assert_lengths(y_test, y_pred):
    assert(len(y_test) == len(y_pred))

def mse_loss(y_test, y_pred):
    assert_lengths(y_test, y_pred)
    acc = 0

    for i in range(len(y_test)):
        acc += (y_test[i] - y_pred[i]) ** 2

    return acc/len(y_test)

def mae_loss(y_test, y_pred):
    assert_lengths(y_test, y_pred)
    acc = 0

    for i in range(len(y_test)):
        acc += abs(y_test[i] - y_pred[i])

    return acc/len(y_test)

def classification_results(y_test, y_pred, on_threshold = 10):
    assert_lengths(y_test, y_pred)

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(len(y_test)):
        if y_test[i] > on_threshold:
            if y_pred[i] > on_threshold:
                tp += 1
            else:
                fn += 1
        else:
            if y_pred[i] > on_threshold:
                fp += 1
            else:
                tn += 1

    return tp, tn, fp, fn

def f1(classification_results):
    p = precision(classification_results)
    r = recall(classification_results)

    return 2 * (p * r) / (p + r)

def precision(classification_results):
    tp, _, fp, _ = classification_results

    return tp / (tp + fp)

def recall(classification_results):
    tp, _, _, fn = classification_results

    return tp / (tp + fn)

def accuracy(classification_results):
    tp, tn, fp, fn = classification_results

    return (tp + tn) / (tp + tn + fp + fn)
