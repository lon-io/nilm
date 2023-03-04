import numpy as np


def get_ref_norm(df_train, norm_type):
    ref_mean = df_train[['mains_active']].values.mean()
    ref_max = df_train[['mains_active']].values.max()
    ref_std = df_train[['mains_active']].values.std()

    print(f'ref_mean: {ref_mean}; ref_max: {ref_max}; ref_std: {ref_std}')

    if norm_type == 'mean':
        ref_norm = ref_mean
    elif norm_type == 'std':
        ref_norm = ref_std
    else:
        ref_norm = ref_max

    print(f'norm_type: {norm_type}; ref_norm {ref_norm}')
    return ref_norm

def normalize(x, ref_norm):
    return x / ref_norm


def denormalize(x, ref_norm):
    return x * ref_norm

def reshape_x(x):
    return np.reshape(x, (x.shape[0], 1, 1))
