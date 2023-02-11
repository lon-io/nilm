import pathlib

import algos as ALGOS


def get_file_path(house, params, is_loss=False):
    algo = params['algo']
    feature = params['feature']

    assert algo in ALGOS.ALGOS

    logDir = 'logs';
    pathlib.Path(logDir).mkdir(parents=True, exist_ok=True)

    houseDir = f"{logDir}/house_{house}"
    pathlib.Path(houseDir).mkdir(parents=True, exist_ok=True)

    algoDir = f"{houseDir}/algo_{algo}"
    pathlib.Path(algoDir).mkdir(parents=True, exist_ok=True)

    if is_loss:
        file_path = f"{algoDir}/loss_{feature}.dat"
    else:
        file_path = f"{algoDir}/{feature}.dat"

    return file_path


def losses_to_file(house, params, losses):
    file_path = get_file_path(house, params, is_loss=True)

    train_loss = losses['loss']
    val_loss = losses['val_loss']

    with open(file_path, 'a') as f:
        f.write('\n\n-------\n')

        for index, loss in enumerate(zip(train_loss, val_loss)):
            train, val = loss
            f.write(f'{index + 1}: {train}, {val}\n')


def metrics_to_file(house, params, metrics):
    file_path = get_file_path(house, params)

    with open(file_path, 'a') as f:
        f.write('\n\n-------\n')
        f.write('*Params*\n')
        if params["algo"] in [ALGOS.LSTM, ALGOS.DAE]:
            f.write(f'seq_len: {params["seq_len"]}\n')
            f.write(f'seq_per_batch: {params["seq_per_batch"]}\n')
            f.write(f'strides: {params["strides"]}\n')

        f.write(f'epochs: {params["epochs"]}\n')
        f.write(f'loss: {params["loss"]}\n')
        f.write(f'train_rows: {params["train_rows"]}\n')
        f.write(f'val_rows: {params["val_rows"]}\n')
        f.write(f'test_rows: {params["test_rows"]}\n')
        f.write(f'norm_value: {params["norm_value"]}\n')
        f.write(f'norm_type: {params["norm_type"]}\n')
        f.write(f'lr: {params["learning_rate"]}\n')

        if 'clipvalue' in params:
            f.write(f'clipvalue: {params["clipvalue"]}\n')

        f.write('*Metrics*\n')
        f.write(f'f1: {metrics["f1"]}\n')
        f.write(f'precision: {metrics["f1"]}\n')
        f.write(f'recall: {metrics["f1"]}\n')
        f.write(f'accuracy: {metrics["f1"]}\n')
        f.write(f'total_e_e: {metrics["f1"]}\n')
        f.write(f'mae: {metrics["f1"]}\n')


if __name__ == '__main__':
    params = {'algo': 'lstm', 'feature': 'ft_kettle', 'seq_len': 32, 'seq_per_batch': 1024, 'strides': 1, 'epochs': 40, 'train_rows': 521300,
        'val_rows': 316800, 'test_rows': 190700, 'norm_value': 6002.92, 'norm_type': 'max', 'learning_rate': 0.001, 'loss': 'adam', 'clipvalue': 10.0}
    metrics = {'mae': 4699.804996695011, 'mse': 4.816044563871674, 'f1': 0.3269096005606167, 'accuracy': 0.9799360795454546, 'precision': 0.19758576874205844, 'recall': 0.9462474645030426}


    metrics_to_file(1, params, metrics)
