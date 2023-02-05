import pathlib

import algos as ALGOS


def get_file_path(house, params):
    algo = params.algo
    feature = params.feature

    assert algo in ALGOS.ALGOS

    logDir = 'logging';
    pathlib.Path(logDir).mkdir(parents=True, exist_ok=True)

    houseDir = f"{logDir}/house_{house}"
    pathlib.Path(houseDir).mkdir(parents=True, exist_ok=True)

    algoDir = f"{houseDir}/algo_{algo}"
    pathlib.Path(algoDir).mkdir(parents=True, exist_ok=True)

    file_path = f"{algoDir}/ft_{feature}.dat"

    return file_path

def losses_to_file(house, params, losses):
    file_path = get_file_path(house, params)

    with open(file_path, 'a') as f:
        f.write('\n\n-------\n')

        for index, loss in enumerate(losses):
            f.write(f'{index + 1}: {loss}\n')

def metrics_to_file(house, params, metrics):
    file_path = get_file_path(house, params)

    with open(file_path, 'a') as f:
        f.write('\n\n-------\n')
        f.write('*Params*\n')
        if params.algo == ALGOS.LSTM:
            f.write(f'seq_len: {params.seq_len}\n')
            f.write(f'seq_per_batch: {params.seq_per_batch}\n')
            f.write(f'strides: {params.strides}\n')

        f.write(f'epochs: {params.epochs}\n')
        f.write(f'loss: {params.loss}\n')
        f.write(f'train_rows: {params.train_rows}\n')
        f.write(f'val_rows: {params.val_rows}\n')
        f.write(f'test_rows: {params.test_rows}\n')
        f.write(f'norm_value: {params.norm_value}\n')
        f.write(f'norm_type: {params.norm_type}\n')
        f.write(f'lr: {params.learning_rate}\n')

        if params.clipvalue:
            f.write(f'clipvalue: {params.clipvalue}\n')

        f.write('*Metrics*\n')
        f.write(f'f1: {metrics.f1}\n')
        f.write(f'precision: {metrics.f1}\n')
        f.write(f'recall: {metrics.f1}\n')
        f.write(f'accuracy: {metrics.f1}\n')
        f.write(f'total_e_e: {metrics.f1}\n')
        f.write(f'mae: {metrics.f1}\n')
