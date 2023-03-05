import math
import warnings

import pandas as pd
from IPython.display import display

warnings.filterwarnings("ignore")
import glob
import math
import pathlib


def print_heads_tails(df, h=True,t=True):
    print(f'Data has shape: {df.shape}')
    if h:
        display(df.head(2))

    print('...')

    if t:
        display(df.tail(2))


def read_labels():
    labels = {}
    for house in range(1, 6):
        file_path = f'data/ukdale/house_{house}/labels.dat'
        labels[house] = {}
        with open(file_path) as f:
            for line in f:
                split_line = line.split(' ')
                channel = int(split_line[0])
                device_name = split_line[1].strip()
                prefix = 'ft_'

                if device_name == 'aggregate':
                    prefix = ''
                    device_name = 'aggregate_apparent'

                labels[house][channel] = prefix + device_name
        labels[house]['mains'] = 'mains'
    return labels
file_labels = read_labels()

def get_house_path(house):
    return f'data/ukdale/house_{house}/'

def get_chan_path(house, channel):
    if channel == 'mains':
        return get_house_path(house) + f'mains.dat'

    return get_house_path(house) + f'channel_{channel}.dat'

def get_num_apps(house):
    return len(glob.glob(get_house_path(house) + 'channel_*[0-9].dat'))

def read_file(house, channel):
    print(f'reading house {house}; channel {channel}');
    file = get_chan_path(house, channel)

    df = pd.read_table(file, sep = ' ', names = ['unix_time', file_labels[house][channel]],
                                       dtype = {'unix_time': 'int64', file_labels[house][channel]:'float64'})

    return df

def read_mains_file(house):

    file = get_chan_path(house, 'mains')

    df = pd.read_table(file, sep = ' ', names = ['unix_time', 'mains_active', 'mains_apparent', 'mains_rms' ],
                                       dtype = {'unix_time': 'float64', 'mains_active':'float64',
                                                'mains_apparent': 'float64', 'mains_rms': 'float64'})

    return df

def parse_data(df, sort_index = True, drop_duplicates = True):
    df['timestamp'] = df['unix_time'].astype("datetime64[s]")
    df.set_index(df['timestamp'].values, inplace=True)
    df.drop(['unix_time'], axis=1, inplace=True)

    if sort_index:
        df = df.sort_index()

    if drop_duplicates:
        dups_in_index = df.index.duplicated(keep='first')
        if dups_in_index.any():
            df = df[~dups_in_index]

    return df

def get_timeframe(df):
    start = df.index[0]
    end = df.index[-1]

    return start, end

def get_feature_columns(df):
    return list(filter(lambda x: x.startswith('ft_'), df.columns.tolist()))

def sub_sample_mains(df):
    print('Before sub sampling')
    print_heads_tails(df)
    df = df.resample('6S').max()
    print('After sub sampling')
    print_heads_tails(df)

    return df

def fill_na(df, dry_run = False):
    for label in df.columns:
        null_count = df[label].isnull().sum()
        zero_count = df[label].isin([0]).sum()
        print(f'[fill_na] checked NaN count for {label}; result is {null_count}')
        print(f'[fill_na] checked zero count for {label}; result is {zero_count}')
        if not dry_run and null_count > 0:
            df[label].interpolate(method='linear', inplace=True)
            df[label].fillna(0.0, inplace=True)
            print(f'[fill_na] post filling - NaN count for {label}; result is {null_count}')
            print(f'[fill_na] post filling - checked zero count for {label}; result is {zero_count}')

DATA_SIZE_LIMIT = 17000000

def read_mains(house):
    df = read_mains_file(house)
    df = parse_data(df)

    return df

mains_df = {}
for house in range(1,2):
    mains_df[house] = read_mains(house)
    fill_na(mains_df[house], dry_run=True)

for house in range(1,2):
    mains_df[house] = sub_sample_mains(mains_df[house])
    fill_na(mains_df[house], dry_run=True)


for house in range(1,2):
    fill_na(mains_df[house])
    mains_df[house]['timestamp'] = mains_df[house].index
    print('')
    fill_na(mains_df[house], dry_run=True)

def read_merge_data(house):
    df = mains_df[house]
    print(f'read house {house}; mains; df.shape is {df.shape}')

    num_apps = get_num_apps(house)
    for i in range(1, num_apps + 1):
        data = read_file(house, i)
        if data.shape[0] >= DATA_SIZE_LIMIT:
            print(f'read house {house}; channel {i}; df.shape is {df.shape}; data.shape is {data.shape}')
            data = parse_data(data)

            start_x, end_x = get_timeframe(data)
            start_y, end_y = get_timeframe(df)

            start = start_x if start_x > start_y else start_y
            end = end_x if end_x < end_y else end_y

            data = data[(data.index >= start) &
                                   (data.index <= end)]
            df = df[(df.index >= start) &
                                   (df.index <= end)]

            df = pd.merge_asof(df, data, on = 'timestamp', tolerance=pd.Timedelta('6s'))
            df.set_index(df['timestamp'].values, inplace=True)
        else:
            print(f'skipping house {house}; channel {i}; df.shape is {df.shape}; data.shape is {data.shape}')

    return df

# Read and merge data
df = {}
for i in range(1,2):
    df[i] = read_merge_data(i)

# Fill NaN
for house in range(1,2):
    fill_na(df[house])


# Convert real to apparent power
power_factors = {}
apparent_labels = {
    1: ['aggregate_apparent', 'ft_boiler', 'ft_solar_thermal_pump', 'ft_kitchen_lights', 'ft_lighting_circuit']
}
for house in range(1,2):
    mains_active_sample = df[house]['mains_active'].iloc[0]
    mains_apparent_sample = df[house]['mains_apparent'].iloc[0]
    power_factors[house] = mains_active_sample / mains_apparent_sample
    for apparent_label in apparent_labels[house]:
        if apparent_label in df[house].columns:
            print(f'Converting apparent to real for {apparent_label}')
            if apparent_label == 'aggregate_apparent':
                df[house]['aggregate_active'] = df[house][apparent_label] * power_factors[house]
            else:
                df[house][apparent_label] = df[house][apparent_label] * power_factors[house]
        else:
            print(f'Cannot convert apparent to real for {apparent_label}, since it is not present in df')


# Drop Low power appliances
averages = {}
def get_devices_mean_power(df):
    for house in range(1,2):
        averages[house] = {}
        num_apps = get_num_apps(i)
        for label in get_feature_columns(df[house]):
            mean = df[house][label].mean()
            averages[house][label] = mean

get_devices_mean_power(df)

# Re-add unix timestamp
def re_add_unix_time_feature(df, inplace=True):
    print('re-adding unix timestamp')
    df['unix_time'] = df.timestamp.map(pd.Timestamp.timestamp)
    df.set_index(df['unix_time'].values, inplace=True)
    df.drop(['timestamp'], axis=1, inplace=True)


# Dump to csv

newChunkDataDir = 'data/ukdale-parsed-chunks';
pathlib.Path(newChunkDataDir).mkdir(parents=True, exist_ok=True)

for house in range(1,2):
    houseDir = f"{newChunkDataDir}/house_{house}"
    pathlib.Path(houseDir).mkdir(parents=True, exist_ok=True)

def df_chunk_to_csv(house, chunk, df):
    file_name = f"{newChunkDataDir}/house_{house}/chunk_{chunk}.dat"
    df.to_csv(file_name, sep='\t', header=True, index=False)

def save_by_chunks(df):
    file_row_size = 4500000
    for house in range(1,2):
        df_row_size = df[house].shape[0]
        no_of_chunks = math.ceil(df_row_size / file_row_size)
        print(f'Total row size - {df_row_size}; Number of chunks - {no_of_chunks}')
        drop_low_power_devices(df[house], get_feature_columns(df[house]), averages[house])
        re_add_unix_time_feature(df[house], inplace=False)
        for n in range(no_of_chunks):
            start_index = n * file_row_size
            end_index = start_index + file_row_size
            end_index = end_index if end_index <= df_row_size else df_row_size

            split_df = df[house].iloc[start_index:end_index]
            print(f'saving chunk - {n+1}, with row range - {start_index} to {end_index} and shape - {split_df.shape}')

            df_chunk_to_csv(house, n+1, split_df)

save_by_chunks(df)
