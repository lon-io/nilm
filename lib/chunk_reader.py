import pandas as pd
import glob

def get_house_path(house):
    return f'data/ukdale-parsed-chunks/house_{house}/'

def get_chunk_path(house, chunk):
    return get_house_path(house) + f'chunk_{chunk}.dat'

def get_num_chunks(house):
    return len(glob.glob(get_house_path(house) + 'chunk_*[0-9].dat'))

def read_file(house, chunk, labels):
    file = get_chunk_path(house, chunk)
    print(f'reading file {file}; for house {house} and chunk {chunk}');

    dtypes = {}
    for label in labels[house]:
        dtypes[label] = 'float32'
    df = pd.read_table(file, sep = '\t', header=0, names = labels[house],
                                       dtype = dtypes)

    return df

def parse_data(df):
    df['timestamp'] = df['unix_time'].astype("datetime64[s]")
    df.set_index(df['timestamp'].values, inplace=True)
    df.drop(['unix_time'], axis=1, inplace=True)

    df['timeslice'] = df.timestamp.dt.hour

    return df

def read_labels():
    labels = {}
    for house in range(1, 2):
        fileName = get_chunk_path(house, 1)
        house_labels = pd.read_csv(fileName, sep = '\t', nrows=1).columns.tolist()
        labels[house] = house_labels
    return labels

def get_house_data_generator(house):
    labels = read_labels()
    num_chunks = get_num_chunks(house)
    for i in range(1, num_chunks + 1):
        if int(i) == 1:
            print(f'reading house {house}; chunk 1');

        df = read_file(house, i, labels)
        df = parse_data(df)

        print(f'read house {house}; chunk {i}; df.shape is {df.shape}')

        yield df

def get_merged_chunks(house, num_chunks):
    max_num_chunks = get_num_chunks(house)
    num_chunks = max_num_chunks if num_chunks > max_num_chunks else num_chunks
    house_gen = get_house_data_generator(house)
    data = [next(house_gen) for i in range(num_chunks)]

    return pd.concat(data)

def get_all_data_generators():
    gen = {}
    for house in range(1,2):
        gen[house] = get_house_data_generator(house)

    return gen

def get_dates(house_df, house):
    dates = [str(time)[:10] for time in house_df.index.values]
    dates = sorted(list(set(dates)))
    print('House {0} data contain {1} days from {2} to {3}.'.format(house,len(dates),dates[0], dates[-1]))
    print(dates, '\n')

    return dates
