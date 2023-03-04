import math


def split_df_by_dates(df, dates, house, split_points = (0.5, 0.8)):
    test_split_point, val_split_point = split_points;

    assert(test_split_point >= 0.5 and test_split_point < 1)
    assert(val_split_point >= (test_split_point + 0.1) and val_split_point <= (1 - 0.1))

    # Separate house 1 data into train, validation and test data
    n_days = len(dates[house])
    train_index = math.ceil(n_days *test_split_point)
    test_index = math.ceil(n_days *val_split_point)
    df_train = df.loc[:dates[house][train_index]]
    df_val = df.loc[dates[house][train_index]:dates[house][test_index]]
    df_test = df.loc[dates[house][test_index]:]
    print('df_train.shape: ', df_train.shape)
    print('df_val.shape: ', df_val.shape)
    print('df_test.shape: ', df_test.shape)

    return df_train, df_val, df_test
