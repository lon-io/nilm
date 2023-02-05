import pandas as pd

datapointFilenames = ['1475708700932', '1477227096132', '1477592018787', '1478884263362', '1482282276343', '1483205843836']

sample = {}
chunksize = 10000
#read data in chunks of chunksize rows at a time
for v in datapointFilenames:
    chunk = pd.read_csv(f'data/V_I_P_Q/1Hz/{v}.csv',chunksize=chunksize)
    pd_df = pd.concat(chunk)
    print('---')
    print(pd_df.head())
    print()

    # sample[k] = pd_df

# chunk = pd.read_csv(f"data/appliances/{applianceFileNames['Freezer']}.csv",chunksize=chunksize)
# pd_df = pd.concat(chunk)
# print(pd_df.head())
# print(sample[appliances['Freezer']])
