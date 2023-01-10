import pandas as pd

appliances = {
    'CoffeeMachine': 'CoffeeMachine',
    'FridgeFreezer': 'FridgeFreezer',
    'Freezer': 'Freezer',
    'HandMixer': 'HandMixer',
    'HairDryerStraightener': 'HairDryerStraightener',
    'Kettle': 'Kettle',
    'MacBook2007': 'MacBook2007',
    'MacBookPro2011': 'MacBookPro2011',
    'Microwave': 'Microwave',
    'StoveOven': 'Microwave',
    'TVPhillips': 'TVPhillips',
    'TVSharp': 'TVSharp',
    'TVGrundig': 'TVGrundig',
    'TVSamsung': 'TVSamsung',
    'TVLG': 'TVLG',
    'Toaster': 'Toaster',
    'VacuumCleaner': 'VacuumCleaner',
}

applianceFileNames = {
    'CoffeeMachine': '1_CoffeeMachine',
    'FridgeFreezer': '2_Fridge-Freezer',
    'Freezer': '3_Freezer',
    'HandMixer': '4_HandMixer',
    'HairDryerStraightener': '5_HairDryer-Straightener',
    'Kettle': '6_Kettle',
    'MacBook2007': '7_MacBook2007',
    'MacBookPro2011': '8_MacBookPro2011-1',
    'Microwave': '10_Microwave',
    'StoveOven': '11_Stove-Oven',
    'TVPhillips': '12_TV-Philips',
    'TVSharp': '13_TV-Sharp',
    'TVGrundig': '14_TV-Grundig',
    'TVSamsung': '15_TV-Samsung',
    'TVLG': '16_TV-LG',
    'Toaster': '17_Toaster',
    'VacuumCleaner': '18_VaccumCleaner',
}

sample = {}
chunksize = 10000
#read data in chunks of chunksize rows at a time
for k, v in applianceFileNames.items():
    chunk = pd.read_csv(f'data/appliances/{v}.csv',chunksize=chunksize)
    pd_df = pd.concat(chunk)
    print(k)
    print('---')
    print(pd_df.head())
    print()
