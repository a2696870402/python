from datetime import datetime
import pandas as pd


# load data
dataparse = lambda dates: pd.datetime.strptime(dates, '%Y %m %d %H')
dataset = pd.read_csv('../data/baidu/LSTM_data/PRSA_data.csv', parse_dates=[['year', 'month', 'day', 'hour']],index_col=0,
                      date_parser=dataparse)
# print(dataset)
dataset.drop('No', axis=1, inplace=True)
print(dataset)
# manually specify column names
dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.index.name = 'date'
# mark all NA values with 0
dataset['pollution'].fillna(0, inplace=True)
# drop the first 24 hours
dataset = dataset[24:]

print(dataset.head(5))
dataset.to_csv('../data/baidu/LSTM_data/pollution.csv')



