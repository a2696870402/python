# coding=utf-8
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from pandas import datetime
from pandas import Series
import pandas as pd


# load data
def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')


series = read_csv('../data/baidu/LSTM_data/data_dan.csv', header=0, parse_dates=[0], index_col=0, squeeze=True,
                  date_parser=parser)
'''                     
将数据转换成有监督数据  
即包含input output      
训练的目的就是找到训练数据input和output的关系                                                                           
此处的input是t时间步的数据，output为t+1时间步的数据                                                                     
具体实现就是将整体的时间数据向后滑动一格，和原始数据拼接，就是有监督的数据                                              
'''


def timeseries_to_supervised(data, lag=1):  # lag表示的是当前的值只与历史lag个时间步长的值有关，也就是用lag个数据预测下一个
    df = DataFrame(data)
    colums = [df.shift(i) for i in range(1, lag + 1)]  # 原始数据时间窗向后移动lag步长
    colums.append(df)  # 拼接数据
    df = concat(colums, axis=1)  # 横向拼接重塑数据，格式:input putput
    df.fillna(0, inplace=True)  # 由于数据整体向后滑动lag后，前面的lag个数据是Na形式，用0来填充
    return df


X = series.values
supervised = timeseries_to_supervised(X, 1)
series = read_csv('../data/baidu/LSTM_data/data_dan.csv', header=0, parse_dates=[0], index_col=0, squeeze=True,
                  date_parser=parser)


# 做差分，去趋势，获得差分序列
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]  # 当前时间步t的值减去时间步t-interval的值
        diff.append(value)
    return Series(diff)


# 将预测值进行逆处理，得到真实的销售预测
def inverse_difference(history, yhat, interval=1):  # 历史数据，预测数据，差分间隔
    return yhat + history[-interval]


# 数据处理

# 将数据转换成稳定的
differenced = difference(series, 1)
# 逆处理，从差分逆转得到真实值
inverted = list()
for i in range(len(differenced)):
    value = inverse_difference(series, differenced[i], len(series) - i)
    inverted.append(value)
inverted = Series(inverted)
