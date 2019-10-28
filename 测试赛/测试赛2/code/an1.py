#  取前面一年的月份数据 进行预测该年的月份数据

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from datetime import datetime

DATA = []
def create_dataset(dataset, look_back = 1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        dataX.append(dataset[i:(i+look_back), 0])
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


dataframe = pd.read_csv('../data/unit1.csv', usecols=[2], engine='python')
dataset = dataframe.values
print(dataset)
print(dataset[-1])
#plt.plot(dataset)
#plt.show()
# 将整型变为float
dataset = dataset.astype('float32')
chushi = np.asarray(dataset)
train_size = int(len(dataset) * 0.65)
# 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
trainlist = dataset[:train_size]
testlist = dataset[train_size:]

look_back = 1  # 步数
trainX, trainY = create_dataset(trainlist, look_back)
testX, testY = create_dataset(testlist, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
dataset_e = np.reshape(dataset, (dataset.shape[0], dataset.shape[1], 1))


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(None, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
dataset_e = dataset_e[-12][:, None]
trainPredict = model.predict(np.asarray(dataset_e))
print(scaler.inverse_transform(dataset_e[-1]))
print(trainPredict)
print(scaler.inverse_transform(trainPredict))













# a = trainPredict[-12]
# print("a= ", a)
# print(scaler.inverse_transform(a[:, None]))
# timeset = np.asarray(pd.read_csv('../data/unit1.csv', usecols=[0], engine='python'))
# year_e = []
# for a in timeset:
#     year_e.append(a)
# time_predict = np.asarray(pd.read_csv('../data/unit2.csv', usecols=[0], engine='python'))
# xss = [datetime.strptime(d, '%Y/%m/%d').date() for d in np.asarray(year_e).ravel()]
# for a in time_predict:
#     year_e.append(a)
# year_e = np.asarray(year_e).ravel()
# print(year_e)
# xs = [datetime.strptime(d, '%Y/%m/%d').date() for d in year_e]
# print(xs)
# plt.plot(xss, chushi, '.-', label="result")
# avg = np.average(scaler.inverse_transform(trainPredict))
# plt.plot(xs, scaler.inverse_transform(trainPredict), '^-', label="prediect")
# year = []
# avv = []
# for i in range(len(trainPredict)):
#     year.append(i)
#     avv.append(avg)
# plt.plot(xs, avv, '.')
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# plt.xlabel("年月份/月")
# plt.ylabel("车辆数/百万")
# plt.legend()
# plt.show()

# testPredict = model.predict(testX)
# 反归一化

# trainPredict = scaler.inverse_transform(trainPredict)
# trainY = scaler.inverse_transform(trainY[:, None])
# testPredict = scaler.inverse_transform(testPredict)
# testY = scaler.inverse_transform(testY[:, None])
#
# list1 = []
# for a in trainY:
#     list1.append(a)
# for a in testY:
#     list1.append(a)
# list2 = []
# for a in trainPredict[1:]:
#     list2.append(a)
# for a in testPredict[1:]:
#     list2.append(a)
# print(np.asarray(list1))
# print(np.asarray(list2))
# # plt.plot(trainY)
# # plt.plot(trainPredict[1:])
# # plt.show()
# # plt.plot(testY)
# # plt.plot(testPredict[1:])
# # plt.show()
# list1 = np.asarray(list1)
# list2 = np.asarray(list2)
# plt.plot(list1)
# plt.plot(list2)
# plt.show()
