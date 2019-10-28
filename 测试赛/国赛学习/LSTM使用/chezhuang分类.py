import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import warnings


def return_train_data():
    # 把数据读取出来 并且把x 和y 分离开
    col_names = ["ID", "K1K2驱动信号", "电子锁驱动信号",
                 "急停信号", "门禁信号", "THDV-M", "THDI-M", "label"]
    data = pd.read_csv("../../data/baidu/data_train.csv", names=col_names)
    data["test1"] = data["THDI-M"] * data["THDV-M"]
    data["test2"] = data["急停信号"] * data["THDV-M"]
    data["test3"] = data["THDI-M"] / data["THDV-M"]
    data["test4"] = data["THDI-M"] / (data["急停信号"] * data["THDV-M"])
    scaler = preprocessing.StandardScaler()
    data_x = data[["K1K2驱动信号", "电子锁驱动信号",
                 "急停信号", "门禁信号", "THDV-M", "THDI-M"]]
    data_y = data[["label"]].values
    data_y = np.asarray(data_y).reshape(len(data_y))  # 降维 从二维降维到一维
    return data_x, data_y


def return_test_data():
    # 预测数据的预处理
    col_names = ["ID", "K1K2驱动信号", "电子锁驱动信号",
                 "急停信号", "门禁信号", "THDV-M", "THDI-M"]
    data = pd.read_csv("../../data/baidu/data_test.csv", names=col_names)
    data["test1"] = data["THDI-M"] * data["THDV-M"]
    data["test2"] = data["急停信号"] * data["THDV-M"]
    data["test3"] = data["THDI-M"] / data["THDV-M"]
    data["test4"] = data["THDI-M"] / (data["急停信号"] * data["THDV-M"])
    scaler = preprocessing.StandardScaler()
    dataSet_X = data[["K1K2驱动信号",
                      "电子锁驱动信号", "急停信号", "门禁信号",
                      "THDV-M", "THDI-M"]]
    return dataSet_X


warnings.filterwarnings("ignore")
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_x, dataset_y = return_train_data()
# 归一化
dataset_x = scaler.fit_transform(dataset_x)
X_train, X_test, y_train, y_test = \
    train_test_split(dataset_x, dataset_y,
                     test_size=0.3, random_state=1)


X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(loss="mae", optimizer='adam')
history = model.fit(X_train, y_train, epochs=5, batch_size=72, validation_data=(X_test, y_test),
                    verbose=2, shuffle=False)
# 误差可视化
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

X_predict = return_test_data()
X_predict = scaler.fit_transform(X_predict)
X_predict = X_predict.reshape((X_predict.shape[0], 1, X_predict.shape[1]))
y_predict = model.predict(X_predict)
print(y_predict)
# with open("../../data/baidu/predict1.csv", "w") as f:
#     for i in range(len(y_predict)):
#         f.write(str(i+1)+","+str(int(y_predict[i]))+"\n")
# print("predict over!!!")
# print("y_predict=", y_predict)
