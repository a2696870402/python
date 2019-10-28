#
# 预测版块
#

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn import metrics


def return_test_data():
    # 预测数据的预处理
    col_names = ["ID", "K1K2驱动信号", "电子锁驱动信号",
                 "急停信号", "门禁信号", "THDV-M", "THDI-M"]
    data = pd.read_csv("../data/baidu/data_test.csv", names=col_names)
    data["test1"] = data["THDI-M"] * data["THDV-M"]
    data["test2"] = data["急停信号"] * data["THDV-M"]
    data["test3"] = data["THDI-M"] / data["THDV-M"]
    data["test4"] = data["THDI-M"] / (data["急停信号"] * data["THDV-M"])
    scaler = preprocessing.StandardScaler()
    dataSet_X = data[["K1K2驱动信号",
                      "电子锁驱动信号", "急停信号", "门禁信号",
                      "THDV-M", "THDI-M"]]
    return dataSet_X


clf = joblib.load('train_model_result.m')
x_test_data = return_test_data()
y_test_data = clf.predict(x_test_data)
print(y_test_data)
with open("../data/baidu/yuce.csv", "w") as f:
    for i in range(len(y_test_data)):
        f.write(str(i+1)+","+str(int(y_test_data[i]))+"\n")
print("predict over!!!")
print("y_predict=", y_test_data)

