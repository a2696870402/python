import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import warnings


def return_train_data():
    # 训练数据的处理
    col_names = ["ID", "K1K2驱动信号", "电子锁驱动信号",
                 "急停信号", "门禁信号", "THDV-M", "THDI-M", "label"]
    data = pd.read_csv("../data/baidu/data_train.csv", names=col_names)
    # print(data.info())
    data["test1"] = data["THDI-M"]*data["THDV-M"]
    data["test2"] = data["急停信号"]*data["THDV-M"]
    data["test3"] = data["THDI-M"]/data["THDV-M"]
    data["test4"] = data["THDI-M"]/(data["急停信号"]*data["THDV-M"])
    scaler = preprocessing.StandardScaler()
    dataSet_X = data[["K1K2驱动信号", "电子锁驱动信号",
                 "急停信号", "门禁信号", "THDV-M", "THDI-M"]]
    dataSet_Y = data[["label"]].as_matrix()
    dataSet_Y = np.asarray(dataSet_Y).reshape(len(dataSet_Y))
    # print(dataSet_X)
    # print(dataSet_Y)
    return dataSet_X, dataSet_Y



'''
模型训练和预测
'''

warnings.filterwarnings("ignore")
import time
dataset_X, dataset_Y = return_train_data()
X_train, X_test, y_train, y_test = \
    train_test_split(dataset_X, dataset_Y,
                     test_size=0.2, random_state=21)
clf = GradientBoostingClassifier(
    learning_rate=0.1, n_estimators=1500, min_samples_split=350,
    min_samples_leaf=20, max_depth=8, max_features="auto", subsample=0.8
    , random_state=0)
clf.fit(X_train, y_train) # 模型训练
# 模型保存
joblib.dump(clf, 'train_model_result.m')

acc_train = clf.score(X_train, y_train)
acc_test = clf.score(X_test, y_test)

print("acc_train = ", acc_train)
print("acc_test = ", acc_test)

