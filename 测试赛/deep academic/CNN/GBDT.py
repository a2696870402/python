import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import warnings
import time
from sklearn import datasets
from sklearn import svm


standarscaler = StandardScaler()


def return_train_data():
    # 训练数据的处理
    data = pd.read_csv("../../ccf/data/first_round_training_data.csv")
    data_X = data[['Parameter1', 'Parameter2', 'Parameter3',
                   'Parameter4', 'Parameter5', 'Parameter6',
                   'Parameter7', 'Parameter8', 'Parameter9', 'Parameter10']]
    data_y = data[['Quality_label']].values
    data_y = list(data_y)
    for i in range(len(data_y)):
        if data_y[i] == 'Fail':
            data_y[i] = '1'
        elif data_y[i] == 'Pass':
            data_y[i] = '2'
        elif data_y[i] == 'Good':
            data_y[i] = '3'
        elif data_y[i] == 'Excellent':
            data_y[i] = '4'
    data_y = np.asarray(data_y).astype(np.float64)[:, None]
    data_y = np.asarray(data_y).reshape((len(data_y)))
    data_X = standarscaler.fit_transform(data_X)
    # data_y = standarscaler.fit_transform(data_y)
    return data_X, data_y


warnings.filterwarnings("ignore")
dataset_X, dataset_Y = return_train_data()
X_train, X_test, y_train, y_test = \
    train_test_split(dataset_X, dataset_Y,
                     test_size=0.31, random_state=21)
clf = GradientBoostingClassifier(
    learning_rate=0.1, n_estimators=1500, min_samples_split=300,
    min_samples_leaf=20, max_depth=12, max_features="auto", subsample=0.8
    , random_state=0, verbose=1)
clf.fit(X_train, y_train)  # 模型训练

# 模型保存


joblib.dump(clf, 'train_model_result.m')

acc_train = clf.score(X_train, y_train)
acc_test = clf.score(X_test, y_test)

print("acc_train = ", acc_train)
print("acc_test = ", acc_test)
