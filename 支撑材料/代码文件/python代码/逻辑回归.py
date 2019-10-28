# coding: utf-8

import numpy as np
import math
from sklearn import datasets
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
import pandas as pd

infinity = float(-2 ** 31)
'''
2018-8-5 
逻辑回归的实现
'''

data1 = open('first_data.csv')
data2 = pd.read_csv(data1)

X = data2[['人口', 'GDP', '垃圾处理厂数量', '城市化率', '污水排放量', '食品制造业固体废物产生量（万吨）']]
y = data2[['垃圾总产量']]
print(X)
print(y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.5)

print(X_train)
print(X_test)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
print('intercept_:%.3f' % clf.intercept_)
print(clf.coef_)
# print('Mean squared error: %.3f' % mean_squared_error(y_test, clf.predict(X_test)))
print('Variance score: %.3f' % r2_score(y_test, clf.predict(X_test)))
print('score: %.3f' % clf.score(X_test, y_test))

# 加入进来
# 预测:
data3 = open('first_test_data.csv')
data4 = pd.read_csv(data3)
X_predict = data4[['人口', 'GDP', '垃圾处理厂数量', '城市化率', '污水排放量', '食品制造业固体废物产生量（万吨）']]
Y_predict = clf.predict(X_predict)
print(Y_predict)
year1 = []
y1 = np.asarray(y)
year2 = []
y2 = np.asarray(Y_predict)
for i in range(1996, 2019):
    year1.append(i)
for i in range(2019, 2039):
    year2.append(i)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
print(year1)
print(year2)




