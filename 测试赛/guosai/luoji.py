import warnings
import pandas as pd
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
warnings.filterwarnings('ignore')


f = open('data/data1.csv', encoding='GB2312')
data = pd.read_csv(f)

data = data[['乘客到达/出发数量(千万人次)','距离市中心的距离(km)','出租车数量(万辆)','城市人口数量(万人)',
             '出租车价格(/km)','交通等级','选择']]
X = data[['乘客到达/出发数量(千万人次)','距离市中心的距离(km)','出租车数量(万辆)','城市人口数量(万人)',
             '出租车价格(/km)','交通等级']]
Y = data[['选择']]
X = X.as_matrix()
Y = Y.as_matrix()
train_X, test_X, train_y, test_y = train_test_split(X, Y, train_size=0.8)
modelLR = LogisticRegression(solver='liblinear')
modelLR.fit(train_X, train_y)
score = modelLR.score(test_X, test_y)
print(modelLR.coef_)
print(modelLR.intercept_)
print(score)
joblib.dump(modelLR, 'train_luoji.m')
# modelLR2 = joblib.load('train_luoji.m')
# print(modelLR2.score(test_X,test_y))
# print(modelLR2.intercept_)
# print(modelLR2.coef_)
# if score > modelLR2.score(test_X, test_y):
#     joblib.dump(modelLR, 'train_luoji.m')
#     print(score)

