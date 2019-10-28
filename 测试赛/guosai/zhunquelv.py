import warnings
import pandas as pd
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib


model = joblib.load('train_luoji.m')
f = open('data/1.csv')
data = pd.read_csv(f)
X_predict = data[['乘客到达/出发数量(万人次)','距离市中心的距离(km)','出租车数量(万辆)','城市人口数量(万人)',
             '出租车价格(/km)','交通等级']]
X = X_predict.as_matrix()
y_predict = model.predict(X)
print(model.coef_)
print(y_predict)
se = pd.DataFrame(y_predict)
# se.to_csv("se.csv")
