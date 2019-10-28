import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATH = "C:/Users/ouguangji/Desktop/machine-learning-ex1/ex1"
data = \
    pd.read_csv(PATH+'/ex1data2.txt', names=['size', 'bedrooms', 'prices'])
print(data.head())
# 数据标准化

data = (data-data.mean())/data.std()
# 对每一列求均值 标准差
# print(data.head())

data.insert(0, 'ones', 1) # 在最左边添加列名叫ones的列
# 提取特征
X = data.iloc[:, 0:-1].values
y = data.iloc[:, -1].values

# 损失函数：
def cost_function(X, y, theta):
    inner = np.power(np.dot(X, theta)-y, 2)
    return np.sum(inner)/(2*len(X))
# 梯度下降函数：
def tD(X, y, theta, al, iters, is_plot=False):
    costs = []
    for i in range(iters):
        theta = theta-al/len(X)*(np.dot(X.T, np.dot(X, theta)-y))
        cost = cost_function(X, y, theta)
        costs.append(cost)

        if i%100 ==0:
            if is_plot:
                print(cost)
    return theta, costs

# 准备使用不同的学习率
n_al = 0.03
iters = 2000
theta = np.zeros((3, 1))
fig ,ax = plt.subplots()
theta_final, costs = tD(X, y, theta, n_al, iters)

