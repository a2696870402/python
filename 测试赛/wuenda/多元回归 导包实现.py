import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets,linear_model
from mpl_toolkits.mplot3d import Axes3D

PATH = "C:/Users/ouguangji/Desktop/machine-learning-ex1/ex1"
data = \
    pd.read_csv(PATH+'/ex1data2.txt', names=['size', 'bedrooms', 'prices'])


PATH = "C:/Users/ouguangji/Desktop/machine-learning-ex1/ex1"
data = \
    pd.read_csv(PATH+'/ex1data2.txt', names=['size', 'bedrooms', 'prices'])
X = data.iloc[:, 0:-1].values
y = data.iloc[:, -1].values
regs = linear_model.LinearRegression()
print(X)
regs.fit(X, y)
print(regs.coef_)       # b1 b2 b3 b4 b5
print(regs.intercept_)  # b0
fig = plt.figure()
ax = fig.gca(projection='3d')
x = data[['size']].values
y = data[['bedrooms']].values
z = data[['prices']].values
ax.scatter(x, y,z, color='red')
ax.plot(x, y, regs.predict(X), label='parametric curve')
plt.show()
x_pred =[[1380,3]]
y_pred = regs.predict(x_pred)
print(y_pred)