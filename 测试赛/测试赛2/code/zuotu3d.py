import  matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  #绘制三D图形
import pandas as pd
from  datetime import  datetime

fig = plt.figure()
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.family']='sans-serif'
ax = Axes3D(fig)
f = open("../data/unit1.csv")
data = pd.read_csv(f)
z = np.asarray(data[['year']])
x = np.asarray(data[['x2']])
y = np.asarray(data[['y2']])
z = [datetime.strptime(d, '%Y/%m/%d').date() for d in z.ravel()]
z = []
for i in range(len(data)):
    z.append(i)
print(x)
print(y)
print(z)
# ax.scatter(y, z, x)
ax.scatter(y[0:18], z[0:18], x[0:18], c='y')  # 绘制数据点
ax.scatter(y[19:44], z[19:44], x[19:44], c='r')
ax.scatter(y[45:], z[45:], x[45:], c='g')
# ax3d.plot_surface(y, z, x, rstride=1, cstride=1, cmap='rainbow')
ax.set_xlabel('传统车辆', color='r')
ax.set_ylabel('时间', color='g')
ax.set_zlabel('新能源车辆', color='b')#给三个坐标轴注明
plt.show()
