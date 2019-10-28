import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data1 = open('test01.csv')
# data2 = open('first_test2_data.csv')
data1 = pd.read_csv(data1)

#
# x1 = []
# x2 = []
# for i in range(1996, 2019):
#     x1.append(i)
# for i in range(2019, 2039):
#     x2.append(i)

y1 = np.asarray(data1[['sum1']]).ravel()
y2 = np.asarray(data1[['sum2']]).ravel()
x1 = np.asarray(data1[['year']]).ravel()
x2 = np.asarray(data1[['year']]).ravel()
print(y1)
print(y2)
print(x1)
print(x2)
plt.grid(linestyle='-.')
for a, b in zip(x1[-1:], y1[-1:]):
    plt.text(a, b, (a,b),ha='center', va='bottom', fontsize=10)
for a,b in zip(x2[-1:],y2[-1:]):
    plt.text(a, b, (a, b), ha='center', va='bottom', fontsize=10)
plt.plot(x1, y1, '.-', color='r', label="改善前曲线")
plt.plot(x2, y2, '-^', color='b', label="改善后曲线")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False       #解决负数坐标显示问题
plt.xlabel("年份", fontsize=10)
plt.ylabel("垃圾产生总量(万吨)",fontsize=10)
plt.title("未来20年垃圾产生总量预测图(垃圾分类后与之前对比图)")
plt.legend()
plt.show()




