import matplotlib.pyplot as plt
import numpy as np


a = [0,1,2,3,4,5,6,6.5, 7,8,8.5, 9,10,11, 12,13,14, 15,16,17,17.5, 18,19,20,21,22,23]
f1 = [3.75, 3.75, 3.75, 3.75, 3.75, 3.75, 3.75, 3.75, 3.25, 3.25, 3.25, 2.75, 2.75,
      2.75, 3.25, 3.25, 3.25, 2.75, 2.75, 2.75, 2.75, 3.75, 3.75, 3.75, 3.75, 3.75, 3.75]
f2 = []
for i in f1:
    if i == 3.75:
        f2.append(1.5)
    elif i == 3.25:
        f2.append(1.3)
    elif i == 2.75:
        f2.append(1.1)
print(a)
print(f1)
print(f2)














#
# a = []
# for i in range(24):
#     a.append(i)
# f1 = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3.5, 3.5, 3.5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3.5, 3]
# f2 = [1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.4,
#       1.4, 1.4, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.4, 1.2]
# a = np.asarray(a)
# f1 = np.asarray(f1)
# f2 = np.asarray(f2)
# plt.plot(a, f1, '.-', label="新能源车")
# plt.plot(a, f2, '.-', label="传统车")
# plt.legend()
# plt.xlabel("时间", size=15)
# plt.ylabel("价格", size=15)
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# plt.title("商业区新能源车车位和传统车车位一天时序价格曲线", size=20)
# plt.show()
