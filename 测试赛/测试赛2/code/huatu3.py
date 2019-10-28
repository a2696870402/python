import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

f = open("../data/unit1.csv")
data = pd.read_csv(f)
data = data[['year', '新能源车辆保有量', '传统车车辆保有量', '停车产生率居住区', '停车产生率商业区']]
print(data)
length = len(data['year'])
print(length)
Ri1 = np.asarray(data['停车产生率居住区'].astype('float32'))
Ri2 = np.asarray(data['停车产生率商业区'].astype('float32'))
Li1 = np.asarray(data['新能源车辆保有量'].astype('float32'))
Li2 = np.asarray(data['传统车车辆保有量'].astype('float32'))
K0 = 1.5
# 计算出 传统车 和 新能源车 的停车需求
sum11 = []  # 传统车 商业区
sum12 = []  # 传统车 居住区
sum21 = []  # 新能源车 商业区
sum22 = []  # 新能源车 居住区
for i in range(length):
    sum11.append(0)
    sum12.append(0)
    sum21.append(0)
    sum22.append(0)
print(Ri1[0])
for i in range(length):
    sum11[i] += Ri2[i]*Li2[i]
    sum12[i] += Ri1[i]*Li2[i]
    sum21[i] += Ri2[i]*Li1[i]
    sum22[i] += Ri1[i]*Li1[i]
P11 = np.add(sum11, K0)
P12 = np.add(sum12, K0)
P21 = np.add(sum21, K0)
P22 = np.add(sum22, K0)
date = np.asarray([P11.T, P12.T, P21.T, P22.T]).T
pd_data = pd.DataFrame(date, columns=['传统车商业区', '传统车居住区', '新能源商业区', '新能源居住区'])
print(pd_data)
pd_data.to_csv('1.csv')
