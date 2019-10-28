import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

f = open('1.csv')
data = pd.read_csv(f)
print(data)
year_e = data[['year']]
year = [datetime.strptime(d, '%Y/%m/%d').date() for d in np.asarray(year_e).ravel()]
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

y11 = np.asarray(data[['传统车商业区']])
y12 = np.asarray(data[['传统车居住区']])
y21 = np.dot(np.asarray(data[['新能源商业区']]),10)
y22 = np.dot(np.asarray(data[['新能源居住区']]),10)
b1 = np.asarray(data[['商业区的比例']])
b2 = np.asarray(data[['居住区的比例']])
plt.plot(year, y11, label='传统车商业区车位需求')
plt.plot(year, y12, label='传统车居住区车位需求')
plt.plot(year, y21, label='新能源车商业区车位需求(*10)')
plt.plot(year, y22, label='新能源车居住区车位需求(*10)')
plt.title("传统车和新能源车分别在商业区和居住区的需求", size=18)
plt.xlabel("时间", size=15)
plt.ylabel("指数", size=15)
plt.legend()
plt.show()


plt.plot(year, b1, label='商业区')
plt.plot(year, b2, label='居住区')
plt.title("传统车车位需求和新能源车位需求的比例曲线(传统车/新能源车)", size=16)
plt.xlabel("时间", size=12)
plt.ylabel("指数", size=12)
plt.legend()
plt.show()



