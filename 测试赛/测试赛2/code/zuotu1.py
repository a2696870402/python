import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


f = open("../data/unit1.csv")
data = pd.read_csv(f)
print(data)
data = data[['year', 'y2']]
year1 = data[['year']]
year2 = year1[:-12]
year1 = np.asarray(year1)
year2 = np.asarray(year2)
data1 = data[['y2']]
data2 = data1[:-12]
data1 = np.asarray(data1)
data2 = np.asarray(data2)
xs1 = [datetime.strptime(d, '%Y/%m/%d').date() for d in year1.ravel()]
xs2 = [datetime.strptime(d, '%Y/%m/%d').date() for d in year2.ravel()]
plt.plot(xs1, data1, '^-', label="predict")
plt.plot(xs2, data2, '.-', label="result")
plt.xlabel("月份")
plt.ylabel("万辆")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.legend()
plt.show()
