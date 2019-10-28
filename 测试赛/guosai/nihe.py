import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

f = open('data/e.csv')
data = pd.read_csv(f)
x = data[['x']]
y = data[['y']]
x = np.asarray(x).ravel()
y = np.asarray(y).ravel()
f1 = np.polyfit(x, y, 1)  # 用一次拟合
print(f1)
plot1 = plt.plot(x, y, 's',label='original values')
plt.show()