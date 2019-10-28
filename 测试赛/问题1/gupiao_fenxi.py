import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame

'''
股票预测实现
'''
'''
1.数据加载
'''
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2019, 7, 31)
df = web.DataReader("AAPL", 'yahoo', start, end)
#df = pd.DataFrame(df)
#df.to_csv('../data/gupiao.csv')

'''
得到滚动平均率
'''
close_px = df['Adj Close']
mavg = close_px.rolling(window=100).mean()

import matplotlib.pyplot as plt
from matplotlib import style

# import matplotlib as mpl
# mpl.rc('figure', figsize=(8, 7))
# '''
# 画出趋势图
# '''
# style.use('ggplot')
# close_px.plot(label='AAPL')
# mavg.plot(label='mavg')
# plt.legend()

# '''
# 收益率显示
# '''
# rets = close_px/close_px.shift(1)-1
# rets.plot(label='return')
# plt.show()

# '''
# 分析竞争对手股票
# '''
# dfcomp = web.DataReader(['AAPL', 'GE', 'GOOD', 'IBM', 'MSFT'],  'yahoo', start=start, end=end)['Adj Close']
# print(dfcomp)
# retscomp = dfcomp.pct_change()
# corr = retscomp.corr()

# plt.scatter(retscomp.AAPL, retscomp.GE)
# plt.xlabel('Returns AAPL')
# plt.ylabel('Returns GE')
# plt.show()

# 股票回报率和风险
# plt.scatter(retscomp.mean(), retscomp.std())
# plt.xlabel('Expected returns')
# plt.ylabel('Risk')
# for label, x, y in zip(retscomp.columns, retscomp.mean(), retscomp.std()):
#     plt.annotate(label, xy=(x, y), xytext=(20, -20),
#                  textcoords='offset points', ha='right', va='bottom',
#                  bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
#                  arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=0'))
# plt.show()
#

dfreg = df.loc[:, ['Adj Close', 'Volume']]
dfreg['HL_PCT'] = (df['High']-df['Low'])/df['Close']*100.0       # 上升率
dfreg['PCT_change'] = (df['Close']-df['Open'])/df['Open']*100.0  # 增长率

# 删除缺失值

import math
import numpy as np
from sklearn import preprocessing
dfreg.fillna(value=-99999, inplace=True)
# 分离标签 预测的Adjclose
forecast_out = int(math.ceil(0.01*len(dfreg)))
forecast_col = 'Adj Close'
dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
# 缩放X 数据的归一化
X = np.array(dfreg.drop(['label'], 1))
X = preprocessing.scale(X)
# 寻找近期能用于训练的数据
X_lately = X[-forecast_out:]
X1 = X[:-forecast_out]
# 分离标签并识别Y
y = np.array(dfreg['label'])
y = y[:-forecast_out]
X_train = X1
X_test = X1
y_train = y
y_test = y
''' 
预测模块
'''
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# 线性回归 的使用
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)

clf2 = make_pipeline(PolynomialFeatures(2), Ridge())
clf2.fit(X_train, y_train)

clf3 = make_pipeline(PolynomialFeatures(3), Ridge())
clf3.fit(X_train, y_train)

# K最近邻KNN 的使用
clf_knn = KNeighborsRegressor(n_neighbors=3)
clf_knn.fit(X_train, y_train)

# 评估

confince = clf.score(X_test, y_test)
confince2 = clf2.score(X_test, y_test)
confince3 = clf3.score(X_test, y_test)
confince_knn = clf_knn.score(X_test, y_test)

print("clf:", confince)
print("clf1:", confince2)
print("clf2:", confince3)
print("clf_knn:", confince_knn)

# 预测股票发展趋势 可视化
# 使用knn方式进行预测
#

forecast_set = clf_knn.predict(X_lately)
dfreg['Forecast'] = np.nan
last_date = dfreg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)
for i in forecast_set:
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]
dfreg['Adj Close'].tail(500).plot()
dfreg['Forecast'].tail(500).plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
print(forecast_set)

