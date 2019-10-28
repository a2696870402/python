import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

PATH = "C:/Users/ouguangji/Desktop/machine-learning-ex1/ex1"

data = pd.read_csv(PATH+'/ex1data1.txt', names=['populdation', 'profit'])
# names=['population', 'profit'] 给数据集的列添加标题

print(data.head())      # 前五行头
print(data.describe())  # 方差  最大最小值 平均值
print(data.info())      # 信息
# # 数据可视化
# data.plot.scatter('populdation', 'profit', c='b', label='populdation', s=30)
# plt.show()

# 加入特征列 x0=1
data.insert(0, 'ones', 1)  # 要插入的位置
# print(data.head())

# 获取特征与标签
x = data.iloc[:, 0:-1]
# print(x.head())
x = x.values  # 类型转换，便于numpy计算

# print(x)

y = data.iloc[:, -1].values
# print(data.head())
# print(y.shape)
# print(y)
y = y.reshape(-1, 1)  # 转为二维
# y = y[:, None]  # 转为二维
# print(y.shape)
# print(y)

# 损失函数
def costFunction(x, y, theta):
    inner = np.power(np.dot(x, theta)-y, 2)
    return np.sum(inner)/(2*len(x))


# 梯度下降算法
def tD_xiajiang(x, y, theta, al, iters, is_plot=False):
    costs = [] # 存放每一次迭代的cost值
    for i in range(iters):
        theta = theta - al/len(x)*(np.dot(x.T, np.dot(x, theta)-y))
        cost = costFunction(x, y, theta)
        costs.append(cost)

        if i % 100 == 0:
            if is_plot:
                print(cost)
    return theta, costs


# 初始化为参数0
theta = np.zeros((2, 1))    # (97,2)*(2，1)=（97，1）
# 计算初始的代价函数 J(ce_ta)
cost_init = costFunction(x, y, theta)
# 学习率
al = 0.02
# 迭代次数
iters = 2000

# fig 表示整个图
# ax表示实例化的对象
theta_final, costs = tD_xiajiang(x, y, theta, al, iters, True)
#
# # 可视化 损失函数
# ax.plot(np.arange(iters), costs)
# ax.set(
#     xlabel='number of iters',
#     ylabel='costs',
#     title='cost vs iters'
# )
# plt.show()
#
# 拟合的可视化
a = np.linspace(y.min(), y.max(), 100)
# 利用得到的最优参数， 拟合曲线塞
y_pred = theta_final[0, 0]+theta_final[1, 0]*a

fig, ax = plt.subplots()
# 真实样本
ax.scatter(x[:, 1], y, label='train data')
# 预测结果
ax.plot(a, y_pred, label='predict',color='red')
ax.legend()
plt.show()




