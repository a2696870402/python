import math
import random
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd


random.seed(0)
mm = MinMaxScaler()
frequency = []
error_s = []


def rand(a, b):
    return (b - a) * random.random() + a


def make_matrix(m, n, fill=0.0):
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat


def sigmoid(x):
    return (2.0*1.0 / (1.0 + np.exp(-x)))-1


def sigmoid_derivative(x):
    return 2.0*np.exp(-x)/((1+np.exp(-x))**2)


class BPNeuralNetwork:
    # 网络初始化
    def __init__(self):
        self.input_n = 0
        self.hidden_n = 0
        self.output_n = 0
        self.input_cells = []
        self.hidden_cells = []
        self.output_cells = []
        self.input_weights = []
        self.output_weights = []
        self.input_correction = []
        self.output_correction = []

    # 前向传播
    def setup(self, ni, nh, no):
        self.input_n = ni + 1
        self.hidden_n = nh
        self.output_n = no
        # init cells
        self.input_cells = [1.0] * self.input_n
        self.hidden_cells = [1.0] * self.hidden_n
        self.output_cells = [1.0] * self.output_n
        # init weights
        self.input_weights = make_matrix(self.input_n, self.hidden_n)
        self.output_weights = make_matrix(self.hidden_n, self.output_n)
        # random activate
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = rand(-0.2, 0.2)
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = rand(-2.0, 2.0)
        # init correction matrix
        self.input_correction = make_matrix(self.input_n, self.hidden_n)
        self.output_correction = make_matrix(self.hidden_n, self.output_n)

    # 预测
    def predict(self, inputs):
        # activate input layer
        for i in range(self.input_n - 1):
            self.input_cells[i] = inputs[i]
        # activate hidden layer
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_weights[i][j]
            self.hidden_cells[j] = sigmoid(total)
        # activate output layer
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n):
                total += self.hidden_cells[j] * self.output_weights[j][k]
            self.output_cells[k] = sigmoid(total)
        return self.output_cells[:]

    # 反向传播
    def back_propagate(self, case, label, learn, correct):
        # feed forward
        self.predict(case)
        # get output layer error
        output_deltas = [0.0] * self.output_n
        for o in range(self.output_n):
            error = label[o] - self.output_cells[o]
            output_deltas[o] = sigmoid_derivative(self.output_cells[o]) * error
        # get hidden layer error
        hidden_deltas = [0.0] * self.hidden_n
        for h in range(self.hidden_n):
            error = 0.0
            for o in range(self.output_n):
                error += output_deltas[o] * self.output_weights[h][o]
            hidden_deltas[h] = sigmoid_derivative(self.hidden_cells[h]) * error
        # update output weights
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden_cells[h]
                self.output_weights[h][o] += learn * change + correct * self.output_correction[h][o]
                self.output_correction[h][o] = change
        # update input weights
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                change = hidden_deltas[h] * self.input_cells[i]
                self.input_weights[i][h] += learn * change + correct * self.input_correction[i][h]
                self.input_correction[i][h] = change
        # get global error
        error = 0.0
        for o in range(len(label)):
            error += 0.5 * (label[o] - self.output_cells[o]) ** 2
        return error

    # 训练网络
    def train(self, cases, labels, limit=1000, learn=0.05, correct=0.1):
        t = 0.0
        for j in range(limit):
            error = 0.0
            for i in range(len(cases)):
                label = labels[i]
                case = cases[i]
                error += self.back_propagate(case, label, learn, correct)
            print("次数=", j, "error=", error, "t= ", t)
            if t != 0.0 and t < error and j > 1000:
                break
            t = error
            frequency.append(j)
            error_s.append(error)
            if error < correct:
                break

    # 网络测试
    '''
    网络的初始化加载
    1.数据加载
    2.数据归一化
    3.网络训练
    '''
    def test(self):
        data1 = open('first_data.csv')
        data2 = pd.read_csv(data1)

        X = data2[['人口', 'GDP', '垃圾处理厂数量', '城市化率', '污水排放量', '食品制造业固体废物产生量（万吨）']]
        y = data2[['垃圾总产量']]
        X = np.asarray(X)
        y = np.asarray(y)
        # 数据归一化
        # cases = mm.fit_transform(X)
        x = mm.fit_transform(X).T
        # print(X)
        y = mm.fit_transform(y).T
        print(y)
        # print(y)
        # # 数据归一化
        self.setup(6, 12, 1)
        self.train(x, y, 100000, 0.2, 0.0001)
        # 训练


if __name__ == '__main__':
    # 网络训练
    nn = BPNeuralNetwork()
    nn.test()
    # 数据预测
    # 1.加载数据
    data3 = open('first_test_data.csv')
    data4 = pd.read_csv(data3)
    X_predict = data4[['人口', 'GDP', '垃圾处理厂数量', '城市化率', '污水排放量', '食品制造业固体废物产生量（万吨）']]
    X_predict = np.asarray(X_predict)
    # 最大最小归一化
    X_predict = mm.fit_transform(X_predict)
    print(X_predict)
    ee = []
    X_predict = np.asarray(X_predict)
    for i in range(0, 20):
        predict = nn.predict(X_predict[i])
        print(predict)
        ee.append(predict)
        print(type(predict))
    print("ee:", ee)
    fee = open('ce1.csv')
    # 设置最大最小值范围
    fee = pd.read_csv(fee)[['垃圾总产量']]
    fee = mm.fit_transform(np.asarray(fee))
    #
    # 预测值反归一
    result = mm.inverse_transform(ee)
    #
    print(result)
    # print(predict_s)
    for i in range(20):
        print(2019+i, "年预测值：", result[i])
    # plt.title('BP神经网络训练次数和误差值的关系', size=14)
    # plt.xlabel('训练次数', size=10)
    # plt.ylabel('误差值', size=10)
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    # plt.plot(frequency, np.power(error_s, 2), '.-')
    # plt.show()

    plt.title("预测重庆20年的垃圾总产量")
    plt.xlabel('年份', size=10)
    plt.ylabel('垃圾总产量', size=10)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    year = []
    for i in range(20):
        year.append(i+2019)
    plt.plot(year, result, '.-')
    plt.show()

