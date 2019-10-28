import math
import random
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.externals import joblib

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
            # if t != 0.0 and t < error and j > 1000:
            #     break
            # t = error
            # frequency.append(j)
            # error_s.append(error)
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
        data1 = open('data/data1.csv')
        data2 = pd.read_csv(data1)

        X = data2[['乘客到达/出发数量(千万人次)','距离市中心的距离(km)','出租车数量(万辆)','城市人口数量(万人)',
                    '出租车价格(/km)','交通等级']]
        y = data2[['选择']]
        X = np.asarray(X)
        y = np.asarray(y)
        # 数据归一化
        # cases = mm.fit_transform(X)
        x = mm.fit_transform(X)
        # print(X)
        y = mm.fit_transform(y)
        print(y)
        # print(y)
        # # 数据归一化
        self.setup(6, 12, 1)
        self.train(x, y, 100000, 0.1, 0.0012)
        # 训练

falt = 0
if __name__ == '__main__':
    if falt == 1:
        nn = joblib.load('train_model_result.m')
    else:
        nn = BPNeuralNetwork()
    nn.test()

    joblib.dump(nn, 'train_model_result.m')
    # f = open('data/1.csv')
    # data = pd.read_csv(f)
    # X_predict = data[['乘客到达/出发数量(万人次)', '距离市中心的距离(km)', '出租车数量(万辆)', '城市人口数量(万人)',
    #                   '出租车价格(/km)', '交通等级']]
    # X = mm.fit_transform(X_predict)
    # X = np.asarray(X)
    # # nn = joblib.load('train_model_result.m')
    # a = []
    # for i in range(len(X_predict)):
    #     y_predict = nn.predict(X[i])
    #     a.append(y_predict)
    #     print(y_predict)
    # ne = pd.DataFrame(a)
    # ne.to_csv('se.csv')



    # print(nn)
    input = nn.input_cells
    hidden = nn.hidden_cells
    output = nn.output_cells
    print(input)
    print(hidden)
    print(output)
    input_wegiht = np.asarray(nn.input_weights)[:-1,:]
    output_wegiht = np.asarray(nn.output_weights)
    print(input_wegiht.shape)
    print(output_wegiht.shape)
    print(np.dot(input_wegiht, output_wegiht))





