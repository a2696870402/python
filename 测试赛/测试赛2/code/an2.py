# coding=utf-8
import matplotlib.pyplot as plt


def calc_next_s(alpha, s):
    # 计算s的值，哪怕指数的次数再高，计算的流程都是一样的
    s2 = [0 for i in range(len(s))]
    # 我是取前三个的平均来初始化s0
    s2[0] = sum(s[0:3]) / float(3)
    for i in range(1, len(s2)):
        # 跳过第一个，因为第一个之前没有可用来预测的参数
        s2[i] = alpha * s[i] + (1 - alpha) * s2[i - 1]
        # 这里其实有点小bug，也就是s2[0]到底是保留当第一个，
        # 还是用来做第0个，只用来预测新的s2值，之后不保留进列表
    return s2


def exponential_smoothing():
    # 设定alpha，这是平滑系数
    alpha = 0.5

    # 数据载入可选择1
    data = []
    # 读取文件
    with open('../data/unit11.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            data.append(line.strip('\r\n').split())
    # 提取数据文件中的值
    data_value = [int(ele[2]) for ele in data if ele[2].isdigit()]

    # 数据载入可选择2
    # data_value = [10, 15, 8, 20, 10, 16, 18, 20, 22, 24, 20, 26, 27, 29, 29]  # 文章第一部分数据，实现和理论稍微有偏差


    # 以下为系数求解，这些都是根据定义的公式得来的：
    # 如果不懂，建议多看几遍公式
    # 一次平滑系数，只有一个s，没算上alpha
    s1 = calc_next_s(alpha, data_value)
    # 二次平滑系数，有一个s和两个系数
    s2 = calc_next_s(alpha, s1)
    a2 = [(2 * s1[i] - s2[i]) for i in range(len(s1))]
    b2 = [(alpha / (1 - alpha) * (s2[i] - s1[i])) for i in range(len(s1))]

    # 三次平滑系数，有一个s和三个系数
    s3 = calc_next_s(alpha, s2)
    a3 = [(3 * s1[i] - 3 * s2[i] + s3[i]) for i in range(len(s3))]
    b3 = [((alpha / (2 * (1 - alpha) ** 2)) * (
            (6 - 5 * alpha) * s1[i] - 2 * (5 - 4 * alpha) * s2[i] + (4 - 3 * alpha) * s3[i])) for i in
          range(len(s3))]
    c3 = [(alpha ** 2 / (2 * (1 - alpha) ** 2) * (s1[i] - 2 * s2[i] + s3[i])) for i in range(len(s3))]

    # 以下为预测部分
    # 这是一次的，是按我自己理解的，感觉不一定是这样求的
    # 我没认真细究，我觉得那个s1要一次次求，所以这里只求了一天
    # 如有什么疏漏欢迎指正
    # s_single = [0 for i in range(len(s1))]
    # for i in range(1, len(s2)):
    #     s_single[i] = data_value[i - 1] + (1 - alpha) * s1[i - 1]  # 这里有个和时间差有关的系数
    # predict_single = [0 for i in range(1)]
    # for i in range(len(predict_single)):
    #     predict_single[i] = alpha * data_value[-1] + (1 - alpha) * s1[-1]
    # s_single.extend(predict_single)
    #
    # # 二次的预测
    # # 和原始数据横坐标相同的那些天
    # s_double = [0 for i in range(len(s2))]
    # for i in range(1, len(s2)):
    #     s_double[i] = a2[i - 1] + b2[i - 1] * 1  # 这里有个和时间差有关的系数
    # # 预测未来两年,即新增未来两年的坐标
    # predict_double = [0 for i in range(2)]
    # for i in range(len(predict_double)):
    #     predict_double[i] = a2[-1] + b2[-1] * (i + 1)
    # # 把预测的两年和原本的组合放一起
    # s_double.extend(predict_double)

    # 三次的预测
    s_triple = [0 for i in range(len(s3))]
    for i in range(1, len(s3)):
        s_triple[i] = a3[i - 1] + b3[i - 1] * 1 + c3[i - 1] * (1 ** 2)
    # 预测未来两年,即新增未来两年的坐标
    predict_triple = [0 for i in range(2)]
    for i in range(len(predict_triple)):
        predict_triple[i] = a3[-1] + b3[-1] * (i + 1) + c3[-1] * (1 ** (i + 1))
    s_triple.extend(predict_triple)

    # 绘制所有的曲线
    plt.plot(data_value,'.-')
    plt.plot(s_triple,'.-')
    plt.show()




if __name__ == '__main__':
    exponential_smoothing()
