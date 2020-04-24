import numpy as np
import matplotlib.pyplot as plt


def gaussain_dis(x, mean, sigma2):
    y = np.zeros(len(x))
    for i in range(0, len(x)):
        y[i] = np.exp(-((x[i] - mean) ** 2) / (2 * sigma2)) / np.sqrt(2 * np.pi * sigma2)
    return y


def question751(num):
    plt.figure(1)
    color_choice = ['red', 'green', 'blue', 'black']
    echo = 3
    for i in range(0, echo):
        ag = np.random.normal(loc=0, scale=1, size=num)
        # 估计均值
        mean = np.mean(ag)
        # 求方差
        var = np.var(ag)
        # 求正态分布函数
        x = np.linspace(mean - 3 * np.sqrt(var), mean + 3 * np.sqrt(var), 100)
        y = gaussain_dis(x, mean, var)
        # 开始画图
        plt.plot(x, y, color_choice[i], label='Test{}'.format(i))
    # 另绘标准正态分布
    x = np.linspace(-3, 3, 100)
    y = gaussain_dis(x, 0, 1)
    plt.plot(x, y, color_choice[3], label='Standard')
    plt.legend()
    plt.grid()
    plt.show()


def question752(mean, var):
    plt.figure()
    color_choice = ['red', 'green', 'blue', 'purple', 'black']
    priori_var = 0.01 * var
    # 生成样本
    ag = np.random.normal(loc=0, scale=1, size=1000)
    for i in range(0, 4):
        # 估计均值
        mean_pred = (1000 * priori_var) * np.mean(ag) / (1000 * priori_var + 1) + mean / (1000 * priori_var + 1)
        x = np.linspace(mean_pred - 3, mean_pred + 3, 100)
        y = gaussain_dis(x, mean_pred, 1)
        plt.plot(x, y, color_choice[i], label='Sigma0^2={}'.format(priori_var))
        # 先验方差翻十倍
        priori_var = priori_var * 10
    # 另绘标准正态分布
    x = np.linspace(-3, 3, 100)
    y = gaussain_dis(x, 0, 1)
    plt.plot(x, y, color_choice[4], label='Standard')
    plt.legend()
    plt.grid()
    plt.show()


def question753(num):
    plt.figure(1)
    color_choice = ['red', 'green', 'blue', 'black']
    echo = 3
    for i in range(0, echo):
        ag = np.random.uniform(0, 1, num)
        # 估计均值
        mean = np.mean(ag)
        # 求方差
        var = np.var(ag)
        # 求正态分布函数
        x = np.linspace(mean - 3 * np.sqrt(var), mean + 3 * np.sqrt(var), 100)
        y = gaussain_dis(x, mean, var)
        # 开始画图
        plt.plot(x, y, color_choice[i], label='Test{}'.format(i))
    # 另绘(0, 1)均匀分布
    x = np.linspace(-3, 3, 7)
    y = [0, 0, 0, 0, 1, 0, 0]
    plt.step(x, y, color_choice[3], label='U(0,1)')
    # plt.plot(x, y, color_choice[3], label='U(0,1)')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    # question751(1000)
    # question752(-5, 1)
    question753(100)
