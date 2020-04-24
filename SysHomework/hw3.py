import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f
from scipy.stats import norm


# 第1问要求函数,输入为N*2的list
def linear_regeression1(data, alpha):
    # 转为numpy array
    data = np.array(data)
    # 输入非法时抛出异常
    if data.ndim is not 2 or data.shape[1] is not 2:
        raise ValueError('The dimension of your input data is false. N*2 is suitable.')
    if alpha < 0 or alpha > 1:
        raise ValueError('The value of your input alpha is false. expected between 0 and 1.')
    N = data.shape[0]
    Y = data[:, 0].reshape(-1, 1)
    X = data[:, 1].reshape(-1, 1)
    # 去均值归一化
    X = X - np.mean(X)
    Y = Y - np.mean(Y)
    # 求a,b
    Lxx = np.matmul(X.T, X) - np.sum(X)**2 / 10
    Lxy = np.matmul(X.T, Y) - np.sum(X) * np.sum(Y) / 10
    Lyy = np.matmul(Y.T, Y) - np.sum(Y)**2 / 10
    b = Lxy / Lxx
    a = np.mean(Y) - b*np.mean(X)
    if b > 0:
        print('1-d Linear Regression Equation:y={:.4f}+{:.4f}*x'.format(float(a), float(b)))
    else:
        print('1-d Linear Regression Equation:y={:.4f}{:.4f}*x'.format(float(a), float(b)))
    # 求ESS和RSS
    Y_pred = a + b * X
    ESS = b * Lxy
    RSS = Lyy - ESS
    F = (N-2) * ESS / RSS
    # 求临界值
    Fa = f.ppf(1-alpha, 1, N-2)
    # 输出检验结果
    if F > Fa:
        print('输入数据存在线性关系')
        # 绘图操作
        plt.figure('1-d Linear Regression Picture')
        plt.scatter(X, Y, s=10, c='k')
        X_line = np.linspace(np.min(X), np.max(X), 10)
        Y_line = (a + b * X_line).reshape(-1)
        plt.plot(X_line, Y_line, 'red', label='Regression Line')
        S = np.sqrt(RSS / (N-2))
        Z = norm.ppf(1-0.05/2)
        Y_line_low = (Y_line - S*Z).reshape(-1)
        Y_line_high = (Y_line + S*Z).reshape(-1)
        plt.plot(X_line, Y_line_low, 'blue', label='Confidence Interval')
        plt.plot(X_line, Y_line_high, 'blue')
        plt.legend()
        plt.grid()
        plt.show()
        # 打印置信区间
        for i in range(0, data.shape[0]):
            boundary_low = float(Y_pred[i]-S*Z)
            boundary_high = float(Y_pred[i]+S*Z)
            print('x:{:.4f},置信区间[{:.4f},{:.4f}]'.format(float(X[i]), boundary_low, boundary_high))
    else:
        print('输入数据不存在线性关系.')


if __name__ == '__main__':
    data = [[4.0, 0.009],
            [3.44, 0.013],
            [3.6, 0.006],
            [1.0, 0.025],
            [2.04, 0.022],
            [4.74, 0.007],
            [0.6, 0.036],
            [1.7, 0.014],
            [2.92, 0.016],
            [4.8, 0.014],
            [3.28, 0.016],
            [4.16, 0.012],
            [3.35, 0.020],
            [2.2, 0.018]]
    linear_regeression1(data, 0.05)
