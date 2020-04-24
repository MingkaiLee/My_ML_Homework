import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# 去除含有'?'项的过滤器
def strcut(x):
    for i in range(0, len(x)):
        if x[i] == '?':
            return False
    return True


# 构建LDA模型类,仅针对本题数据而不具有普适性
class LDA_classifier:
    def __init__(self):
        direction = 0
        boundary = 0

    # 模型训练
    def fit(self, data):
        # 类分割
        data_list = data.tolist()
        data0 = []
        data1 = []
        for i in range(0, len(data_list)):
            if data_list[i][10] == 0:
                data0.append(data_list[i])
            else:
                data1.append(data_list[i])
        data0 = np.array(data0)
        data1 = np.array(data1)
        X0 = data0[:, 1:10]
        y0 = data0[:, 10]
        X1 = data1[:, 1:10]
        y1 = data1[:, 10]
        # 计算类均值向量
        m0 = np.mean(X0, axis=0)
        m1 = np.mean(X1, axis=0)
        m0_ = np.mean(y0)
        m1_ = np.mean(y1)
        # 计算类内离散度矩阵
        S0 = np.zeros([9, 9])
        S1 = np.zeros([9, 9])
        for i in range(0, len(X0)):
            S0 += np.matmul((X0[i]-m0).reshape(-1, 1), (X0[i]-m0).reshape(1, -1))
        for i in range(0, len(X1)):
            S1 += np.matmul((X1[i]-m1).reshape(-1, 1), (X1[i]-m1).reshape(1, -1))
        # 总类内离散度矩阵
        Sw = S0 + S1
        # 计算投影方向
        self.direction = np.matmul(np.linalg.inv(Sw), (m0-m1).reshape(-1, 1))
        self.direction = self.direction.reshape(1, -1)
        self.boundary = -(m0_ + m1_) / 2

    # 结果预测,直接给出准确率
    def pred(self, data):
        X = data[:, 1:10]
        y = data[:, 10]
        # 预测正确数
        correct_num = 0
        for i in range(0, len(X)):
            result = np.matmul(self.direction, X[i].reshape(-1, 1))
            if result > self.boundary:
                y_ = 0
            else:
                y_ = 1
            if y_ == y[i]:
                correct_num += 1
        return correct_num/len(data)


# Logistic模型构建
Lmodel = LogisticRegression(solver='liblinear')
# LDA模型构建
Fmodel = LDA_classifier()
# 数据读取
data = pd.read_table('./HomeworkData/breast-cancer-wisconsin.txt', header=None)
data = np.array(data)
data_list = data.tolist()
# 过滤含有'?'的数据行
data = filter(strcut, data_list)
data = list(data)
data = np.array(data, dtype=np.uint64)
# 训练集和测试集划分,共进行五轮实验,由于过滤后共683个数据,取其中500个为训练集,剩下的为测试集
for echo in range(0, 5):
    np.random.shuffle(data)
    train_X = data[:500, 1:10]
    train_y = data[:500, 10]
    test_X = data[500:, 1:10]
    test_y = data[500:, 10]
    # Logistic回归
    Lmodel.fit(train_X, train_y)
    Lscore = Lmodel.score(test_X, test_y)
    print('echo {}'.format(echo))
    print('Logistic准确率:{}'.format(Lscore))
    Fmodel.fit(data[:500, :])
    Fscore = Fmodel.pred(data[500:, :])
    print('LDA准确率:{}'.format(Fscore))


