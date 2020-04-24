import numpy as np


# 自建1维高斯核parzen窗
class Parzen1d:
    def __init__(self, r=1.0):
        self.r = r
        self.X = []
        self.Xe = []

    def fit(self, X):
        self.X = X
        for i in range(0, len(self.X)):
            self.Xe.append(np.exp(-X[i]**2 / (2*(self.r**2))))

    # 单个预测
    def pred_single(self, y):
        out = 0
        con = np.exp(-(y**2) / (2*(self.r**2)))
        for i in range(0, len(self.X)):
            out = out + np.exp(y*self.X[i] / (self.r**2)) * self.Xe[i]
        out = con * out
        return out

    # 集合预测
    def pred_series(self, Y):
        out = []
        for i in range(0, len(Y)):
            out.append(self.pred_single(Y[i]))
        return out


def bayesian_decision():
    pos_ag = np.random.normal(loc=2.5, scale=2, size=250)
    neg_ag = np.random.normal(loc=-2.5, scale=1, size=250)
    # 打乱后取前70%
    np.random.shuffle(pos_ag)
    np.random.shuffle(neg_ag)
    pos_train = pos_ag[:175]
    pos_test = pos_ag[175:]
    neg_train = neg_ag[:175]
    neg_test = neg_ag[175:]
    pos_model = Parzen1d()
    neg_model = Parzen1d()
    pos_model.fit(pos_train)
    neg_model.fit(neg_train)
    p_pos_pos = pos_model.pred_series(pos_test)
    p_pos_neg = pos_model.pred_series(neg_test)
    p_neg_pos = neg_model.pred_series(pos_test)
    p_neg_neg = neg_model.pred_series(neg_test)
    # 最小错误率决策
    min_error_pos = []
    min_error_neg = []
    for i in range(0, len(p_pos_pos)):
        if p_pos_pos[i] / p_pos_neg[i] > 1.0:
            min_error_pos.append(1)
        else:
            min_error_pos.append(0)
    for i in range(0, len(p_neg_pos)):
        if p_neg_pos[i] / p_neg_neg[i] > 1.0:
            min_error_neg.append(1)
        else:
            min_error_neg.append(0)
    correct_num = 0
    for i in range(0, len(p_pos_pos)):
        if min_error_pos[i] == 1:
            correct_num = correct_num + 1
        if min_error_neg[i] == 0:
            correct_num = correct_num + 1
    print('最小错误率贝叶斯决策的正确率:{}'.format(correct_num / (len(min_error_pos) + len(min_error_neg))))
    # 最小风险决策
    # 计算后验概率
    post_pos_pos = []
    post_pos_neg = []
    post_neg_pos = []
    post_neg_neg = []
    for i in range(0, len(p_pos_pos)):
        p1 = 0.5 * p_pos_pos[i] / (0.5*p_pos_pos[i] + 0.5*p_neg_pos[i])
        p2 = 1 - p1
        p3 = 0.5 * p_pos_neg[i] / (0.5*p_pos_neg[i] + 0.5*p_neg_neg[i])
        p4 = 1 - p3
        post_pos_pos.append(p1)
        post_pos_neg.append(p2)
        post_neg_pos.append(p3)
        post_neg_neg.append(p4)
    min_risk_pos = []
    min_risk_neg = []
    for i in range(0, len(p_pos_pos)):
        if 10*post_pos_neg[i] < post_pos_pos[i]:
            min_risk_pos.append(1)
        else:
            min_risk_pos.append(0)
        if 10*post_neg_neg[i] < post_neg_pos[i]:
            min_risk_neg.append(1)
        else:
            min_risk_neg.append(0)
    correct_num = 0
    for i in range(0, len(p_pos_pos)):
        if min_risk_pos[i] == 1:
            correct_num = correct_num + 1
        if min_risk_neg[i] == 0:
            correct_num = correct_num + 1
    print('最小风险贝叶斯决策的正确率:{}'.format(correct_num / (len(min_risk_pos) + len(min_risk_neg))))


if __name__ == '__main__':
    bayesian_decision()