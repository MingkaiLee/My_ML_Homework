import numpy as np
from minepy import MINE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import heapq


L_model = LogisticRegression(solver='lbfgs', max_iter=1000)
tree_model = DecisionTreeClassifier(random_state=0)


# 计算类内类间距离,度量为欧氏距离,输入为列向量
def distance_measure(X, y):
    # 获取类别数
    c = np.unique(y)
    c_num = len(c)
    # 数据集划分,数据集仍为行向量
    c_list = []
    for i in range(0, c_num):
        t_list = []
        for j in range(0, len(X)):
            if y[j] == c[i]:
                t_list.append(X[j])
        c_list.append(t_list)
    # 计算均值,结果为列向量
    m_list = []
    m = np.zeros(np.shape(X[0].T))
    for i in range(0, c_num):
        c_list[i] = np.array(c_list[i])
        m_list.append((np.mean(c_list[i])).T)
        m = m + c_list[i].shape[0] * (np.mean(c_list[i])).T / X.shape[0]
    # 计算Sb和Sw
    Sb = np.zeros(np.shape(np.dot(X[0].T, X[0])))
    Sw = np.zeros(np.shape(np.dot(X[0].T, X[0])))
    for i in range(0, c_num):
        Sb = Sb + np.dot((m_list[i]-m), (m_list[i]-m).T) * (c_list[i].shape[0] / X.shape[0])
        S = np.zeros(np.shape(np.dot(X[0].T, X[0])))
        for k in range(0, c_list[i].shape[0]):
            S = S + np.dot((c_list[i])[k].T-m_list[i], (c_list[i])[k]-m_list[i].T)
        Sw = Sw + S / X.shape[0]
    if Sb.ndim > 1:
        return np.trace(Sb) / np.trace(Sw)
    else:
        return Sb / Sw


# 特征选择函数
def feature_choice(X, y, num=1, judgement='distance'):
    # num不合理时抛出异常
    if num > X.shape[1]:
        raise ValueError('You input characteristic quantity is more than original.(d>D)')
    dis = []
    # 类内类间距
    if judgement is 'distance':
        for i in range(0, X.shape[1]):
            dis.append(distance_measure(X[:, i], y))
    # 最大信息系数
    elif judgement is 'mine':
        mine = MINE(alpha=0.6, c=15, est='mic_approx')
        for i in range(0, X.shape[1]):
            mine.compute_score(X[:, i].flatten(), y.flatten())
            dis.append(mine.mic())
    else:
        raise ValueError('Incorrect judgement type.(Only support \'distance\' or \'mine\')')
    # 取最大的n个特征维度
    ranks = list(map(dis.index, heapq.nlargest(num, dis)))
    # 返回选中特征的秩
    return ranks


# 特征选择前向算法,判据为类内类间距
def forward_feature_choice(X, y, num=1):
    # num不合理时抛出异常
    if num > X.shape[1]:
        raise ValueError('You input characteristic quantity is more than original.(d>D)')
    dis = []
    # 先取最大的特征
    for i in range(0, X.shape[1]):
        dis.append(distance_measure(X[:, i], y))
    ranks = list(map(dis.index, heapq.nlargest(1, dis)))
    f_vec = X[:, ranks[0]]
    for i in range(1, num):
        max_dis = 0
        new_feature = 0
        # 逐一找新特征
        for j in range(0, X.shape[1]):
            if j not in ranks:
                temp_vec = np.column_stack((f_vec, X[:, j]))
                temp_dis = distance_measure(temp_vec, y)
                if temp_dis > max_dis:
                    max_dis = temp_dis
                    new_feature = j
        ranks.append(new_feature)
        f_vec = np.column_stack((f_vec, X[:, new_feature]))
    # 返回选中的秩
    return ranks


# (1)实验代码
def exp1(X, y, X_test, y_test):
    f_num = [1, 5, 10, 20, 50, 100]
    # 类内类间距离判据选出的特征
    d_choices = []
    # 最大信息系数判据选出的特征
    m_choices = []
    for i in range(0, len(f_num)):
        d_choices.append(feature_choice(X, y, f_num[i], 'distance'))
        m_choices.append(feature_choice(X, y, f_num[i], 'mine'))
    for i in range(0, len(f_num)):
        print('选出{}个特征,两方法选出的特征重合数为{}'.format(f_num[i], len(set(d_choices[i]) & set(m_choices[i]))))
        # 类内类间判据选出的进行训练
        x = np.zeros((X.shape[0], len(d_choices[i])))
        x_test = np.zeros((X_test.shape[0], len(d_choices[i])))
        for j in range(0, len(d_choices[i])):
            x[:, j] = X[:, (d_choices[i])[j]]
            x_test[:, j] = X_test[:, (d_choices[i])[j]]
        L_model.fit(x, y.flatten())
        print('类内类间距法准确率:{}'.format(L_model.score(x_test, y_test.flatten())))
        for j in range(0, len(m_choices[i])):
            x[:, j] = X[:, (m_choices[i])[j]]
            x_test[:, j] = X_test[:, (m_choices[i])[j]]
        L_model.fit(x, y.flatten())
        print('最大信息系数准确率:{}'.format(L_model.score(x_test, y_test.flatten())))
    # 不作特征选择的结果
    L_model.fit(X, y.flatten())
    print('不作特征选择准确率:{}'.format(L_model.score(X_test, y_test.flatten())))


# (2)实验代码,exp1基础上作些修改
def exp2(X, y, X_test, y_test):
    f_num = [1, 5, 10, 20, 50, 100]
    # 类内类间距离判据选出的特征
    d_choices = []
    # 前向算法选出的特征
    f_choices = []
    for i in range(0, len(f_num)):
        d_choices.append(feature_choice(X, y, f_num[i], 'distance'))
        f_choices.append(forward_feature_choice(X, y, f_num[i]))
    for i in range(0, len(f_num)):
        print('选出{}个特征,两方法选出的特征重合数为{}'.format(f_num[i], len(set(d_choices[i]) & set(f_choices[i]))))
        # 类内类间判据选出的进行训练
        x = np.zeros((X.shape[0], len(d_choices[i])))
        x_test = np.zeros((X_test.shape[0], len(d_choices[i])))
        for j in range(0, len(d_choices[i])):
            x[:, j] = X[:, (d_choices[i])[j]]
            x_test[:, j] = X_test[:, (d_choices[i])[j]]
        L_model.fit(x, y.flatten())
        print('类内类间距法准确率:{}'.format(L_model.score(x_test, y_test.flatten())))
        for j in range(0, len(f_choices[i])):
            x[:, j] = X[:, (f_choices[i])[j]]
            x_test[:, j] = X_test[:, (f_choices[i])[j]]
        L_model.fit(x, y.flatten())
        print('前向算法(类内类间距判据)准确率:{}'.format(L_model.score(x_test, y_test.flatten())))
    # 不作特征选择的结果
    L_model.fit(X, y.flatten())
    print('不作特征选择准确率:{}'.format(L_model.score(X_test, y_test.flatten())))


# (3)实验代码,exp2基础上作些添加
def exp3(X, y, X_test, y_test):
    f_num = [1, 5, 10]
    # 类内类间距离判据选出的特征
    d_choices = []
    # 前向算法选出的特征
    f_choices = []
    # 添加部分:决策树选特征
    tree_model.fit(X, y)
    t_info = list(tree_model.feature_importances_)
    t_choices = []
    for i in range(0, len(f_num)):
        d_choices.append(feature_choice(X, y, f_num[i], 'distance'))
        f_choices.append(forward_feature_choice(X, y, f_num[i]))
        t_choices.append(list(map(t_info.index, heapq.nlargest(f_num[i], t_info))))
    for i in range(0, len(f_num)):
        print('选出{}个特征,类内类间距和前向算法选出的特征重合数为{}'.format(f_num[i], len(set(d_choices[i]) & set(f_choices[i]))))
        print('类内类间距和决策树选出的特征重合数为{}'.format(len(set(d_choices[i]) & set(t_choices[i]))))
        print('前向算法和决策树选出的特征重合数为{}'.format(len(set(f_choices[i]) & set(t_choices[i]))))
        # 类内类间判据选出的进行训练
        x = np.zeros((X.shape[0], len(d_choices[i])))
        x_test = np.zeros((X_test.shape[0], len(d_choices[i])))
        for j in range(0, len(d_choices[i])):
            x[:, j] = X[:, (d_choices[i])[j]]
            x_test[:, j] = X_test[:, (d_choices[i])[j]]
        L_model.fit(x, y.flatten())
        print('类内类间距法准确率:{}'.format(L_model.score(x_test, y_test.flatten())))
        for j in range(0, len(f_choices[i])):
            x[:, j] = X[:, (f_choices[i])[j]]
            x_test[:, j] = X_test[:, (f_choices[i])[j]]
        L_model.fit(x, y.flatten())
        print('前向算法(类内类间距判据)准确率:{}'.format(L_model.score(x_test, y_test.flatten())))
        for j in range(0, len(t_choices[i])):
            x[:, j] = X[:, (t_choices[i])[j]]
            x_test[:, j] = X_test[:, (t_choices[i])[j]]
        L_model.fit(x, y.flatten())
        print('决策树准确率:{}'.format(L_model.score(x_test, y_test.flatten())))
    # 不作特征选择的结果
    L_model.fit(X, y.flatten())
    print('不作特征选择准确率:{}'.format(L_model.score(X_test, y_test.flatten())))


if __name__ == '__main__':
    features = pd.read_table('./HomeworkData/feature_selection_X.txt', header=None)
    features = np.array(features)
    labels = pd.read_table('./HomeworkData/feature_selection_Y.txt', header=None)
    labels = np.array(labels)
    # 随机划分数据集
    X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=100)
    # exp1(X_train, Y_train, X_test, Y_test)
    # exp2(X_train, Y_train, X_test, Y_test)
    exp3(X_train, Y_train, X_test, Y_test)
