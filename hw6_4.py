import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from condenseknn import CondensedKNeighborClassifier
from skimage import io
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os
import random
import time

# 模型创建
# model = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='auto', metric='cosine')
model = RandomForestClassifier(n_estimators=100, max_depth=20)


# model = CondensedKNeighborClassifier()

def read_images(rank):
    images = []
    for filename in os.listdir(r'./HomeworkData/Pictures/' + rank):
        img = io.imread('./HomeworkData/Pictures/' + rank + '/' + filename, as_gray=True)
        images.append(img)
    return images


def two_classify(dataset):
    # 随机选取两组图片
    rank1 = random.randint(0, 9)
    rank2 = rank1
    while rank2 == rank1:
        rank2 = random.randint(0, 9)
    X1 = dataset[rank1].copy()
    X2 = dataset[rank2].copy()
    print('本次随机选择了第{}组和第{}组图片进行实验'.format(rank1, rank2))
    # 创建标签
    Y1 = np.empty([X1.shape[0]], dtype=np.int)
    Y2 = np.empty([X2.shape[0]], dtype=np.int)
    for i in range(0, X1.shape[0]):
        Y1[i] = 0
        Y2[i] = 1
    # 同类内图片乱序
    np.random.shuffle(X1)
    np.random.shuffle(X2)
    train_X1 = X1[:225, :]
    train_X2 = X2[:225, :]
    train_X = np.concatenate((train_X1, train_X2), axis=0)
    test_X1 = X1[225:, :]
    test_X2 = X2[225:, :]
    test_X = np.concatenate((test_X1, test_X2), axis=0)
    train_Y1 = Y1[:225]
    train_Y2 = Y2[:225]
    train_Y = np.concatenate((train_Y1, train_Y2), axis=0)
    test_Y1 = Y1[225:]
    test_Y2 = Y2[225:]
    test_Y = np.concatenate((test_Y1, test_Y2), axis=0)
    # 训练模型
    model.fit(train_X, train_Y)
    # 获取预测结果
    pred = model.predict(test_X)
    # 打印准确率
    print('正确率:{}'.format(model.score(test_X, test_Y)))
    # 参数计算
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(0, len(pred)):
        if pred[i] == test_Y[i] and test_Y[i] == 0:
            TP = TP + 1
        elif pred[i] != test_Y[i] and test_Y[i] == 0:
            FN = FN + 1
        elif pred[i] == test_Y[i] and test_Y[i] == 1:
            TN = TN + 1
        else:
            FP = FP + 1
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    TNR = 1 - FPR
    FNR = 1 - TPR
    FDR = FP / (FP + TP)
    print('TPR(sensitivity):{}\n'
          'TNR(specificity):{}\n'
          'FPR:{}\n'
          'FNR:{}\n'
          'FDR:{}'.format(TPR, TNR, FPR, FNR, FDR))

    # ROC曲线绘制,此处部分代码参考自网络资料
    pred_prob = model.predict_proba(test_X)[:, 1]
    fpr, tpr, threshold = roc_curve(test_Y, pred_prob)
    roc_auc = auc(fpr, tpr)
    print('AUC:{}'.format(roc_auc))

    lw = 2
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def half_classify(dataset):
    # 随机选取五组图片
    ranks = random.sample(range(0, 10), 5)
    X1 = dataset[ranks[0]].copy()
    X2 = dataset[ranks[1]].copy()
    X3 = dataset[ranks[2]].copy()
    X4 = dataset[ranks[3]].copy()
    X5 = dataset[ranks[4]].copy()
    Y = np.empty([5, X1.shape[0]])
    for i in range(0, 5):
        for j in range(0, Y.shape[1]):
            Y[i][j] = ranks[i]
    np.random.shuffle(X1)
    np.random.shuffle(X2)
    np.random.shuffle(X3)
    np.random.shuffle(X4)
    np.random.shuffle(X5)
    train_X1 = X1[:225, :]
    test_X1 = X1[225:, :]
    train_X2 = X2[:225, :]
    test_X2 = X2[225:, :]
    train_X3 = X3[:225, :]
    test_X3 = X3[225:, :]
    train_X4 = X4[:225, :]
    test_X4 = X4[225:, :]
    train_X5 = X5[:225, :]
    test_X5 = X5[225:, :]
    train_X = np.concatenate((train_X1, train_X2, train_X3, train_X4, train_X5), axis=0)
    test_X = np.concatenate((test_X1, test_X2, test_X3, test_X4, test_X5), axis=0)
    train_Y = np.concatenate((Y[0, :225], Y[1, :225], Y[2, :225], Y[3, :225], Y[4, :225]), axis=0)
    test_Y = np.concatenate((Y[0, 225:], Y[1, 225:], Y[2, 225:], Y[3, 225:], Y[4, 225:]), axis=0)
    # 训练模型
    model.fit(train_X, train_Y)
    # 打印准确率
    print('五分类正确率:{}'.format(model.score(test_X, test_Y)))


def full_classify(dataset):
    init = dataset[0].copy()
    np.random.shuffle(init)
    train_X = init[:225, :]
    test_X = init[225:, :]
    for i in range(1, 10):
        newgroup = dataset[i].copy()
        np.random.shuffle(newgroup)
        train = newgroup[:225, :]
        test = newgroup[225:, :]
        train_X = np.concatenate((train_X, train), axis=0)
        test_X = np.concatenate((test_X, test), axis=0)
    train_Y = np.empty(225 * 10, dtype=np.int)
    test_Y = np.empty(75 * 10, dtype=np.int)
    for i in range(0, 225 * 10):
        train_Y[i] = i // 225
    for j in range(0, 75 * 10):
        test_Y[j] = j // 75
    # 训练模型
    model.fit(train_X, train_Y)
    # 打印正确率
    print('十分类训练正确率:{}'.format(model.score(train_X, train_Y)))
    print('十分类测试正确率:{}'.format(model.score(test_X, test_Y)))


if __name__ == '__main__':
    images = []
    # 读入数据
    for i in range(0, 10):
        group = read_images(str(i))
        images.append(group)
    images = np.array(images)
    X_ = np.empty([images.shape[0], images.shape[1], 48 * 48], dtype=np.float32)
    # 图像矩阵展平
    for i in range(0, 10):
        X_[i] = np.reshape(images[i], (len(images[i]), 48 * 48))
    # 两类分类
    two_classify(X_)
    # 五类分类
    half_classify(X_)
    # 全分类
    full_classify(X_)
    # 时间输出,用于6.4题
    # print('time:{}sec'.format(time.clock()))
