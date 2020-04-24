import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics


if __name__ == '__main__':
    # 数据读取预处理
    data = pd.read_table('./HomeworkData/spambase/spambase.data', sep=',', header=None)
    data = np.array(data)
    features = np.array(data[:, :57], dtype=np.float32)
    labels = np.array(data[:, 57], dtype=np.int32)
    # 数据集划分
    X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=1000)
    # 模型训练
    model = GaussianNB()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    # 打印正确率
    print('测试集正确率:{}'.format(model.score(X_test, Y_test)))
    # 打印混淆矩阵
    label = list(set(labels))
    print('混淆矩阵:')
    confusion_matrix = metrics.confusion_matrix(Y_test, Y_pred, labels=label)
    print(confusion_matrix)
    # 计算混淆矩阵
    TP = confusion_matrix[0][0]
    FP = confusion_matrix[0][1]
    FN = confusion_matrix[1][0]
    TN = confusion_matrix[1][1]
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    TNR = 1 - FPR
    FNR = 1 - TPR
    print('TPR:{}\n'
          'TNR:{}\n'
          'FPR:{}\n'
          'FNR:{}'.format(TPR, TNR, FPR, FNR))
    # 绘制roc曲线并计算AUC
    pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(Y_test, pred_prob)
    roc_auc = metrics.auc(fpr, tpr)
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
