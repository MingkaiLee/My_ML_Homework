import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from sklearn.datasets import load_iris


def LiPCA(data, clusters=1):
    num = len(data)

    # 样本的去均值化
    data = np.array(data)
    mean = np.mean(data, axis=0)
    print(num)
    for i in range(num):
        data[i] = data[i] - mean
    # 求协方差矩阵
    Cov_data = np.cov(data, rowvar=0)
    # 求特征值和特征向量
    eigVals, eigVecs = np.linalg.eig(np.mat(Cov_data))
    # 将得到的特征值从大到小排序并提取其前clusters个
    sorted_eigranks = np.argsort(-eigVals)
    aim_eigranks = sorted_eigranks[:clusters]
    # 提取对应的特征向量
    aim_eigvecs = eigVecs[:, aim_eigranks]
    # 计算降维后的数据
    PCA_data = data * aim_eigvecs
    return PCA_data


if __name__ == '__main__':
    data = load_iris()
    iris_data = data['data']
    iris_pca2 = LiPCA(iris_data, 2)
    k_means = KMeans(n_clusters=3)
    pred = k_means.fit_predict(iris_pca2)
    colors = ['m', 'g', 'b']
    x_max = 0
    x_min = 0
    y_max = 0
    y_min = 0

    for i in range(len(iris_pca2)):
        if x_max < iris_pca2[i, 0]:
            x_max = iris_pca2[i, 0]
        if x_min > iris_pca2[i, 0]:
            x_min = iris_pca2[i, 0]
        if y_max < iris_pca2[i, 1]:
            y_max = iris_pca2[i, 1]
        if y_min > iris_pca2[i, 1]:
            y_min = iris_pca2[i, 1]

    for i in range(len(iris_pca2)):
        x = iris_pca2[i, 0]
        y = iris_pca2[i, 1]
        label = int(pred[i])
        plt.scatter(x, y, marker='.', c=colors[label])
    plt.title('{}KMeans'.format('Prob4'))
    plt.axis([x_min - 1, x_max + 1, y_min - 1, y_max + 1])
    plt.show()
