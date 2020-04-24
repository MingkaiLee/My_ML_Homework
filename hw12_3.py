import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.utils import shuffle
from sklearn.metrics import normalized_mutual_info_score, silhouette_score


# 聚类数目评估
def kmeans_n_evaluation(x, y):
    Je = []
    NMI = []
    SC = []
    rank = range(2, 11)
    for j in rank:
        model = KMeans(n_clusters=j)
        y_pred = model.fit_predict(x)
        Je.append(model.inertia_)
        NMI.append(normalized_mutual_info_score(y, y_pred, average_method='arithmetic'))
        SC.append(silhouette_score(x, y_pred))
    # 绘图
    fig = plt.figure('KMeans_and_evaluation')
    ax1 = fig.add_subplot(1, 3, 1)
    plt.title('Je')
    plt.plot(rank, Je)
    plt.scatter(rank, Je, c='red', marker='x')
    ax2 = fig.add_subplot(1, 3, 2)
    plt.title('NMI')
    plt.plot(rank, NMI)
    plt.scatter(rank, NMI, c='red', marker='x')
    ax3 = fig.add_subplot(1, 3, 3)
    plt.title('SC')
    plt.plot(rank, SC)
    plt.scatter(rank, SC, c='red', marker='x')
    plt.show()


# 初始化方法评估
def init_method_evaluation(x, y, init_point):
    init_method = ['k-means++', 'random', np.array(init_point)]
    SC = []
    for k in range(0, 3):
        model = KMeans(n_clusters=3, init=init_method[k])
        temp = []
        # 5次实验取平均值
        for j in range(0, 5):
            y_pred = model.fit_predict(x)
            temp.append(normalized_mutual_info_score(y, y_pred, average_method='arithmetic'))
        SC.append(np.mean(temp))
    print('三种初始化方法的NMI:{}'.format(SC))


# 层次聚类实验
def hc_evaluation(x, y):
    dis = ['euclidean', 'manhattan', 'cosine']
    link = ['ward', 'average', 'complete', 'single']
    for k in range(0, len(link)):
        if link[k] == 'ward':
            model = AgglomerativeClustering(n_clusters=3, affinity=dis[0], linkage=link[k])
            y_pred = model.fit_predict(x)
            print('样本距:{},类别距:{},NMI:{}'.format(dis[0], link[k],
                                               normalized_mutual_info_score(y, y_pred, average_method='arithmetic')))
        else:
            for j in range(0, len(dis)):
                model = AgglomerativeClustering(n_clusters=3, affinity=dis[j], linkage=link[k])
                y_pred = model.fit_predict(x)
                print('样本距:{},类别距:{},NMI:{}'.format(dis[j], link[k],
                                                   normalized_mutual_info_score(y, y_pred, average_method='arithmetic'))
                      )


if __name__ == '__main__':
    # 数据读取
    mnist = np.load('./HomeworkData/mnist.npz')
    x_train = np.array(mnist['X_train'])
    y_train = np.array(mnist['y_train'])
    x_test = np.array(mnist['X_test'])
    y_test = np.array(mnist['y_test'])
    number_0 = []
    number_1 = []
    number_2 = []
    # 训练集分类并展平
    for i in range(0, x_train.shape[0]):
        if y_train[i, 0] == 1:
            number_0.append(x_train[i].flatten())
        elif y_train[i, 1] == 1:
            number_1.append(x_train[i].flatten())
        elif y_train[i, 2] == 1:
            number_2.append(x_train[i].flatten())
    number_0 = np.array(number_0)
    label_0 = np.zeros(number_0.shape[0])
    number_1 = np.array(number_1)
    label_1 = np.ones(number_1.shape[0])
    number_2 = np.array(number_2)
    label_2 = np.ones(number_2.shape[0]) * 2
    numbers = np.row_stack((number_0, number_1, number_2))
    labels = np.hstack((label_0, label_1, label_2))
    # 顺序打乱;x_shuffle, y_shuffle为实验用标准数据
    x_shuffle, y_shuffle = shuffle(numbers, labels)
    # 第一问实验
    # kmeans_n_evaluation(x_shuffle, y_shuffle)
    # 第二问实验
    # init_method_evaluation(x_shuffle, y_shuffle, (number_0[0], number_1[0], number_2[0]))
    # 第三问实验
    hc_evaluation(x_shuffle, y_shuffle)
