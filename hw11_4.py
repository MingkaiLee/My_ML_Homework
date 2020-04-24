import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier


# 训练集降维可视化实验
def mnist_vision(x, y):
    number_0 = []
    number_8 = []
    # 训练集分类并展平
    for i in range(0, x.shape[0]):
        if y[i, 0] == 1:
            number_0.append(x[i].flatten())
        elif y[i, 8] == 1:
            number_8.append(x[i].flatten())
    # PCA降维
    pca = PCA(n_components=2)
    pca_0 = pca.fit_transform(number_0)
    pca_8 = pca.fit_transform(number_8)
    fig = plt.figure('result')
    ax1 = fig.add_subplot(2, 2, 1)
    plt.title('PCA')
    plt.scatter(pca_0[:, 0], pca_0[:, 1], c='blue', marker='^', s=1)
    plt.scatter(pca_8[:, 0], pca_8[:, 1], c='red', marker='x', s=1)
    # ISOMAP降维
    isomap = Isomap(n_neighbors=5, n_components=2, max_iter=100)
    iso_0 = isomap.fit_transform(number_0)
    iso_8 = isomap.fit_transform(number_8)
    ax2 = fig.add_subplot(2, 2, 2)
    plt.title('ISOMAP')
    plt.scatter(iso_0[:, 0], iso_0[:, 1], c='blue', marker='^', s=1)
    plt.scatter(iso_8[:, 0], iso_8[:, 1], c='red', marker='x', s=1)
    # LLE降维
    lle = LocallyLinearEmbedding()
    lle_0 = lle.fit_transform(number_0)
    lle_8 = lle.fit_transform(number_8)
    ax3 = fig.add_subplot(2, 2, 3)
    plt.title('LLE')
    plt.scatter(lle_0[:, 0], lle_0[:, 1], c='blue', marker='^', s=1)
    plt.scatter(lle_8[:, 0], lle_8[:, 1], c='red', marker='x', s=1)
    # TSNE降维
    tsne = TSNE(n_components=2)
    tsne_0 = tsne.fit_transform(number_0)
    tsne_8 = tsne.fit_transform(number_8)
    ax4 = fig.add_subplot(2, 2, 4)
    plt.title('TSNE')
    plt.scatter(tsne_0[:, 0], tsne_0[:, 1], c='blue', marker='^', s=1)
    plt.scatter(tsne_8[:, 0], tsne_8[:, 1], c='red', marker='x', s=1)
    plt.show()


# 降维与训练
def dr_train(x_train, y_train, x_test, y_test):
    # 使用多层感知器作为分类器
    model = MLPClassifier(hidden_layer_sizes=(100, 100, 50), solver='lbfgs', activation='relu', learning_rate_init=0.1, max_iter=1000)
    number_0 = []
    number_8 = []
    # 训练集分类并展平
    for i in range(0, x_train.shape[0]):
        if y_train[i, 0] == 1:
            number_0.append(x_train[i].flatten())
        elif y_train[i, 8] == 1:
            number_8.append(x_train[i].flatten())
    # 给出训练集标签
    label_0 = np.zeros((len(number_0), 1))
    label_8 = np.ones((len(number_8), 1))
    train_label = np.row_stack((label_0, label_8)).flatten()
    # 训练集合并
    train_data = np.row_stack((np.array(number_0), np.array(number_8)))
    # 测试集分类并展平
    for i in range(0, x_test.shape[0]):
        if y_test[i, 0] == 1:
            number_0.append(x_test[i].flatten())
        elif y_test[i, 8] == 1:
            number_8.append(x_test[i].flatten())
    # 给出训练集标签
    label_0 = np.zeros((len(number_0), 1))
    label_8 = np.ones((len(number_8), 1))
    test_label = np.row_stack((label_0, label_8)).flatten()
    # 训练集合并
    test_data = np.row_stack((np.array(number_0), np.array(number_8)))
    dims = [1, 10, 20, 50, 100, 300]
    for i in range(0, len(dims)):
        pca = PCA(n_components=dims[i])
        d_train = pca.fit_transform(train_data)
        d_test = pca.fit_transform(test_data)
        model.fit(d_train, train_label)
        print('数据降维到{}维后的测试集正确率:{}'.format(dims[i], model.score(d_test, test_label)))
    model.fit(train_data, train_label)
    print('不作降维的测试集正确率:{}'.format(model.score(test_data, test_label)))


if __name__ == '__main__':
    # 数据读取
    mnist = np.load('./HomeworkData/mnist.npz')
    x_train = np.array(mnist['X_train'])
    y_train = np.array(mnist['y_train'])
    x_test = np.array(mnist['X_test'])
    y_test = np.array(mnist['y_test'])
    # mnist_vision(x_train, y_train)
    dr_train(x_train, y_train, x_test, y_test)
