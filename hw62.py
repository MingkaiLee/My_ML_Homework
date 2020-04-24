import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from torchvision import datasets, transforms
import torch.utils.data as Data
import numpy as np

if __name__ == "__main__":
    Batch_size = 300
    mnist_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=False)
    mnist_loader = Data.DataLoader(dataset=mnist_dataset, batch_size=Batch_size, shuffle=True)
    # 先取Batch_size个数据出来
    for i, (image, label) in enumerate(mnist_loader):
        break
    # 转换为np.array并将图片展平
    image = image.numpy()
    data = np.empty([Batch_size, 28 * 28], dtype=np.int)
    for i in range(Batch_size):
        data[i] = np.reshape(image[i], (28 * 28))
    # label = label.numpy()
    # print(label)
    # 使用TSNE降维到2维
    tsne = TSNE(n_components=2)
    data = tsne.fit_transform(data)
    x_max = 0
    x_min = 0
    y_max = 0
    y_min = 0

    for i in range(Batch_size):
        if x_max < data[i, 0]:
            x_max = data[i, 0]
        if x_min > data[i, 0]:
            x_min = data[i, 0]
        if y_max < data[i, 1]:
            y_max = data[i, 1]
        if y_min > data[i, 1]:
            y_min = data[i, 1]
    # print(data)
    # 进行kmeans聚类
    k_means = KMeans(n_clusters=10)
    pred = k_means.fit_predict(data)
    print(pred)
    cents = k_means.cluster_centers_
    # print(cents)
    # 画出聚类结果
    colors = ['#FF69B4', '#000000', '#FF4500', '#EE82EE', '#FF6347', '#00008B', '#20B2AA', '#6495ED', '#B22222',
              '#FF7F50']
    for i in range(Batch_size):
        label = int(pred[i])
        x = data[i, 0]
        y = data[i, 1]
        plt.text(x, y, str(label), color=colors[label], fontsize=8)
    plt.title('{}KMeans'.format('Prob2'))
    plt.axis([x_min - 1, x_max + 1, y_min - 1, y_max + 1])
    plt.show()
