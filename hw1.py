from hw1dataset import TrainDataLoader, TestDataLoader
from hw1alex import Linet, Linet2, Linet3
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import csv
import copy
import matplotlib.pyplot as plt
import torch.utils.data as Data
from PIL import Image

if __name__ == '__main__':
    BATCH_SIZE = 32
    # 读入numpy数据
    test_npy = np.load('./data/test.npy')
    train_npy = np.load('./data/train.npy')
    # 读入训练数据标签并转为numpyarray,并只提取值
    train_label = pd.read_csv('./data/train.csv')
    train_label = np.array(train_label)
    train_label = train_label[:, 1]
    # transform
    Mytransform = transforms.ToTensor()

    # 加载数据集
    train_dataset = TrainDataLoader(train_npy, train_label, transform=Mytransform)
    test_dataset = TestDataLoader(test_npy, transform=Mytransform)
    # train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [24000, 6000])
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 学习率调整
    lrh = 0.1
    lrm = 0.05
    lrl = 0.001
    model = Linet3()
    loss = nn.CrossEntropyLoss()  # 损失函数选择，交叉熵函数
    optimizerh = optim.SGD(model.parameters(), lr=lrh)
    optimizerm = optim.SGD(model.parameters(), lr=lrm)
    optimizerl = optim.SGD(model.parameters(), lr=lrl)

    num_epochs = 21

    eval_losses = []
    eval_accs = []
    losses = []
    acces = []


    for echo in range(num_epochs):
        train_loss = 0  # 定义训练损失
        train_acc = 0  # 定义训练准确度
        model.train()  # 将网络转化为训练模式
        for i, (X, label) in enumerate(train_loader):  # 使用枚举函数遍历train_loader
            # X = X.view(-1,784)       #X:[64,1,28,28] -> [64,784]将X向量展平
            X = Variable(X)  # 转化数据类型
            X = X.float()
            label = Variable(label)
            out = model(X)  # 正向传播
            lossvalue = loss(out, label)  # 求损失值
            """
            if echo > 1 and eval_accs[echo - 1] > 0.8 and eval_accs[echo - 1] < 0.84:
                optimizerm.zero_grad()  # 优化器梯度归零
                lossvalue.backward()  # 反向转播，刷新梯度值
                optimizerm.step()  # 优化器运行一步，注意optimizer搜集的是model的参数
            elif echo > 1 and eval_accs[echo - 1] > 0.84:
                optimizerl.zero_grad()
                lossvalue.backward()
                optimizerl.step()
            else:
                optimizerh.zero_grad()
                lossvalue.backward()
                optimizerh.step()
            """
            if echo < 7:
                optimizerh.zero_grad()
                lossvalue.backward()
                optimizerh.step()
            elif echo < 14:
                optimizerm.zero_grad()
                lossvalue.backward()
                optimizerm.step()
            else:
                optimizerl.zero_grad()
                lossvalue.backward()
                optimizerl.step()

            # 计算损失
            train_loss += float(lossvalue)
            # 计算精确度
            _, pred = out.max(1)
            num_correct = (pred == label).sum()
            acc = int(num_correct) / X.shape[0]
            train_acc += acc

        losses.append(train_loss / len(train_loader))
        acces.append(train_acc / len(train_loader))
        print("echo:" + ' ' + str(echo))
        print("lose:" + ' ' + str(train_loss / len(train_loader)))
        print("accuracy:" + ' ' + str(train_acc / len(train_loader)))
        print("time: " + str(time.clock()))
        """
        eval_loss = 0
        eval_acc = 0
        model.eval()  # 模型转化为评估模式
        for X, label in test_loader:
            # X = X.view(-1,784)
            X = Variable(X)
            label = Variable(label)
            testout = model(X)
            testloss = loss(testout, label)
            # 计算损失值
            eval_loss += float(testloss)

            _, pred = testout.max(1)
            num_correct = (pred == label).sum()
            acc = int(num_correct) / X.shape[0]
            eval_acc += acc

        eval_losses.append(eval_loss / len(test_loader))
        eval_accs.append(eval_acc / len(test_loader))
        print("test loss:" + ' ' + str(eval_loss / len(test_loader)))
        print('test_accuracy:' + ' ' + str(eval_acc / len(test_loader)))

    max_accuracy = max(eval_accs)
    suitableecho = eval_accs.index(max(eval_accs))
    print(str(max_accuracy) + ' ' + str(suitableecho))

    """
    pred_all = None
    model.eval()
    for i, X in enumerate(test_loader):
        X = Variable(X[0])
        res = model(X)
        _, pred = res.max(1)
        if pred_all is None:
            pred_all = torch.cat([pred])
        else:
            pred_all = torch.cat([pred_all, pred])

    y_pred = pred_all.cpu().detach().numpy()
    headers = ['image_id', 'label']
    x_id = np.linspace(0, 4999, 5000).astype(np.int64)
    rows = list(zip(x_id, y_pred))
    with open('./data/test{}.csv'.format(2), 'w+', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(rows)

