import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def testnet(model, test_loader):
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
    print("test loss:" + ' ' + str(eval_loss / len(test_loader)))
    print('test_accuracy:' + ' ' + str(eval_acc / len(test_loader)))
