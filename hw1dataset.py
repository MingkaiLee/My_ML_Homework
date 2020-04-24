import torch.utils.data as Data
import numpy as np
import torch
from PIL import Image


class TrainDataLoader(Data.Dataset):
    def __init__(self, train_npy, train_label, transform=None):
        super(TrainDataLoader, self).__init__()
        # 对输入作归一化
        self.train_data = np.empty([train_npy.shape[0], 28, 28], dtype=np.float32)
        for i in range(train_npy.shape[0]):
            self.train_data[i] = np.reshape(train_npy[i], (28, 28))
            self.train_data[i] = self.train_data[i] / 255

        self.train_label = train_label

        self.transform = transform

    def __getitem__(self, item):
        img, target = self.train_data[item], self.train_label[item]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.train_data)


class TestDataLoader(Data.Dataset):
    def __init__(self, test_npy, transform=None):
        super(TestDataLoader, self).__init__()
        # 对输入作归一化
        self.test_data = np.empty([test_npy.shape[0], 28, 28], dtype=np.float32)
        for i in range(test_npy.shape[0]):
            self.test_data[i] = np.reshape(test_npy[i], (28, 28))
            self.test_data[i] = self.test_data[i] / 255

        self.test_label = np.zeros(test_npy.shape[0], dtype=int)

        self.transform = transform

    def __getitem__(self, item):
        img, target = self.test_data[item], self.test_label[item]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.test_data)
