import torch
import torch.nn as nn
import torch.nn.functional as F


class Linet(nn.Module):
    def __init__(self):
        super(Linet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(40, 80, kernel_size=3, stride=1)

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(1280, 600),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(600, 600),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(600, 10)
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 80 * 4 * 4)
        x = self.dense(x)
        return x


class Linet2(nn.Module):
    def __init__(self):
        super(Linet2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=0),  # output=13*13*32
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=96, kernel_size=3, stride=2, padding=0),  # output=6*6*96
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # output=3*3*96
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=180, kernel_size=3, stride=1, padding=1),  # output=3*3*180
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=180, out_channels=100, kernel_size=2, stride=1, padding=1),  # output=4*4*100
            nn.ReLU()
        )

        self.dense = nn.Sequential(
            nn.Linear(1600, 800),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(800, 400),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(200, 50),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(50, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 1600)
        x = self.dense(x)
        return x


class Linet3(nn.Module):
    def __init__(self):
        super(Linet3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=30, kernel_size=7, stride=2, padding=2),  # output=13*13*30
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=90, kernel_size=2, stride=1, padding=0),  # output=12*12*90
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # output=6*6*90
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=90, out_channels=180, kernel_size=3, stride=1, padding=0),  # output=4*4*180
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # output=2*2*180
        )
        self.dense = nn.Sequential(
            nn.Linear(720, 720),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(720, 360),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(360, 180),
            nn.ReLU(),
            nn.Dropout(0, 5),
            nn.Linear(180, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 720)
        x = self.dense(x)
        return x


class Resnet34(nn.Module):
    def __init__(self):
        super(Resnet34, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 24, 7, 1, 3),  # output=112*112*24
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output=56*56*24
            nn.BatchNorm2d(24)
        )
        self.conv21 = nn.Sequential(
            nn.Conv2d(24, 24, 3, 1, 1),  # output=56*56*24
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 24, 3, 1, 1),  # output=56*56*24
            nn.ReLU(),
            nn.BatchNorm2d(24)
        )
        self.conv21fb = nn.Sequential(
            nn.Conv2d(24, 24, 1, 1, 0),
            nn.BatchNorm2d(24)
        )
        self.conv22 = nn.Sequential(
            nn.Conv2d(24, 24, 3, 1, 1),  # output=56*56*24
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 24, 3, 1, 1),  # output=56*56*24
            nn.ReLU(),
            nn.BatchNorm2d(24)
        )
        # 上一残差块群转到下一残差块群先经31, 后经3层32
        self.conv31 = nn.Sequential(
            nn.Conv2d(24, 48, 3, 1, 1),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 48, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output=28*28*48
            nn.BatchNorm2d(48)
        )
        self.conv31fb = nn.Sequential(
            nn.Conv2d(24, 48, 1, 1, 0),
            nn.BatchNorm2d(48)
        )
        self.conv32 = nn.Sequential(
            nn.Conv2d(48, 48, 3, 1, 1),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 48, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(48)
        )
        # 上一残差块群传到下一残差块群先经41, 后经5层42
        self.conv41 = nn.Sequential(
            nn.Conv2d(48, 96, 3, 1, 1),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 96, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output=14*14*96
            nn.BatchNorm2d(96)
        )
        self.conv41fb = nn.Sequential(
            nn.Conv2d(48, 96, 1, 1, 0),
            nn.BatchNorm2d(96)
        )
        self.Conv42 = nn.Sequential(
            nn.Conv2d(96, 96, 3, 1, 1),
            nn.BatchNorm2d(96),
            nn.Conv2d(96, 96, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(96)
        )
        # 上一残差块群传到下一残差块群先经51, 后经2层52
        self.conv51 = nn.Sequential(
            nn.Conv2d(96, 192, 3, 1, 1),
            nn.BatchNorm2d(192),
            nn.Conv2d(192, 192, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output=7*7*192
            nn.BatchNorm2d(192)
        )
        self.conv51fb = nn.Sequential(
            nn.Conv2d(96, 192, 1, 1, 0),
            nn.BatchNorm2d(192)
        )
        self.conv52 = nn.Sequential(
            nn.Conv2d(192, 192, 3, 1, 1),
            nn.BatchNorm2d(192),
            nn.Conv2d(192, 192, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(192)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(7, 7)
        self.linear = nn.Linear(192*7*7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv21(x) + self.conv21fb(x)
        x = F.relu(x)
        for i in range(2):
            x = self.conv22(x) + x
            x = F.relu(x)
        x = self.conv31(x) + self.conv31fb(x)
        x = F.relu(x)
        for i in range(3):
            x = self.conv32(x) + x
            x = F.relu(x)
        x = self.conv41(x) + self.conv41fb(x)
        x = F.relu(x)
        for i in range(5):
            x = self.conv42(x) + x
            x = F.relu(x)
        x = self.conv51(x) + self.conv51fb(x)
        x = F.relu(x)
        for i in range(2):
            x = self.conv52(x) + x
            x = F.relu(x)
        x = self.avgpool(x)
        x = x.view(-1, 7*7*192)
        x = self.linear(x)
        return x


