# -*-coding:utf-8 -*-

"""
# File       : model.py
# Time       ：2023/4/29 13:56
# Author     ：
# Email      ：
# Description：
"""
import torch
from torch.nn import Conv1d, ReLU, Linear, Module, Sequential, AdaptiveAvgPool1d, CrossEntropyLoss, Dropout


class CnnNet(Module):
    def __init__(self, object='binary', drop_out=0.2):
        super(CnnNet, self).__init__()
 #        self.weight = torch.tensor([1.0,
 # 10.950545921644187,
 # 10.850493204236557,
 # 10.833257692656803,
 # 4.470790957277444,
 # 7.994775094617678,
 # 11.361256544502618,
 # 14.884329986905282,
 # 36.796670263604135,
 # 10.82147066823828,
 # 22.974013474494708,
 # 26.877603873437675,
 # 56.45695364238411,
 # 612.0512820512821,
 # 1705.0])
        self.object = object
        self.loss_func = CrossEntropyLoss()
        self.conv = Sequential(
            Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            Dropout(p=drop_out),
            ReLU(),
            Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            Dropout(p=drop_out),
            ReLU(),
            Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            Dropout(p=drop_out),
            ReLU(),
        )
        self.avg_pool = AdaptiveAvgPool1d(output_size=1)
        self.linear = Sequential(
            Linear(in_features=16, out_features=120),
            Dropout(p=drop_out),
            ReLU(),
            Linear(in_features=120, out_features=30),
            Dropout(p=drop_out),
            ReLU()
        )
        if object == 'binary':
            self.out_layer = Linear(in_features=30, out_features=2)
        elif object == 'multi':
            self.out_layer = Linear(in_features=30, out_features=15)

    def forward(self, x):
        out = self.conv(x)
        out = self.avg_pool(out)
        out = out.reshape((-1, 16))
        out = self.linear(out)
        out = self.out_layer(out)
        return out

    def predict(self, x):
        out = self.forward(x)
        return torch.argmax(out, dim=1)

    def loss(self, pred, target):
        return self.loss_func(pred, target)

