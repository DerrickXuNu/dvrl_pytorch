# coding=utf-8
"""
Basic Backbone for Covid image classfication
"""

from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d, AdaptiveAvgPool2d


class BasicCNN(Module):
    def __init__(self):
        super(BasicCNN, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer 256
            Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(128),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer 128
            Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(256),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer 64
            Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer 32
            Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(1024),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer 16
            Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(2048),
            ReLU(inplace=True),
            AdaptiveAvgPool2d((1, 1)),
        )

        self.linear_layers = Sequential(
            Linear(2048, 1024),
            ReLU(inplace=True),
            Linear(1024, 2)
        )

    # Defining the forward pass
    def forward(self, x, last_layer=True):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        if last_layer:
            x = self.linear_layers(x)
        return x
