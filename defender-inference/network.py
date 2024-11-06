import functools
import torch.nn as nn
import torch.nn.functional as F
import math


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, bn=False, stride=1):
        super(ResBlock, self).__init__()
        self.bn = bn
        if bn:
            self.bn0 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        if bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.shortcut = nn.Sequential()
        if stride > 1:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn0(x))
        else:
            out = F.relu(x)

        if self.bn:
            out = F.relu(self.bn1(self.conv1(out)))
        else:
            out = F.relu(self.conv1(out))

        out = self.conv2(out)
        out += self.shortcut(x)
        return out


def client_resnet(level, channels=3):
    net = []

    net += [nn.Conv2d(channels, 64, 3, 1, 1)]
    net += [nn.BatchNorm2d(64)]
    net += [nn.ReLU()]
    net += [nn.MaxPool2d(2)]
    net += [ResBlock(64, 64)]

    if level == 1:
        return nn.Sequential(*net)

    net += [ResBlock(64, 128, stride=2)]

    if level == 2:
        return nn.Sequential(*net)

    net += [ResBlock(128, 128)]

    if level == 3:
        return nn.Sequential(*net)

    net += [ResBlock(128, 256, stride=2)]

    if level <= 4:
        return nn.Sequential(*net)
    else:
        raise Exception('No level %d' % level)


def server_pilot(level, channels=3):
    net = []
    net += [nn.Conv2d(channels, 64, 3, 2, 1)]

    if level == 1:
        net += [nn.Conv2d(64, 64, 3, 1, 1)]
        return nn.Sequential(*net)

    net += [nn.Conv2d(64, 128, 3, 2, 1)]

    if level <= 3:
        net += [nn.Conv2d(128, 128, 3, 1, 1)]
        return nn.Sequential(*net)

    net += [nn.Conv2d(128, 256, 3, 2, 1)]

    if level <= 4:
        net += [nn.Conv2d(256, 256, 3, 1, 1)]
        return nn.Sequential(*net)
    else:
        raise Exception('No level %d' % level)


def server_decoder(level, channels=3):
    net = []
    input_shape = set_input_shape_server(level)
    net += [nn.ConvTranspose2d(input_shape, 256, 3, 2, 1, output_padding=1)]

    if level == 1:
        net += [nn.Conv2d(256, channels, 3, 1, 1)]
        net += [nn.Tanh()]
        return nn.Sequential(*net)

    net += [nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1)]

    if level <= 3:
        net += [nn.Conv2d(128, channels, 3, 1, 1)]
        net += [nn.Tanh()]
        return nn.Sequential(*net)

    net += [nn.ConvTranspose2d(128, channels, 3, 2, 1, output_padding=1)]
    net += [nn.Tanh()]
    return nn.Sequential(*net)


def discriminator(level):
    net = []
    input_shape = set_input_shape_server(level)
    if level == 1:
        net += [nn.Conv2d(input_shape, 128, 3, 2, 1)]
        net += [nn.ReLU()]
        net += [nn.Conv2d(128, 256, 3, 2, 1)]
    elif level <= 3:
        net += [nn.Conv2d(input_shape, 256, 3, 2, 1)]
    elif level <= 4:
        net += [nn.Conv2d(input_shape, 256, 3, 1, 1)]

    bn = False

    net += [ResBlock(256, 256, bn=bn)]
    net += [ResBlock(256, 256, bn=bn)]
    net += [ResBlock(256, 256, bn=bn)]
    net += [ResBlock(256, 256, bn=bn)]
    net += [ResBlock(256, 256, bn=bn)]
    net += [ResBlock(256, 256, bn=bn)]

    net += [nn.Conv2d(256, 256, 3, 2, 1)]
    net += [nn.Flatten()]
    net += [nn.Linear(1024, 1)]
    return nn.Sequential(*net)


def server_dis(level):
    net = []
    input_shape = set_input_shape_server(level)
    if level == 1:
        net += [nn.Conv2d(input_shape, 128, 3, 2, 1)]
        net += [nn.ReLU()]
        net += [nn.Conv2d(128, 256, 3, 2, 1)]
    elif level <= 3:
        net += [nn.Conv2d(input_shape, 256, 3, 2, 1)]
    elif level <= 4:
        net += [nn.Conv2d(input_shape, 256, 3, 1, 1)]

    bn = False

    net += [ResBlock(256, 256, bn=bn)]
    net += [ResBlock(256, 256, bn=bn)]

    net += [nn.Conv2d(256, 256, 3, 2, 1)]
    net += [nn.Flatten()]
    net += [nn.Linear(1024, 2)]
    return nn.Sequential(*net)


def server_tasker(level):
    net = []
    input_shape = set_input_shape_server(level)
    if level == 1:
        net += [nn.Conv2d(input_shape, 128, 3, 2, 1)]
        net += [nn.ReLU()]
        net += [nn.Conv2d(128, 256, 3, 2, 1)]
    elif level <= 3:
        net += [nn.Conv2d(input_shape, 256, 3, 2, 1)]
    elif level <= 4:
        net += [nn.Conv2d(input_shape, 256, 3, 1, 1)]

    bn = False
    net += [ResBlock(256, 256, bn=bn)]

    net += [nn.Conv2d(256, 256, 3, 2, 1)]
    net += [nn.Flatten()]
    net += [nn.Linear(1024, 10)]
    return nn.Sequential(*net)


def set_input_shape_server(level):
    shape = [64, 128, 128, 256]
    return shape[level-1]

def set_input_shape_size(level):
    size = [16, 8, 8, 4]
    return size[level-1]
