import torch
import torch.nn as nn
from torch.nn import functional as F

'''
    Block的各个plane值：
        in_channel:输入block的之前的通道数
        mid_channel:在block中间处理的时候的通道数（这个值是输出维度的1/4）
        mid_channel * self.extension:输出的维度
        downsample:是否下采样，将宽高缩小
'''


class Bottleneck(nn.Module):
    # 每个stage中维度拓展的倍数
    extension = 4

    def __init__(self, in_channel, mid_channel, stride, downsample=None):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channel, mid_channel, stride=stride, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channel)
        self.conv2 = nn.Conv2d(mid_channel, mid_channel, stride=1, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channel)
        self.conv3 = nn.Conv2d(mid_channel, mid_channel * self.extension, stride=1, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channel * self.extension)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        # 残差数据
        residual = x

        # 卷积操作
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))

        # 是否直连（如果是Identity block就是直连；如果是Conv Block就需要对参差边进行卷积，改变通道数和size）
        if (self.downsample != None):
            residual = self.downsample(x)

        # 将残差部分和卷积部分相加
        out = out + residual
        out = self.relu(out)

        return out


class Resnet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        super(Resnet, self).__init__()
        self.in_channel = 64
        self.block = block
        self.layers = layers

        # stem网络层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.in_channel, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        self.stage2 = self.make_layer(self.block, 64, self.layers[0], stride=1)  # 因为在maxpool中stride=2
        self.stage3 = self.make_layer(self.block, 128, self.layers[1], stride=2)
        self.stage4 = self.make_layer(self.block, 256, self.layers[2], stride=2)
        self.stage5 = self.make_layer(self.block, 512, self.layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.extension, num_classes)

    def forward(self, x):
        # stem部分:conv+bn+relu+maxpool
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        # block
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.stage5(out)
        # 分类
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

    def make_layer(self, block, mid_channel, block_num, stride):
        """

        :param block:
        :param mid_channel:
        :param block_num: 重复次数
        :param stride:
        :return:
        """
        block_list = []
        # projection shortcuts are used for increasing dimensions, and other shortcuts are identity
        downsample = None
        if stride != 1 or self.in_channel != mid_channel * block.extension:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, mid_channel * block.extension, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_channel * block.extension)
            )
        # Conv Block
        conv_block = block(self.in_channel, mid_channel, stride=stride, downsample=downsample)
        block_list.append(conv_block)
        self.in_channel = mid_channel * block.extension

        # Identity Block
        for i in range(1, block_num):
            block_list.append(block(self.in_channel, mid_channel, stride=1))

        return nn.Sequential(*block_list)


# 打印网络结构
# resnet = Resnet(Bottleneck, [3, 4, 6, 3])
# print(resnet)

