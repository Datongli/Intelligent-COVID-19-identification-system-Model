import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential


class BasicBlock(layers.layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(filter_num, (3, 3), stride=stride, padding='same')
        self.bn1 = layers.BatchNormalization()  # 批量处理规范化
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num, (3, 3), stride=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), stride=stride))
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(inputs)

        output = layers.add([out, identity])
        output = tf.nn.relu(output)

        return output


class ResNet(keras.Model):
    # layer_dims:一个数组，表示每一个resblock里有几个basicblock  例：layer_dims = [2,2,2,2]
    # num_classes:预计最后可以分多少类
    def __init__(self, layer_dims, num_classes=100):
        super(ResNet, self).__init__()

        # 创建ResNet的第一层,即为一个预处理的节点
        self.stem = Sequential([layers.Conv2D(64, (3, 3), strides=(1, 1)),
                                layers.BatchNormalization(),
                                layers.Activation('relu'),
                                layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')
                                ])
        self.layer1 = self.bulid_resblock(64, layer_dims[0])
        # 后面三个有降维的功能
        self.layer2 = self.bulid_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.bulid_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.bulid_resblock(512, layer_dims[3], stride=2)

        # 例如，可以认为前面输出为[b,512,h,w]
        self.avgpool = layers.GlobalAvgPool2D()  # 将输出的参数长宽平均（可以不加）
        self.fc = layers.Dense(num_classes)  # 全连接层，判断类别

    def call(self, inputs, training=None):
        x = self.stem(inputs)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x

    def bulid_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        # 可能会进行下采样
        res_blocks.add(BasicBlock(filter_num, stride))

        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))

        return res_blocks


def resnet18():
    return ResNet([2, 2, 2, 2])


def resnet34():
    return ResNet([3, 4, 6, 3])
