"""
此文件用于改善交叉熵的计算，加上不平衡的权重
同时搞清楚一下计算的过程
更改了写入txt的程序
"""

import torch, gc
import imageio.v2 as imageio
from torch.nn import init
from torch.optim.lr_scheduler import CosineAnnealingLR
import datetime
import itertools
import os
import random
# import cv2
import matplotlib.pyplot as plt
import numpy as np
# PaddyDataSet
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
# from pytorchtools import EarlyStopping
import GhostNet
import efficientnet
import GhostNet_res
import Covnet
import Covnet_2
import Covnet_3
import ResNet

# 从自定义的ResNet.py文件中导入resnet50这个函数
# import ResNet

negative = 'cat'
positive = 'dog'
# negative = 'negative'
# positive = 'positive'

# 工作目录
work_path = r"D:\学习\大创\data\训练数据集\model"
# 数据集文件夹位置
filepath = r"D:\学习\大创\data\训练数据集\data\cat_vs_dog"
# 权重文件位置
# weightpath = "D:/学习/大创/data/训练数据集/data/path/resnet50.pth"

paddy_labels = {negative: 0,
                positive: 1}


class PaddyDataSet(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        数据集
        """
        self.label_name = {negative: 0, positive: 1}
        # data_info 存储所有图片路径和标签, 在DataLoader中通过index读取样本
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform
        self.temp = np.zeros((224, 224))

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')
        # print(img.size)
        if img.size == self.temp.shape:
            img = img.resize((224, 224))
            # print(img.size)
        if self.transform is not None:
            # 加入给每一张图片分别做归一化的操作
            img_mean, img_std = self.count_mean_std(path_img)
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=img_mean, std=img_std)
            ])
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data_info)

    def count_mean_std(self, filepath):
        """
        :param filepath: 传入的单张图片的地址
        :return: 返回计算的方差和均值
        """
        R_channel = 0
        G_channel = 0
        B_channel = 0

        img = imageio.imread(filepath) / 255.0
        R_channel = R_channel + np.sum(img[:, :, 0])
        G_channel = G_channel + np.sum(img[:, :, 1])
        B_channel = B_channel + np.sum(img[:, :, 2])

        num = 224 * 224  # 这里（512,512）是每幅图片的大小，所有图片尺寸都一样
        R_mean = R_channel / num
        G_mean = G_channel / num
        B_mean = B_channel / num

        R_channel = 0
        G_channel = 0
        B_channel = 0

        img = imageio.imread(filepath) / 255.0
        R_channel = R_channel + np.sum((img[:, :, 0] - R_mean) ** 2)
        G_channel = G_channel + np.sum((img[:, :, 1] - G_mean) ** 2)
        B_channel = B_channel + np.sum((img[:, :, 2] - B_mean) ** 2)

        R_var = np.sqrt(R_channel / num)
        G_var = np.sqrt(G_channel / num)
        B_var = np.sqrt(B_channel / num)
        # print("R_mean is %f, G_mean is %f, B_mean is %f" % (R_mean, G_mean, B_mean))
        # print("R_var is %f, G_var is %f, B_var is %f" % (R_var, G_var, B_var))
        mean = [R_mean, G_mean, B_mean]
        std = [R_var, G_var, B_var]
        return mean, std


    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))

                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    # print(sub_dir)
                    label = paddy_labels[sub_dir]
                    data_info.append((path_img, int(label)))

        return data_info


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


def init_weights(layer):
    """
    参数初始化设置使用
    :param layer:
    :return:
    """
    # 如果为卷积层，使用 He initialization 方法正态分布生成值，生成随机数填充张量
    if type(layer) == nn.Conv2d:
        # nn.init.normal_(layer.weight, mean=0, std=0.5)
        nn.init.kaiming_normal_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
    # 如果为全连接层，权重使用均匀分布初始化，偏置初始化为0.1
    elif type(layer) == nn.Linear:
        nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
        nn.init.constant_(layer.bias, 0.1)


class Focal_Loss():
    """
    二分类Focal Loss
    """
    def __init__(self, alpha=0.25, gamma=1):
        super(Focal_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        preds:sigmoid的输出结果
        labels：标签
        """
        eps = 1e-7
        loss = []
        for i in range(len(labels)):
            loss_1 = -1 * (1 - self.alpha) * torch.pow((1 - preds[i][1]), self.gamma) * torch.log(preds[i][1] + eps) * labels[i]
            loss_0 = -1 * self.alpha * torch.pow(preds[i][1], self.gamma) * torch.log(1 - preds[i][1] + eps) * (1 - labels[i])
            loss.append(loss_0 + loss_1)
        loss = torch.as_tensor(loss)
        return torch.mean(loss)


# -------------------------------------------------- #
# （0）参数设置
# -------------------------------------------------- #
batch_size = 32  # 每个step训练batch_size张图片
epochs = 32  # 共训练epochs次
k = 5  # k折交叉验证
net_name = "resnet18"
dropout_num_1 = 0.0
dropout_num_2 = 0.0
learning_rate = 0.001
pre_score_k = []
labels_k = []
# wd：正则化惩罚的参数
wd = 0
# wd = None
# stop_epoch: 早停的批量数
stop_epoch = 5


# -------------------------------------------------- #
# （1）文件配置
# -------------------------------------------------- #
# 计算图片的总数量
negative_path = filepath + "\\" + negative
positive_path = filepath + "\\" + positive
all_photo_num = len(os.listdir(negative_path))
all_photo_num += len(os.listdir(positive_path))
# 计算两种样本的比例alpha = p:(n+p)
negative_num = len(os.listdir(negative_path))
positive_num = len(os.listdir(positive_path))
alpha = positive_num / (positive_num + negative_num)
print("alpha:{}".format(alpha))

# train_val_num:训练和验证的比例= (train_val_num -1) :1
train_val_num = 4
train_num = all_photo_num * (k - 1) / k * (train_val_num - 1) / train_val_num

# 显示一下文件夹的名称
dir_count = filepath.rfind('\\') + 1
dir_path = filepath[dir_count:]
print(dir_path)
# 创建权重的文件夹
pth_flodr = os.path.join(work_path, 'pth', dir_path)
cd = os.path.exists(pth_flodr)
if cd:
    print("权重保存文件夹已存在")
else:
    print("创建权重保存文件夹")
    os.mkdir(pth_flodr)
# 权重保存文件夹路径
savepath = pth_flodr

# 获取GPU设备
if torch.cuda.is_available():  # 如果有GPU就用，没有就用CPU
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# -------------------------------------------------- #
# （2）构造数据集
# -------------------------------------------------- #

# 读取数据集后再进行划分
data_dir = filepath
# 实例化一个对象
data_0 = PaddyDataSet(data_dir=data_dir,
                      transform=transforms.Compose([
                          # 将输入图像大小调整为224*224
                          transforms.Resize((224, 224)),
                          # # 数据增强，随机水平翻转
                          # transforms.RandomHorizontalFlip(),
                          # 数据变成tensor类型，像素值归一化，调整维度[h,w,c]==>[c,h,w]
                          transforms.ToTensor(),
                          # transforms.Normalize(mean=image_mean, std=image_std)
                      ]))



# # 先划分成 5份
kf = KFold(n_splits=k, shuffle=True, random_state=34)
# 初始化混淆矩阵
cnf_matrix = np.zeros([2, 2])
step_num = int(train_num / batch_size)
# classes = data.classes


# -------------------------------------------------- #
# （3）加载模型
# -------------------------------------------------- #


# -------------------------------------------------- #
# （4）网络训练
# -------------------------------------------------- #

# 这一段代码是为了过程化训练进程做准备
line_num = step_num / 20.0
num_1 = line_num

dir_count = filepath.rfind('\\') + 1
dir_path = filepath[dir_count:]
print(dir_path)
# 判断文件夹是否存在
photo_folder = os.path.join(work_path, 'photo', dir_path)
cd = os.path.exists(photo_folder)
if cd:
    print("图片保存文件夹已存在")
else:
    print("创建图片保存文件夹")
    os.mkdir(photo_folder)
# 加时间戳
nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


"""
模型的训练
"""
k_num = 0
best_acc_all = 0
for train_val_both, test_index in kf.split(data_0):
    # 按照train:val:test=6:2:2的比例划分
    num_all = len(train_val_both)
    num_val = int(num_all / train_val_num)
    num_train = num_all - num_val
    # 准备train 和 val 二者的标签
    # train_index = train_val_both[:num_train]
    # val_index = train_val_both[:num_val]
    train_index = []
    train_val_both = train_val_both.tolist()
    val_index = random.sample(train_val_both, num_val)
    for i in range(len(val_index)):
        train_val_both.remove(val_index[i])
    train_index = train_val_both
    # 保存准确率最高的一次迭代
    best_acc = 0.0
    """
    每一折都要实例化新的模型，不然模型会学到测试集的东西
    """
    # 2分类层
    net = ResNet.resnet18(num_classes=2, include_top=True)
    # net = Covnet_2.Covnet(drop_1=dropout_num_1, drop_2=dropout_num_2)
    # net = Covnet.Covnet(drop_1=dropout_num_1, drop_2=dropout_num_2)
    # net = GhostNet.ghostnet()
    # net = Covnet_3.Covnet()
    # net = efficientnet.efficientnet_b0(num_classes=2)
    # net = GhostNet_res.resnet18()
    # net = ResNet_attention.resnet18(num_classes=1000, include_top=True)
    # 加载预训练权重
    # net.load_state_dict(torch.load(weightpath, map_location=device))
    # 为网络重写分类层
    # in_channel = net.fc.in_features  # 2048
    # net.fc = nn.Linear(in_channel, 2)  # [b,2048]==>[b,2]
    # 给模型参数进行初始化
    # net.apply(init_weights)
    # 将模型搬运到GPU上
    net.to(device)
    # 获取网络名称
    net_name = net.__class__.__name__
    # 定义优化器
    # weight_decay：用于L2正则化，有助于抑制过拟合
    # optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    # 定义交叉熵损失
    loss_function = Focal_Loss(alpha=alpha)
    # loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)

    # 写一个txt文件用于保存超参数
    file_name = r"{}\{}网络 {}.txt".format(photo_folder, net_name, nowTime)
    file = open(file_name, 'w', encoding='utf-8')
    if os.path.exists(file_name):
        file.write("batch_size:{}\n epoch:{}\n learning_rate:{}\n".format(batch_size, epochs, learning_rate))
        file.write("weight_decay:{}\n".format(wd))
        file.write("dropout_1:{}, dropout_2:{}\n".format(dropout_num_1, dropout_num_2))

    # 初始化一些空白矩阵
    train_loss = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    pre_score = []
    labels_epoch = []
    min_val_loss = 10
    # 显示此时是第k折交叉验证
    k_num += 1
    print("-" * 30)
    print("第{}折验证".format(k_num))
    train_fold = torch.utils.data.dataset.Subset(data_0, train_index)
    val_fold = torch.utils.data.dataset.Subset(data_0, val_index)
    test_fold = torch.utils.data.dataset.Subset(data_0, test_index)
    # 计算训练集,验证集,测试集的大小
    train_num = len(train_fold)
    val_num = len(val_fold)
    test_num = len(test_fold)
    # 打包成DataLoader类型 用于 训练
    train_loader = DataLoader(dataset=train_fold, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_fold, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_fold, batch_size=batch_size, shuffle=True, drop_last=True)
    # # 早停的实例
    # early_stopping = EarlyStopping(patience=4, delta=0.1, path=savepath + '/model_' + dir_path + "_第{}折验证{}层网络".format(k_num, net_num) + '.pth')


    """
    训练过程
    """
    for epoch in range(epochs):
        # torch.cuda.empty_cache()  # 释放显存
        print('-' * 30, '\n', '共', epochs, '个epoch, 第', epoch + 1, '个epoch')
        file.write('{}\n,共{}个epoch,第{}个epoch\n'.format('-' * 30, epochs, epoch + 1))
        # 将模型设置为训练模型, dropout层和BN层只在训练时起作用
        net.train()
        # 计算训练一个epoch的总损失
        running_loss = 0.0
        epoch_acc = 0.0
        num_0 = 0
        num_1 = line_num
        line = "[" + ">" + "·" * 19 + "]"

        # 每个step训练一个batch
        # enumerate：遍历，返回索引和元素
        for step, data in enumerate(train_loader):
            running_acc = 0.0
            num_0 += 1

            # data中包含图像及其对应的标签
            images, labels = data

            # 梯度清零，因为每次计算梯度是一个累加
            optimizer.zero_grad()
            # 前向传播
            # output是torch.tensor类型的数据 [batch_size, 2]
            outputs = net(images.to(device))
            # print("output:{}".format(outputs))

            # 计算预测值和真实值的交叉熵损失,交叉熵损失也叫softmax损失
            # loss = loss_function(outputs, labels.to(device))
            loss = loss_function.forward(outputs, labels.to(device))

            # 计算acc
            # 预测分数的最大值
            predict_y = torch.max(outputs, dim=1)[1]

            # 累加每个step的准确率
            running_acc = (predict_y == labels.to(device)).sum().item()
            epoch_acc += running_acc

            # # 梯度计算
            loss.requires_grad_(True)
            loss.backward()
            # loss1 = loss.detach_().requires_grad_(True)
            # loss1.backward()

            # gc.collect()
            # torch.cuda.empty_cache()
            # 权重更新
            optimizer.step()
            # 学习率更新
            scheduler.step()

            # 累加每个step的损失
            running_loss += loss.item()

            # 可视化训练过程（进度条的形式）
            if num_0 <= num_1:
                line = "[" + "=" * int(num_1 / line_num - 1) + ">" + "·" * (19 - int(num_1 / line_num - 1)) + "]"
            else:
                num_1 += line_num
                line = "[" + "=" * int(num_1 / line_num - 1) + ">" + "·" * (19 - int(num_1 / line_num - 1)) + "]"
            # 打印每个step的损失和acc
            print(line, end='')
            print(f'共:{step_num} step:{step + 1} loss:{loss} acc:{running_acc / batch_size}')
            file.write("第{}折, 共:{} step:{} loss:{} acc:{}\n".format(k_num, step_num, step + 1, loss, running_acc / batch_size))


        # -------------------------------------------------- #
        # （5）网络验证
        # -------------------------------------------------- #
        net.eval()  # 切换为验证模型，BN和Dropout不进行参数的更新作用
        # net.train()  # 虽然是训练模式，但是因为没有写反向传播的代码，因此可以视为没有进行参数的更新

        acc = 0.0  # 验证集准确率
        val_loss_run = 0.0

        with torch.no_grad():  # 下面不进行梯度计算

            val_setp = 0.0
            # 每次验证一个batch
            for data_val in val_loader:
                # 获取验证集的图片和标签
                val_images, val_labels = data_val
                # 前向传播
                outputs = net(val_images.to(device))

                # 计算预测值和真实值的交叉熵损失
                # loss = loss_function(outputs, val_labels.to(device))
                loss = loss_function.forward(outputs, labels.to(device))

                # 累加每个step的损失
                val_loss_run += loss.item()


                # 预测分数的最大值
                predict_y = torch.max(outputs, dim=1)[1]

                # 累加每个step的准确率
                acc += (predict_y == val_labels.to(device)).sum().item()

                val_setp += 1

            # 计算所有图片的平均准确率
            acc_val = acc / val_num
            acc_train = epoch_acc / train_num

            # 打印每个epoch的训练损失和验证准确率
            print(f'total_train_loss:{running_loss / (step + 1)}, total_train_acc:{acc_train}')
            print(f'total_val_loss:{val_loss_run / val_setp}, total_val_acc:{acc_val}')
            train_loss.append(running_loss / (step + 1))
            train_acc.append(acc_train)
            val_loss.append(val_loss_run / val_setp)
            val_acc.append(acc_val)
            file.write('total_train_loss:{}, total_train_acc:{}\n'.format(running_loss / (step + 1), acc_train))
            file.write('total_val_loss:{}, total_val_acc:{}\n'.format(val_loss_run / val_setp, acc_val))

            # 进行早停的检查
            # early_stopping(val_loss_run / val_setp, net)
            # if early_stopping.early_stop:
            #     print("早停")
            #     break
            if val_loss[-1] <= min_val_loss:
                min_val_loss = val_loss[-1]
                epoch_num = epoch + 1
            if val_loss[-1] >= min_val_loss + 0.1:
                if (epoch + 1) - epoch_num >= 4:
                    # 保存的权重名称
                    savename = pth_flodr + '\\model_' + dir_path + "_第{}折验证".format(k_num) + net_name + "网络" + '.pth'
                    # 保存当前权重
                    torch.save(net.state_dict(), savename)
                    break

            # -------------------------------------------------- #
            # （6）权重保存
            # -------------------------------------------------- #
            # 保存每一折验证的最好权重
            if acc_val > best_acc:
                # 更新最佳的准确率
                best_acc = acc_val
                # 保存的权重名称
                savename = pth_flodr + '\\model_' + dir_path + "_第{}折验证".format(k_num) + net_name + "网络" + '.pth'
                # 保存当前权重
                torch.save(net.state_dict(), savename)

            # 保存整个训练中的最好权重
            if acc_val > best_acc_all:
                # 更新最佳的准确率
                best_acc_all = acc_val
                # 保存的权重名称
                savename = pth_flodr + '\\model_' + dir_path + "最好的权重" + net_name + "网络" + '.pth'
                # 保存当前权重
                torch.save(net.state_dict(), savename)

    """
    测试集，用于判断测试的准确率以及绘制roc曲线和混淆矩阵
    """
    # 使用刚刚训练的权重
    savename = pth_flodr + '\\model_' + dir_path + "_第{}折验证".format(k_num) + net_name + "网络" + '.pth'
    weightpath = savename
    # 初始化网络
    net = ResNet.resnet18(num_classes=2, include_top=True)
    # net = Covnet_2.Covnet(drop_1=dropout_num_1, drop_2=dropout_num_2)
    # net = GhostNet.ghostnet()
    # net = Covnet_3.Covnet()
    # net = efficientnet.efficientnet_b0(num_classes=2)
    # net = Covnet.Covnet(drop_1=dropout_num_1, drop_2=dropout_num_2)
    # net = GhostNet_res.resnet18()
    # 为网络重写分类层
    # in_channel = net.fc.in_features  # 2048
    # net.fc = nn.Linear(in_channel, 2)  # [b,2048]==>[b,2]
    # 加载权重
    net.load_state_dict(torch.load(weightpath, map_location=device))
    # 模型切换成验证模式，目的是让dropout和bn切换形式
    net.eval()
    # net.train()
    # 将模型搬运到GPU上
    net.to(device)
    test_acc = 0.0
    pre_score = []
    labels_epoch = []
    with torch.no_grad():
        test_step = 0.0
        for data_test in test_loader:
            # 获取测试集的图片和标签
            test_images, test_labels = data_test
            #  前向传播
            outputs = net(test_images.to(device))
            # 添加softmax层
            # outputs = nn.Softmax(dim=1)(outputs)
            # 预测分数的最大值
            predict_y = torch.max(outputs, dim=1)[1]
            # 累加每个step的准确率
            test_acc += (predict_y == test_labels.to(device)).sum().item()
            test_step += 1

            # 准备roc曲线所需要的数据
            positive_pre = outputs[:, 1]
            positive_pre = positive_pre.cpu()
            positive_pre = positive_pre.detach().numpy()
            positive_pre = positive_pre.tolist()
            labels = test_labels.detach().numpy()
            labels = labels.tolist()
            pre_score += positive_pre
            labels_epoch += labels

            # 更新混淆矩阵
            for index in range(len(test_labels)):
                cnf_matrix[predict_y[index]][labels[index]] += 1

        # 计算测试集图片的平均准确率
        acc_test = test_acc / test_num
        # 打印测试集的准确率
        print("第{}折测试集的acc：{}".format(k_num, acc_test))
        file.write("第{}折测试集的acc：{}\n".format(k_num, acc_test))


    # 保存k折的roc参数
    pre_score_k.append(pre_score)
    labels_k.append(labels_epoch)

    # 每一折验证的时候，都绘制loss和acc曲线
    # 加时间戳
    plt.figure()
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    # plt.ylim((0, 1))
    plt.legend(["train", "val"], loc="upper right")
    plt.savefig(photo_folder + "\\" + net_name + "网络 model_loss_第{}折_".format(k_num) + str(nowTime) + ".jpg")
    # plt.show()
    # plt.xlim((0,50))
    # plt.ylim((0,1))
    plt.figure()
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.title("model acc")
    plt.ylabel("acc")
    plt.xlabel("epoch")
    plt.ylim((0, 1))  # 限制一下绘图的幅度，更具有代表性一些
    plt.legend(["train", "val"], loc="lower right")
    plt.savefig(photo_folder + "\\" + net_name + "网络 model_acc_第{}折_".format(k_num) + str(nowTime) + ".jpg")


"""
k折交叉验证的话，在前面绘制了loss和acc
绘制混淆矩阵以及每一折的ROC曲线并取平均，计算每一折AUC并取平均
"""
file.close()
# 以下是用于绘制ROC曲线的代码部分
# # 以下是用于绘制ROC曲线的代码部分
avg_x = []
avg_y = []
sum = 0
clr_1 = 'tab:green'
clr_2 = 'tab:green'
clr_3 = 'k'

plt.figure()
for i in range(k):
    fpr, tpr, thersholds = roc_curve(labels_k[i], pre_score_k[i])
    avg_x.append(sorted(random.sample(list(fpr), len(list(fpr)))))
    avg_y.append(sorted(random.sample(list(tpr), len(list(tpr)))))
    roc_auc1 = auc(fpr, tpr)

    roc_auc = roc_auc1 * 100
    sum = sum + roc_auc
    plt.plot(fpr, tpr, label='V-' + str(i + 1) + ' (auc = {0:.2f})'.format(roc_auc), c=clr_1, alpha=0.2)

data_x = np.array(avg_x, dtype=object)
data_y = np.array(avg_y, dtype=object)
avg = sum / k

# 准备数据
data_x_plt = []

data_x_num = len(data_x[0])
if data_x_num >= len(data_x[1]):
    data_x_num = len(data_x[1])
if data_x_num >= len(data_x[2]):
    data_x_num = len(data_x[2])
if data_x_num >= len(data_x[3]):
    data_x_num = len(data_x[3])
if data_x_num >= len(data_x[4]):
    data_x_num = len(data_x[4])

for i in range(5):
    data_x[i] = sorted(random.sample(data_x[i], data_x_num))

for i in range(data_x_num):
    a = 0.0
    a += data_x[0][i]
    a += data_x[1][i]
    a += data_x[2][i]
    a += data_x[3][i]
    a += data_x[4][i]
    a = a / k
    data_x_plt.append(a)

data_y_plt = []
data_y_num = len(data_y[0])
if data_y_num >= len(data_y[1]):
    data_y_num = len(data_y[1])
if data_y_num >= len(data_y[2]):
    data_y_num = len(data_y[2])
if data_y_num >= len(data_y[3]):
    data_y_num = len(data_y[3])
if data_y_num >= len(data_y[4]):
    data_y_num = len(data_y[4])

for i in range(5):
    data_y[i] = sorted(random.sample(data_y[i], data_y_num))

for i in range(data_y_num):
    a = 0.0
    a += data_y[0][i]
    a += data_y[1][i]
    a += data_y[2][i]
    a += data_y[3][i]
    a += data_y[4][i]
    a = a / k
    data_y_plt.append(a)


plt.plot(data_x_plt, data_y_plt, label='AVG (auc = {0:.2f})'.format(avg), c=clr_2, alpha=1, linewidth=2)
plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
plt.ylim([-0.05, 1.05])
plt.plot([0, 1], [0, 1], linestyle='--', label='chance', c=clr_3, alpha=.5)
plt.legend(loc='lower right', frameon=False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.grid(color='gray', linestyle='--', linewidth=1, alpha=.3)
plt.text(0, 1, 'PATIENT-LEVEL ROC', color='gray', fontsize=12)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig(photo_folder + "\\" + net_name +"网络 model_ROC_" + str(nowTime) + ".jpg")
plt.show()

"""
绘制混淆矩阵，并保存
"""
Confusion_matrix_path = photo_folder + "\\" + net_name + "网络 Confusion matrix" + str(nowTime) + ".jpg"


# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues,
                          path=Confusion_matrix_path):
    """
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #         print("显示百分比：")
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    #         print(cm)
    #     else:
    #         print('显示具体数字：')
    #         print(cm)
    plt.figure(dpi=320, figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
    plt.ylim(len(classes) - 0.5, -0.5)
    # fmt = '.2f' if normalize else 'd'
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    plt.savefig(path)


# 第一种情况：显示百分比
# classes = ['cat', 'dog']
classes = ['negative', 'positive']
plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True, title='Normalized confusion matrix')

# # 第二种情况：显示数字
# plot_confusion_matrix(cnf_matrix, classes=classes, normalize=False, title='Normalized confusion matrix')

