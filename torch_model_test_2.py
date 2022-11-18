"""
此文件用于弄懂混淆矩阵并且绘制混淆矩阵
同时以后可以计算精确度、召回率等四个数据
"""
"""
过拟合现象严重，望解决
"""
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
 # 从自定义的ResNet.py文件中导入resnet50这个函数
import ResNet
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import itertools
import GhostNet

# -------------------------------------------------- #
# （0）参数设置
# -------------------------------------------------- #
batch_size = 32  # 每个step训练batch_size张图片
epochs = 32  # 共训练epochs次
train_loss = []
train_acc = []
val_loss = []
val_acc = []


# -------------------------------------------------- #
# （1）文件配置
# -------------------------------------------------- #
# 数据集文件夹位置
filepath = "D:/学习/大创/data/训练数据集/data/cat_vs_dog(new)_2022-11-04-11-59-51"
# 权重文件位置
weightpath = "D:/学习/大创/data/训练数据集/data/path/resnet50.pth"


dir_count = filepath.rfind('/') + 1
dir_path = filepath[dir_count:]
print(dir_path)
# 创建权重的文件夹
cd = os.path.exists("D:/学习/大创/data/训练数据集/model/pth/" + dir_path)
if cd:
    print("权重保存文件夹已存在")
else:
    print("创建权重保存文件夹")
    os.mkdir("D:/学习/大创/data/训练数据集/model/pth/" + dir_path)
# 权重保存文件夹路径
savepath = "D:/学习/大创/data/训练数据集/model/pth/" + dir_path


# 获取GPU设备
if torch.cuda.is_available():  # 如果有GPU就用，没有就用CPU
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# -------------------------------------------------- #
# （2）构造数据集
# -------------------------------------------------- #
# 训练集的数据预处理
transform_train = transforms.Compose([
    # 将输入图像大小调整为224*224
    transforms.Resize((224, 224)),
    # # 数据增强，随机裁剪224*224大小
    # transforms.RandomResizedCrop(224),
    # 数据增强，随机水平翻转
    transforms.RandomHorizontalFlip(),
    # 数据变成tensor类型，像素值归一化，调整维度[h,w,c]==>[c,h,w]
    transforms.ToTensor(),
    # 对每个通道的像素进行标准化，给出每个通道的均值和方差
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

# 验证集的数据预处理
transform_val = transforms.Compose([
    # 将输入图像大小调整为224*224
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

# 读取训练集并预处理
train_dataset = datasets.ImageFolder(root=filepath + '/train',  # 训练集图片所在的文件夹
                                     transform=transform_train)  # 训练集的预处理方法

# 读取验证集并预处理
val_dataset = datasets.ImageFolder(root=filepath + '/val',  # 验证集图片所在的文件夹
                                   transform=transform_val)  # 验证集的预处理方法

# 获取数据集类别数量
classes = train_dataset.classes

# 初始化混淆矩阵
cnf_matrix = np.zeros([len(classes), len(classes)])

# 查看训练集和验证集的图片数量
train_num = len(train_dataset)
val_num = len(val_dataset)
print('train_num:', train_num, 'val_num:', val_num)
step_num = int(train_num / batch_size)

# 查看图像类别及其对应的索引
class_dict = train_dataset.class_to_idx
print(class_dict)  # {'Bananaquit': 0, 'Black Skimmer': 1, 'Black Throated Bushtiti': 2, 'Cockatoo': 3}
# 将类别名称保存在列表中
class_names = list(class_dict.keys())

# 构造训练集
train_loader = DataLoader(dataset=train_dataset,  # 接收训练集
                          batch_size=batch_size,  # 训练时每个step处理batch_size张图
                          shuffle=True,  # 打乱每个batch
                          num_workers=0,
                          drop_last=True)  # 加载数据时的线程数量，windows环境下只能=0

# 构造验证集
val_loader = DataLoader(dataset=val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=0,
                        drop_last=True)

# -------------------------------------------------- #
# （3）数据可视化
# -------------------------------------------------- #
# 取出一个batch的训练集，返回图片及其标签
train_img, train_label = iter(train_loader).next()
# 查看shape, img=[32,3,224,224], label=[32]
print(train_img.shape, train_label.shape)

# 从一个batch中取出前9张图片
img = train_img[:9]  # [9, 3, 224, 224]
# 将图片反标准化，像素变到0-1之间
img = img / 2 + 0.5
# tensor类型变成numpy类型
img = img.numpy()
class_label = train_label.numpy()
# 维度重排 [b,c,h,w]==>[b,h,w,c]
img = np.transpose(img, [0, 2, 3, 1])

# 创建画板
plt.figure()
# 绘制9张图片
for i in range(img.shape[0]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(img[i])
    plt.xticks([])  # 不显示x轴刻度
    plt.yticks([])  # 不显示y轴刻度
    plt.title(class_names[class_label[i]])  # 图片对应的类别

plt.tight_layout()  # 轻量化布局
plt.show()

# -------------------------------------------------- #
# （4）加载模型
# -------------------------------------------------- #
# 1000分类层
# resnet50
# net = ResNet.resnet50(num_classes=1000, include_top=True)
# resnet18
# net = ResNet.resnet18(num_classes=1000, include_top=True)
net = GhostNet.ghostnet()

# 加载预训练权重
# net.load_state_dict(torch.load(weightpath, map_location=device))

# 为网络重写分类层
# in_channel = net.fc.in_features  # 2048
# net.fc = nn.Linear(in_channel, 2)  # [b,2048]==>[b,2]

# 将模型搬运到GPU上
net.to(device)
# 定义交叉熵损失
loss_function = nn.CrossEntropyLoss()
# 定义优化器
optimizer = optim.Adam(net.parameters(), lr=0.0001)

# 保存准确率最高的一次迭代
best_acc = 0.0

# -------------------------------------------------- #
# （5）网络训练
# -------------------------------------------------- #
# 这一段代码是为了过程化训练进程做准备
line_num = step_num / 20.0
num_1 = line_num

# 网络训练
for epoch in range(epochs):

    print('-' * 30, '\n', '共', epochs, '个epoch, 第', epoch + 1, '个epoch')

    # 将模型设置为训练模型, dropout层和BN层只在训练时起作用
    net.train()

    # 计算训练一个epoch的总损失
    running_loss = 0.0
    epoch_acc = 0.0
    num_0 = 0
    num_1 = line_num
    line = "[" + ">" + "·" * 19 + "]"

    # 每个step训练一个batch
    for step, data in enumerate(train_loader):
        running_acc = 0.0
        num_0 += 1

        # data中包含图像及其对应的标签
        images, labels = data

        # 梯度清零，因为每次计算梯度是一个累加
        optimizer.zero_grad()

        # 前向传播
        outputs = net(images.to(device))

        # 计算预测值和真实值的交叉熵损失
        loss = loss_function(outputs, labels.to(device))

        # 计算acc
        # 预测分数的最大值
        predict_y = torch.max(outputs, dim=1)[1]
        # 累加每个step的准确率
        running_acc = (predict_y == labels.to(device)).sum().item()
        epoch_acc += running_acc

        # 梯度计算
        loss.backward()

        # 权重更新
        optimizer.step()

        # 累加每个step的损失
        running_loss += loss.item()

        # 更新混淆矩阵数据
        for idx in range(len(labels)):
            cnf_matrix[predict_y[idx]][labels[idx]] += 1

        # 可视化训练过程（进度条的形式）
        if num_0 <= num_1:
            line = "[" + "=" * int(num_1 / line_num - 1) + ">" + "·" * (19 - int(num_1 / line_num - 1)) + "]"
        else:
            num_1 += line_num
            line = "[" + "=" * int(num_1 / line_num - 1) + ">" + "·" * (19 - int(num_1 / line_num - 1)) + "]"
        # 打印每个step的损失和acc
        print(line, end='')
        print(f'共:{step_num} step:{step + 1} loss:{loss} acc:{running_acc / batch_size}')

    # -------------------------------------------------- #
    # （6）网络验证
    # -------------------------------------------------- #
    net.eval()  # 切换为验证模型，BN和Dropout不起作用

    acc = 0.0  # 验证集准确率
    test_loss = 0.0

    with torch.no_grad():  # 下面不进行梯度计算

        test_setp = 0.0
        # 每次验证一个batch
        for data_test in val_loader:
            # 获取验证集的图片和标签
            test_images, test_labels = data_test

            # 前向传播
            outputs = net(test_images.to(device))

            # 计算预测值和真实值的交叉熵损失
            loss = loss_function(outputs, test_labels.to(device))

            # 累加每个step的损失
            test_loss += loss.item()

            # 预测分数的最大值
            predict_y = torch.max(outputs, dim=1)[1]

            # 累加每个step的准确率
            acc += (predict_y == test_labels.to(device)).sum().item()

            test_setp += 1

        # 计算所有图片的平均准确率
        acc_test = acc / val_num
        acc_train = epoch_acc / train_num

        # 更新混淆矩阵数据
        for idx in range(len(test_labels)):
            cnf_matrix[predict_y[idx]][test_labels[idx]] += 1

        # 打印每个epoch的训练损失和验证准确率
        print(f'total_train_loss:{running_loss / (step + 1)}, total_train_acc:{acc_train}')
        print(f'total_test_loss:{test_loss / test_setp}, total_test_acc:{acc_test}')
        train_loss.append(running_loss / (step + 1))
        train_acc.append(acc_train)
        val_loss.append(test_loss / test_setp)
        val_acc.append(acc_test)

        # -------------------------------------------------- #
        # （7）权重保存
        # -------------------------------------------------- #
        # 保存最好的准确率的权重
        if acc_test > best_acc:
            # 更新最佳的准确率
            best_acc = acc_test
            # 保存的权重名称
            savename = savepath + '/model_' + dir_path + '.pth'
            # 保存当前权重
            torch.save(net.state_dict(), savename)


"""
绘制acc曲线以及混淆矩阵
"""
dir_count = filepath.rfind('/') + 1
dir_path = filepath[dir_count:]
print(dir_path)
# 判断文件夹是否存在
cd = os.path.exists("D:/学习/大创/data/训练数据集/model/photo/" + dir_path)
if cd:
    print("图片保存文件夹已存在")
else:
    print("创建图片保存文件夹")
    os.mkdir("D:/学习/大创/data/训练数据集/model/photo/" + dir_path)
# 加时间戳
nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
plt.figure()
plt.plot(train_loss)
plt.plot(val_loss)
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
# plt.ylim((0, 1))
plt.legend(["train", "test"], loc="upper right")
plt.savefig("D:/学习/大创/data/训练数据集/model/photo/" + dir_path + "/model_loss" + str(nowTime) + ".jpg")
# plt.show()
# plt.xlim((0,50))
# plt.ylim((0,1))
plt.figure()
plt.plot(train_acc)
plt.plot(val_acc)
plt.title("model acc")
plt.ylabel("acc")
plt.xlabel("epoch")
# plt.ylim((0, 1))  # 限制一下绘图的幅度，更具有代表性一些
plt.legend(["train", "test"], loc="lower right")
plt.savefig("D:/学习/大创/data/训练数据集/model/photo/" + dir_path + "/model_acc" + str(nowTime) + ".jpg")


"""
绘制混淆矩阵，并保存
"""
Confusion_matrix_path = "D:/学习/大创/data/训练数据集/model/photo/" + dir_path + "/Confusion matrix" + str(nowTime) + ".jpg"


# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, path=Confusion_matrix_path):
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
    plt.show()


# 第一种情况：显示百分比
plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True, title='Normalized confusion matrix')

# # 第二种情况：显示数字
# plot_confusion_matrix(cnf_matrix, classes=classes, normalize=False, title='Normalized confusion matrix')



