"""
此文件用于尝试从网上下载下来的例子
"""
import torch
import torchvision
from torchvision import datasets, models, transforms
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import datetime

# from network import Net
bath_size = 32
epoch_n = 64

# class Net(nn.Module):                                       # 新建一个网络类，就是需要搭建的网络，必须继承PyTorch的nn.Module父类
#     def __init__(self):                                     # 构造函数，用于设定网络层
#         super(Net, self).__init__()                         # 标准语句
#         self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)   # 第一个卷积层，输入通道数3，输出通道数16，卷积核大小3×3，padding大小1，其他参数默认
#         self.conv2 = torch.nn.Conv2d(16, 16, 3, padding=1)

#         self.fc1 = nn.Linear(56*56*16, 128)                 # 第一个全连层，线性连接，输入节点数50×50×16，输出节点数128
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 2)

#     def forward(self, x):                                    # 重写父类forward方法，即前向计算，通过该方法获取网络输入数据后的输出值
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)

#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)

#         x = x.view(x.size()[0], -1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         y = self.fc3(x)

#         return y
data_dir = r"D:\学习\大创\data\训练数据集\data\cat_vs_dog(new)_2022-11-27-21-56-25"
dir_count = data_dir.rfind('\\') + 1
dir_path = data_dir[dir_count:]
print(dir_path)
# 工作目录
work_path = r"D:\学习\大创\data\训练数据集\model"
negative = 'cat'
positive = 'dog'
train_negative_path = data_dir + "\\" + "train\\" + negative
train_positive_path = data_dir + "\\" + "train\\" + positive
val_negative_path = data_dir + "\\" + "val\\" + negative
val_positive_path = data_dir + "\\" + "val\\" + positive
all_photo_num = len(os.listdir(train_positive_path)) + len(os.listdir(train_negative_path))
all_photo_num += len(os.listdir(val_positive_path)) + len(os.listdir(val_negative_path))

# 图片处理
data_trainsforms = {
    "train": transforms.Compose([
        # transforms.RandomResizedCrop(300),
        transforms.Resize((224, 224)),
        # transforms.RandomCrop((224,224)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        #     transforms.ToTensor()
    ]),

    "val": transforms.Compose([
        # transforms.RandomResizedCrop(300),
        # transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]), }

# 拼接路径
image_datasets = {
    x: datasets.ImageFolder(root=os.path.join(data_dir, x),
                            transform=data_trainsforms[x])
    for x in ["train", "val"]
}
# 数据加载器

data_loader = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=bath_size, shuffle=True) for x in
               ["train", "val"]}
step_num_all = int(0.8 * all_photo_num / bath_size)

X_example, y_example = next(iter(data_loader["train"]))
example_classees = image_datasets["train"].classes
index_classes = image_datasets["train"].class_to_idx

# 迁移学习模型
model = models.resnet18(pretrained=False)
# 获取网络名称
net_name = model.__class__.__name__
# 自定义模型
# model = Net()

Use_gpu = torch.cuda.is_available()

# for parma in model.parameters():
#     parma.requires_grad = False  # 屏蔽预训练模型的权重，只训练最后一层的全连接的权重
# model.fc = torch.nn.Linear(2048, 2)
in_channel = model.fc.in_features  # 2048
model.fc = nn.Linear(in_channel, 2)  # [b,2048]==>[b,2]
# print(model)

if Use_gpu:
    model = model.cuda()

# 损失函数和优化器
loss_f = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.0001)
# optimizer = torch.optim.Adam(model.parameters(),lr = 0.00001)
train_loss = []
train_acc = []
val_loss = []
val_acc = []


for epoch in range(epoch_n):
    print("Epoch {}/{}".format(epoch+1, epoch_n))
    print("-" * 10)

    for phase in ["train", "val"]:
        if phase == "train":
            print("training")
            model.train(True)
        else:
            print("val")
            model.train(False)
        running_loss = 0.0
        running_corrects = 0

        for batch, data in enumerate(data_loader[phase], 1):
            X, y = data
            if Use_gpu:
                X, y = Variable(X.cuda()), Variable(y.cuda())
            else:
                X, y = Variable(X), Variable(y)

            y_pred = model(X)

            _, pred = torch.max(y_pred.data, 1)
            optimizer.zero_grad()
            loss = loss_f(y_pred, y)
            if phase == "train":
                loss.backward()  # 反向传播计算当前梯度# 误差反向传播，采用求导的方式，计算网络中每个节点参数的梯度，显然梯度越大说明参数设置不合理，需要调整
                optimizer.step()  # 优化采用设定的优化方法对网络中的各个参数进行调整
            running_loss += loss.item()
            running_corrects += torch.sum(pred == y.data)
            # print("Batch{},Loss:{:.4f},ACC:{:.4f}".format(batch + 1, loss, 100*running_corrects/(bath_size*batch)))
            if phase == "train":
                print("共{}, Batch{},Loss:{:.4f},ACC:{:.4f}".format(step_num_all, batch, loss, 100 * torch.sum(pred == y.data) / bath_size))
        epoch_loss = running_loss / batch
        epoch_acc = 100 * running_corrects / (batch * bath_size)
        epoch_acc = epoch_acc.cpu().numpy()
        print("{} Loss:{:.4f} Acc:{:.4f}%".format(phase, epoch_loss, epoch_acc))
        if phase == 'train':
            train_loss.append(epoch_loss)
            train_acc.append(epoch_acc)
        else:
            val_loss.append(epoch_loss)
            val_acc.append(epoch_acc)
# torch.save(model.state_dict(),'model.ckpt1')
# torch.save(model.state_dict(),'model.pth'
print("over")


"""
绘图
"""
dir_count = data_dir.rfind('\\') + 1
dir_path = data_dir[dir_count:]
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
plt.figure()
plt.plot(train_loss)
plt.plot(val_loss)
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
# plt.ylim((0, 1))
plt.legend(["train", "val"], loc="upper right")
plt.savefig(photo_folder + "\\" + net_name + "验证网络_loss_" + str(nowTime) + ".jpg")
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
plt.savefig(photo_folder + "\\" + net_name + "验证网络_acc_" + str(nowTime) + ".jpg")

