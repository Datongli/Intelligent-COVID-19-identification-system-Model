"""
此文件用于从pytorch上下载官方的示例程序
用于猫狗训练
"""
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import datetime

cudnn.benchmark = True
plt.ion()  # interactive mode

data_dir = r"D:\学习\大创\data\训练数据集\data\cat_vs_dog(new)_2022-11-27-21-56-25"
# data_dir = r"C:\Users\28101\Desktop\大创测试数据\test"
epochs_num = 32
bath_size = 32
learning_rate = 0.0001


work_path = r"D:\学习\大创\data\训练数据集\model"
negative = 'cat'
positive = 'dog'
train_negative_path = data_dir + "\\" + "train\\" + negative
train_positive_path = data_dir + "\\" + "train\\" + positive
val_negative_path = data_dir + "\\" + "val\\" + negative
val_positive_path = data_dir + "\\" + "val\\" + positive
all_photo_num = len(os.listdir(train_positive_path)) + len(os.listdir(train_negative_path))
all_photo_num += len(os.listdir(val_positive_path)) + len(os.listdir(val_negative_path))
step_num_all = int(0.8 * all_photo_num / bath_size)

train_loss = []
train_acc = []
val_loss = []
val_acc = []

"""
加载数据
"""
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=bath_size,
                                              shuffle=True, num_workers=4, drop_last=True)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
定义一个模型训练的函数
"""
def train_model(model, criterion, optimizer, scheduler, num_epochs=epochs_num):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 30)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            step_num = 0
            for inputs, labels in dataloaders[phase]:
                step_num += 1
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # 打印一下每一步的损失和准确率
                print("共{}步,第{}步  loss:{}  acc:{}".format(step_num_all, step_num,
                    loss.item(), torch.sum(preds == labels.data) / bath_size))

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_acc = float(epoch_acc.cpu())
            print("acc_type:{}".format(type(epoch_acc)))

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            if phase == 'val':
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


"""
显示几个图像预测值的通用函数
"""


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                plt.imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


model_ft = models.resnet18(weights=None)
# 获取网络名称
model_name = model_ft.__class__.__name__
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
optimizer_ft = optim.Adam(model_ft.parameters(), lr=learning_rate)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

if __name__ == "__main__":


    """
    训练与评估
    """
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=epochs_num)

    """
    打印出图片做验证
    """
    # visualize_model(model_ft)

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
    plt.savefig(photo_folder + "\\" + model_name + "验证网络_loss_" + str(nowTime) + ".jpg")
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
    plt.savefig(photo_folder + "\\" + model_name + "验证网络_acc_" + str(nowTime) + ".jpg")
