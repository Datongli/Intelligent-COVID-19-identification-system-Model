import torch
from torch import nn
# from PolarizedSelfAttention import SequentialPolarizedSelfAttention


class Covnet(nn.Module):
    def __init__(self):
        super(Covnet, self).__init__()  # 对父类属性初始化
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=0)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=2, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(32, 48, 6)#1s
        self.flat = nn.Flatten()
        self.dn1 = nn.Linear(in_features=387200, out_features=256)#1s
        self.drop1 = nn.Dropout(p=0.2)
        self.dn2 = nn.Linear(in_features=256, out_features=128)
        self.drop2 = nn.Dropout(p=0.1)
        self.dn3 = nn.Linear(in_features=128, out_features=2)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn1(x)
        x = self.flat(x)
        x = self.dn1(x)
        x = self.drop1(x)
        x = self.dn2(x)
        x = self.drop2(x)
        x = self.dn3(x)
        x = self.sigmoid(x)
        return(x)


"""
这个是加入了注意力机制的，但是学长说效果不是很好，因此先不用
"""
class atten_Covnet(nn.Module):
    def __init__(self):
        super(atten_Covnet, self).__init__()  # 对父类属性初始化
        self.conv1=nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=0)
        self.relu1=nn.ReLU(inplace=True)
        self.pool1=nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2=nn.Conv2d(in_channels=64, out_channels=32, kernel_size=2, padding=0)
        self.relu2=nn.ReLU(inplace=True)
        self.bn1=nn.BatchNorm2d(32, 48, 6)
        self.flat=nn.Flatten()
        self.dn1=nn.Linear(in_features=9216, out_features=256)
        self.drop1=nn.Dropout(p=0.5)
        self.dn2=nn.Linear(in_features=256, out_features=128)
        self.drop2=nn.Dropout(p=0.3)
        self.dn3=nn.Linear(in_features=128, out_features=1)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        x=self.conv1(x)
        x=self.relu1(x)
        # psa=SequentialPolarizedSelfAttention(channel=64)
        x.reshape(-1,64,)
        # out=psa(x)

        out=self.pool1(x)
        out=self.conv2(out)
        out= self.relu2(out)
        x=self.bn1(out)
        x=self.flat(x)
        x=self.dn1(x)
        x=self.drop1(x)
        x=self.dn2(x)
        x=self.drop2(x)
        x=self.dn3(x)
        x=self.sigmoid(x)
        return(x)