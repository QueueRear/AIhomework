from matplotlib.pyplot import ginput
import torch
import numpy as np
from os import listdir

from plot import plotLosslist
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

import  time

# 数据处理类
class DataProcess(Dataset):
    def __init__(self, path):
        self.data_info = []
        fileList = listdir(path)  # 获取当前文件夹下所有文件
        n = len(fileList)
        for i in range(n):  # 遍历文件
            filename = fileList[i]
            digit = int(filename.split('_')[0])  # 文件名中包含了标签
            self.data_info.append((path + '/' + filename, digit))

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        filepath, label = self.data_info[index]
        temparray = np.zeros([32, 32], dtype='float32')
        f = open(filepath)
        lines = f.readlines()
        for i in range(32):
            for j in range(32):
                temparray[i][j] = lines[i][j]
        t = torch.from_numpy(temparray)
        img = t.unsqueeze(0).float()
        return img, label

# 构建网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 第一层输入层，1：灰度图片的通道 10：输出通道 5：kernel卷积核
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        # 第二层中间层：10：输入通道 20：输出通道 3：kernel
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3)
        # 防止过拟合
        self.dropout = nn.Dropout(0.25)
        # 全连接层，线性层 20*12*12：输入通道 10：输出通道
        self.out = nn.Linear(20*6*6, 10)

    def forward(self, x): # 前向传播
        input_size = x.size(0) # x的格式：batch_size * 1（灰度通道） * 32 * 32
        x = self.conv1(x) # 输入：batch_size*1*32*32 输出：batch_size*10*(32-5+1)*(32-5+1)
        x = func.relu(x)  # 喂给激活函数，让它表达能力更强；输出形状不变：batch_size*10*28*28
        # 喂给池化层，输入：batch_size*10*28*28，输出：batch_size*10*28/2*28/2
        x = func.max_pool2d(x, 2, 2)

        # 调用第二个卷积层
        x = self.conv2(x)  # 输入：batch_size*10*14*14，输出：batch*20*(14-3+1)*(14-3+1)
        x = func.relu(x)
        # 喂给池化层，输入：batch*20*12*12，输出：batch_size*20*12/2*12/2
        x = func.max_pool2d(x, 2, 2)

        x = self.dropout(x)
        x = x.view(input_size, -1) # 拉平成一维向量，-1：自动计算维度，20*6*6=720

        # 喂给全连接层
        x = self.out(x) # 输入：batch*720，输出：batch*500

        output = func.log_softmax(x, dim=1) # 计算分类后每个数字的概率，dim=1表示按行计算
        return output


# 定义训练方法
def train(model, device, trainingLoader, epoch):
    optimizer = optim.Adam(model.parameters()) # 定义优化器
    trainLoss = []
    for i in range(epoch):
        accurate = 0.0
        avgLoss = 0.0
        # 模型训练
        model.train()
        for data, label in trainingLoader:
            # 部署到 device 上
            data, label = data.to(device), label.to(device)
            # 梯度初始化为零
            optimizer.zero_grad()
            # 训练后的结果
            output = model(data)
            # 计算损失
            loss = func.cross_entropy(output, label) # 针对多分类任务的损失函数
            trainLoss.append(loss.item())
            avgLoss += loss.item()
            # 找到概率值最大的下标
            pred = output.argmax(dim=1) # dim表示维度
            # 累计正确值
            accurate += pred.eq(label.view_as(pred)).sum().item()
            # 反向传播
            loss.backward()
            # 参数优化
            optimizer.step()
        avgLoss /= len(trainingData.data_info)
        print("Epoch:{} Loss:{:.6f} Accuracy:{:.4f}%".format(
            i + 1, avgLoss, 100.0*accurate / len(trainingData.data_info)))
    return trainLoss

# 定义测试方法
def test(model, device, testLoader):
    # 模型验证
    model.eval()
    # 统计正确率
    accurate = 0.0
    # 测试损失
    loss = 0.0
    with torch.no_grad(): # 不会计算梯度，也不会反向传播
        for data, label in testLoader:
            data, label = data.to(device), label.to(device)
            # 测试数据
            output = model(data)
            # 计算损失
            loss += func.cross_entropy(output, label).item()
            # 找到概率值最大的下标
            pred = output.argmax(dim=1)
            # 累计正确值
            accurate += pred.eq(label.view_as(pred)).sum().item()
        loss /= len(testData.data_info)
        print("Test Loss:{:.6f} Error:{} Accuracy:{:.4f}%".format(
            loss, len(testData.data_info)-accurate, 100.0*accurate / len(testData.data_info)))


if __name__ == "__main__":
    # 定义超参数
    BATCH_SIZE = 128  # 每批处理的数据量
    DEVICE = torch.device("cuda" if torch.cuda.is_available()
                        else "cpu")  # 根据平台判断是否用显卡
    EPOCHS = 20  # 训练轮数

    # 加载数据集
    trainingData = DataProcess('dataset/testDigits')
    testData = DataProcess('dataset/trainingDigits')

    net = CNN()

    # 部署到 Device 上
    model = net.to(DEVICE)

    trainingLoader = DataLoader(trainingData, BATCH_SIZE, True)
    testLoader = DataLoader(testData, BATCH_SIZE, True)

    # 调用方法训练测试
    start = time.time()
    trainLoss = train(model, DEVICE, trainingLoader, EPOCHS)
    end = time.time()
    print(end-start)
    plotLosslist(trainLoss, "Loss of PyTorch : Epoch=" + str(EPOCHS))
    test(model, DEVICE, testLoader)
