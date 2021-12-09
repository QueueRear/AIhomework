import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.transforms.functional import normalize

# 定义超参数
BATCH_SIZE = 128 # 每批处理的数据量
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 根据平台判断是否用显卡
EPOCHS = 25 # 训练轮数

# 构建 pipeline，对图像做处理
pipeline  = transforms.Compose([
    # 将图片转换成tensor类型
    transforms.ToTensor(), 
    # 正则化，第一个参数是标准差，第二个是均值，当过拟合时降低模型复杂度
    transforms.normalize((0.1307,), (0.3081,))
])

from torch.utils.data import DataLoader

