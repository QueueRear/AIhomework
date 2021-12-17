import numpy as np
from os import listdir


class DataProcess():
    def __init__(self, path):
        self.path = path

    # 加载训练数据
    def img2vector(self, filepath):  # 将图片矩阵展开为一列向量
        ret = np.zeros([1024], int)  # 返回矩阵大小为 1 * 1024
        f = open(filepath)
        lines = f.readlines()  # 读取文件各行
        for i in range(32):
            for j in range(32):
                ret[i * 32 + j] = lines[i][j]  # 将二维数组拉成一维
        return ret

    def readDataset(self):  # 将样本标签转化为 one-hot 向量
        fileList = listdir(self.path)  # 获取当前文件夹下所有文件
        n = len(fileList)
        dataset = np.zeros([n, 1024], int)  # 存放数字文件
        labels = np.zeros([n, 10], int)  # 存放 one-hot 标签，将各维置零
        for i in range(n):  # 遍历文件
            filename = fileList[i]
            digit = int(filename.split('_')[0])  # 文件名中包含了标签
            labels[i][digit] = 1  # 将文件对应的向量对应的维度置一
            dataset[i] = self.img2vector(self.path + '/' + filename)  # 文件对应的向量保存向量化的图像
        return dataset, labels