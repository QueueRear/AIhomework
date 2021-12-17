import numpy as np
from os import error, listdir
from DataProcess import DataProcess
from sklearn.neighbors import KNeighborsClassifier

class NewDataProcess(DataProcess): # 继承数据处理类
    def __init__(self, path):
        super().__init__(path)

    def readDataset(self):
        fileList = listdir(self.path)  # 获取当前文件夹下所有文件
        n = len(fileList)
        dataset = np.zeros([n, 1024], int)  # 存放数字文件
        labels = np.zeros([n])  # 存放标签
        for i in range(n):  # 遍历文件
            filename = fileList[i]
            digit = int(filename.split('_')[0])  # 文件名中包含了标签
            labels[i] = digit  # 将文件对应的标签中存放数字
            dataset[i] = self.img2vector(self.path + '/' + filename)  # 文件对应的向量保存向量化的图像
        return dataset, labels


if __name__ == "__main__":
    trainingDataset, trainingLabels = NewDataProcess(
        'dataset/trainingDigits').readDataset()  # 加载训练数据

    # 构建 KNN 分类器
    # 设置查找算法及邻居点数量（k值）
    knn = KNeighborsClassifier(algorithm='kd_tree', n_neighbors=3)
    print(knn)
    knn.fit(trainingDataset, trainingLabels)

    # 加载测试集
    testDataset, testLabels = NewDataProcess('dataset/testDigits').readDataset()

    ret = knn.predict(testDataset)
    error = np.sum(ret != testLabels)
    n = len(testDataset)
    print("Total:", n, "Error:", error, "Accuracy:", float(n - error) / n)
