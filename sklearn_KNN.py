import numpy as np
from os import error, listdir
from loader import img2vector
from sklearn import neighbors


def readDataset(path):
    fileList = listdir(path)  # 获取当前文件夹下所有文件
    n = len(fileList)
    dataset = np.zeros([n, 1024], int)  # 存放数字文件
    labels = np.zeros([n])  # 存放标签
    for i in range(n):  # 遍历文件
        filename = fileList[i]
        digit = int(filename.split('_')[0])  # 文件名中包含了标签
        labels[i] = digit  # 将文件对应的标签中存放数字
        dataset[i] = img2vector(path + '/' + filename)  # 文件对应的向量保存向量化的图像
    return dataset, labels


if __name__ == "__main__":
    trainingDataset, trainingLabels = readDataset('dataset/trainingDigits')  # 加载训练数据

    # 构建 KNN 分类器
    # 设置查找算法及邻居点数量（k值）
    knn = neighbors.KNeighborsClassifier(algorithm='kd_tree', n_neighbors=3)
    print(knn)
    knn.fit(trainingDataset, trainingLabels)

    # 加载测试集
    testDataset, testLabels = readDataset('dataset/testDigits')

    ret = knn.predict(testDataset)
    error = np.sum(ret != testLabels)
    n = len(testDataset)
    print("Total:", n, "Error:", error, "ErrorRate:", float(error) / n)
