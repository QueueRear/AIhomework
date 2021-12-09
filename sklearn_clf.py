import numpy as np
from os import listdir
from loader import img2vector
from sklearn.neural_network import MLPClassifier
from plot import plotLosslist


def readDataset(path):  # 将样本标签转化为 one-hot 向量
    fileList = listdir(path)  # 获取当前文件夹下所有文件
    n = len(fileList)
    dataset = np.zeros([n, 1024], int)  # 存放数字文件
    labels = np.zeros([n, 10])  # 存放 one-hot 标签，将各维置零
    for i in range(n):  # 遍历文件
        filename = fileList[i]
        digit = int(filename.split('_')[0])  # 文件名中包含了标签
        labels[i][digit] = 1  # 将文件对应的向量对应的维度置一
        dataset[i] = img2vector(path + '/' + filename)  # 文件对应的向量保存向量化的图像
    return dataset, labels

if __name__ == "__main__":
    trainingDataset, trainingLabels = readDataset('dataset/trainingDigits')  # 加载训练数据

    # 设置超参数
    # 设置一个有100个神经元的隐藏层
    alpha = 0.01
    clf = MLPClassifier(hidden_layer_sizes=(50,), activation='logistic', learning_rate_init=alpha, max_iter=2000)
    print(clf)
    clf.fit(trainingDataset, trainingLabels) # fit 函数自动设置多层感知机的输入输出神经元数
    
    lossList = clf.loss_curve_
    plotLosslist(lossList, "Loss of sklearn_clf : alpha=" + str(alpha))

    # 加载测试集
    testDataset, testLabels = readDataset('dataset/testDigits')

    ret = clf.predict(testDataset) # 让训练出来的模型在测试集上跑
    error = 0 # 计算判断错误的数量
    n = len(testDataset)
    for i in range(n):
        # ret[i] == testLabels[i] 会返回一个长度为10的数组，1 表示对应元素相同
        if np.sum(ret[i] == testLabels[i]) < 10:
            error += 1
    print("Total:", n, "Error:", error, "Accuracy:", float(n - error) / n)
