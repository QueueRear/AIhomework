import numpy as np
from DataProcess import DataProcess
from sklearn.neural_network import MLPClassifier
from plot import plotLosslist


if __name__ == "__main__":
    trainingDataset, trainingLabels = DataProcess(
        'dataset/trainingDigits').readDataset()  # 加载训练数据

    # 设置超参数
    # 设置一个有50个神经元的隐藏层
    alpha = 0.01
    clf = MLPClassifier(hidden_layer_sizes=(50,), activation='logistic', learning_rate_init=alpha, max_iter=2000)
    print(clf)
    clf.fit(trainingDataset, trainingLabels) # fit 函数自动设置多层感知机的输入输出神经元数
    
    lossList = clf.loss_curve_
    plotLosslist(lossList, "Loss of sklearn_clf : alpha=" + str(alpha))

    # 加载测试集
    testDataset, testLabels = DataProcess('dataset/testDigits').readDataset()

    ret = clf.predict(testDataset) # 让训练出来的模型在测试集上跑
    accurate = 0  # 计算判断正确的数量
    n = len(testDataset)
    for i in range(n):
        accurate += int(np.argmax(ret[i]) == np.argmax(testLabels[i]))
    print("Total:", n, "Error:", n - accurate, "Accuracy:", float(accurate * 100) / n, "%")
