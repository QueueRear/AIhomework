import numpy as np

from DataProcess import DataProcess
from plot import plotLosslist
import random


def Sigmod(x, diff=False):
    def sigmod(x):
        return 1 / (1 + np.exp(-x))

    def dsigmod(x):
        f = sigmod(x)
        return f * (1 - f)

    return sigmod(x) if diff == False else dsigmod(x)


def squareErrorSum(y_hat, y, diff=False):
    return (np.square(y_hat - y) * 0.5).sum() if diff == False else y_hat - y


class Net():
    def __init__(self, hiddenLayers, sd):
        self.n = len(hiddenLayers)
        # X Input
        np.random.seed(sd)
        random.seed(sd)
        self.X = np.random.randn(1024, 1)

        self.w = [np.random.randn(hiddenLayers[0], 1024)]
        self.b = [np.random.randn(hiddenLayers[0], 1)]
        for i in range(1, self.n):
            self.w.append(np.random.randn(
                hiddenLayers[i], hiddenLayers[i - 1]))
            self.b.append(np.random.randn(hiddenLayers[i], 1))

        self.w.append(np.random.randn(10, hiddenLayers[-1]))
        self.b.append(np.random.randn(10, 1))
        self.alpha = 0.01  # 学习率
        self.trainLoss = []  # 用于作图

    def forward(self, X, y, activate):
        self.X = X
        self.z = [np.dot(self.w[0], self.X) + self.b[0]]
        self.a = [activate(self.z[0])]
        for i in range(1, self.n + 1):
            self.z.append(np.dot(self.w[i], self.a[i - 1]) + self.b[i])
            self.a.append(activate(self.z[i]))

        self.y_hat = activate(self.z[-1])
        Loss = squareErrorSum(self.y_hat, y)
        return Loss, self.y_hat

    def backward(self, y, activate):
        self.delta = [0 for i in range(self.n + 1)]
        self.delta[-1] = activate(self.z[-1], True) * \
            squareErrorSum(self.y_hat, y, True)
        for i in range(self.n - 1, -1, -1):
            self.delta[i] = activate(self.z[i], True) * \
                (np.dot(self.w[i + 1].T, self.delta[i + 1]))

        dw = [np.dot(self.delta[0], self.X.T)]
        for i in range(1, self.n + 1):
            dw.append(np.dot(self.delta[i], self.a[i - 1].T))

        # update weight
        for i in range(self.n + 1):
            self.w[i] -= self.alpha * dw[i]
            self.b[i] -= self.alpha * self.delta[i]

    def train(self, trainingDataset, trainingLabels, Epoch=5, alpha=0.01, batch=None, shuffle=False):
        self.alpha = alpha
        if shuffle == True:
            tmp = list(zip(trainingDataset, trainingLabels))
            for i in range(100):
                random.shuffle(tmp)
            trainingDataset[:], trainingLabels[:] = zip(*tmp)
        for epoch in range(Epoch):
            accurate = 0
            if batch == None:
                batch = len(trainingDataset)
            for i in range(batch):  # 可以用batch，数据较少，一次训练所有数据集
                X = trainingDataset[i, :].reshape([1024, 1])  # 生成输入

                label = trainingLabels[i, :].reshape([10, 1])

                Loss, y_hat = self.forward(X, label, Sigmod)  # 前向传播
                self.backward(label, Sigmod)  # 反向传播

                accurate += int(np.argmax(label) == np.argmax(y_hat))

            acc = 100.0*accurate / len(trainingDataset)
            self.trainLoss.append(Loss)
            print("epoch: %d, loss: %02f, accrucy : %05.3f%%" %
                  (epoch + 1, Loss, acc))

    def test(self, testDataset, testLabels, batch=None, shuffle=False):
        if shuffle == True:
            tmp = list(zip(testDataset, testLabels))
            for i in range(100):
                random.shuffle(tmp)
            testDataset[:], testLabels[:] = zip(*tmp)
        accurate = 0
        if batch == None:
            batch = len(testDataset)
        for i in range(batch):  # 可以用batch，数据较少，一次训练所有数据集
            X = testDataset[i, :].reshape([1024, 1])  # 生成输入

            label = testLabels[i, :].reshape([10, 1])

            Loss, y_hat = self.forward(X, label, Sigmod)  # 前向传播

            accurate += int(np.argmax(label) == np.argmax(y_hat))
        acc = 100.0 * accurate / batch
        print("test num: %d, accurate num : %d, accrucy : %05.3f%%" %
              (batch, accurate, acc))


if __name__ == "__main__":
    trainingDataset, trainingLabels = DataProcess('dataset/trainingDigits').readDataset()
    testDataset, testLabels = DataProcess('dataset/testDigits').readDataset()
    for sd in range(20, 30):
        net = Net([16,16], sd)  # 中间层各层神经元数量
        net.train(trainingDataset, trainingLabels, Epoch=200)
        plotLosslist(net.trainLoss, "Loss of numpy_clf : seed=" + str(sd))

    net.test(testDataset, testLabels)
