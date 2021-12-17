import numpy as np

from DataProcess import DataProcess
from plot import plotLosslist
import pickle
import random


def Sigmoid(x, diff=False):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def dsigmoid(x):
        f = sigmoid(x)
        return f * (1 - f)

    return sigmoid(x) if diff == False else dsigmoid(x)


def squareErrorSum(y_hat, y, diff=False):
    return (np.square(y_hat - y) * 0.5).sum() if diff == False else y_hat - y


class Net():
    def __init__(self, hiddenLayers):
        self.n = len(hiddenLayers)
        # X Input
        np.random.seed(8)
        random.seed(8)
        self.X = np.random.randn(1024, 1)

        self.w = [np.random.randn(hiddenLayers[0], 1024)]
        self.b = [np.random.randn(hiddenLayers[0], 1)]
        for i in range(1, self.n):
            self.w.append(np.random.randn(
                hiddenLayers[i], hiddenLayers[i - 1]))
            self.b.append(np.random.randn(hiddenLayers[i], 1))

        # self.w[0] = np.random.randn(16, 1024)
        # self.W[1] = np.random.randn(16, 16)

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

        # self.z[0] = np.dot(self.w[0], self.X) + self.b[0]
        # self.a[0] = activate(self.z[0])
        # self.z[1] = np.dot(self.w[1], self.a[0]) + self.b[1]
        # self.a[1] = activate(self.z[1])
        # self.z[2] = np.dot(self.w[2], self.a[2]) + self.b[2]
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

        # self.delta[2] = activate(self.z[2], True) * \
        #     squareErrorSum(self.y_hat, y, True)
        # self.delta[1] = activate(self.z[1], True) * \
        #     (np.dot(self.w[2].T, self.delta[2]))
        # self.delta[0] = activate(self.z[0], True) * \
        #     (np.dot(self.w[1].T, self.delta[1]))
        # dw[2] = np.dot(self.delta[2], self.a[1].T)
        # dw[1] = np.dot(self.delta[1], self.a[0].T)
        # dw[0] = np.dot(self.delta[0], self.X.T)
        # d[2] = self.delta[2]
        # d[1] = self.delta[1]
        # d[0] = self.delta[0]

        # update weight
        for i in range(self.n + 1):
            self.w[i] -= self.alpha * dw[i]
            self.b[i] -= self.alpha * self.delta[i]
        # self.w[2] -= self.alpha * dw[2]
        # self.w[1] -= self.alpha * dw[1]
        # self.w[0] -= self.alpha * dw[0]
        # self.b[2] -= self.alpha * self.delta[2]
        # self.b[1] -= self.alpha * self.delta[1]
        # self.b[0] -= self.alpha * self.delta[0]

    def setLearningRate(self, l):
        self.alpha = l

    def save(self, path):
        obj = pickle.dumps(self)
        with open(path, "wb") as f:
            f.write(obj)

    def load(path):
        obj = None
        with open(path, "rb") as f:
            try:
                obj = pickle.load(f)
            except:
                print("IOError")
        return obj

    def train(self, trainingDataset, trainingLabels, Epoch=5, batch=None, shuffle=False):
        if shuffle == True:
            tmp = list(zip(trainingDataset, trainingLabels))
            for i in range(100):
                random.shuffle(tmp)
            trainingDataset[:], trainingLabels[:] = zip(*tmp)
        for epoch in range(Epoch):
            acc = 0.0
            acc_cnt = 0
            # 先生成一个10x1是向量，减少运算。用于生成one_hot格式的label
            # label = np.zeros([10, 1], int)
            for i in range(len(trainingDataset)):  # 可以用batch，数据较少，一次训练所有数据集
                X = trainingDataset[i, :].reshape([1024, 1])  # 生成输入

                # labelidx = trainingLabels[i]
                # label[labelidx][0] = 1.0
                # print(label)
                label = trainingLabels[i, :].reshape([10, 1])

                Loss, y_hat = self.forward(X, label, Sigmoid)  # 前向传播
                self.backward(label, Sigmoid)  # 反向传播

                # label[labelidx][0] = 0.0  # 还原为0向量
                # print(np.argmax(label))
                acc_cnt += int(np.argmax(label) == np.argmax(y_hat))

            acc = acc_cnt / len(trainingDataset)
            self.trainLoss.append(Loss)
            print("epoch:%d,loss:%02f,accrucy : %02f%%" %
                  (epoch + 1, Loss, acc*100))

    def test(self, testDataset, testLabels, batch=None, shuffle=False):
        if shuffle == True:
            tmp = list(zip(testDataset, testLabels))
            for i in range(100):
                random.shuffle(tmp)
            testDataset[:], testLabels[:] = zip(*tmp)
        acc = 0.0
        acc_cnt = 0
        # label = np.zeros([10, 1], int)  # 先生成一个10x1是向量，减少运算。用于生成one_hot格式的label
        if(batch == None):
            batch = len(testDataset)
        for i in range(batch):  # 可以用batch，数据较少，一次训练所有数据集
            X = testDataset[i, :].reshape([1024, 1])  # 生成输入

            # labelidx = testLabels[i]
            # label[labelidx][0] = 1.0
            label = testLabels[i, :].reshape([10, 1])

            Loss, y_hat = self.forward(X, label, Sigmoid)  # 前向传播

            # label[labelidx][0] = 0.0  # 还原为0向量
            acc_cnt += int(np.argmax(label) == np.argmax(y_hat))
        acc = acc_cnt / batch
        print("test num: %d, accurate num : %d, accrucy : %05.3f%%" %
              (batch, acc_cnt, acc*100))


if __name__ == "__main__":
    trainingDataset, trainingLabels = DataProcess(
        'dataset/trainingDigits').readDataset()
    net = Net([16,16])  # 中间层各层神经元数量
    net.setLearningRate(0.1)
    net.train(trainingDataset, trainingLabels, Epoch=100)
    plotLosslist(net.trainLoss, "Loss of numpy_clf : alpha=" + str(net.alpha))

    testDataset, testLabels = DataProcess('dataset/testDigits').readDataset()
    net.test(testDataset, testLabels)
    # net.save("hr.model")

    # newmodel = Net.load("hr.model")
    # newmodel.test(testDataset, testLabels)
