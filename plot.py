import matplotlib.pyplot as plt
import os

def plotLosslist(Loss, title):
    font = {'family': 'simsun',
            'weight': 'bold',
            'size': 20,
            }
    m = len(Loss)
    X = range(m)
    plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
    plt.subplot(111)
    plt.title(title, font)
    plt.plot(X, Loss)
    plt.xlabel(r'Epoch', font)
    plt.ylabel(u'Loss', font)

    # 保存图片
    fileName = title.split()[2]
    path = fileName + '/' + title.split(':')[-1].strip() + '.tif'
    if not os.path.exists(fileName):
        os.mkdir(fileName)
    plt.savefig(path)
    # plt.show()
