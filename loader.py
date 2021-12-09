import numpy as np

# 加载训练数据
def img2vector(path):  # 将图片矩阵展开为一列向量
    ret = np.zeros([1024], int)  # 返回矩阵大小为 1 * 1024
    f = open(path)
    lines = f.readlines()  # 读取文件各行
    for i in range(32):
        for j in range(32):
            ret[i * 32 + j] = lines[i][j]  # 将二维数组拉成一维
    return ret
