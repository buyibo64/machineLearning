import os
import numpy as np
import cv2 as cv

cpText = ["京", "津", "沪", "渝", "冀", "豫"
    , "云", "辽", "黑", "湘", "皖", "鲁", "新"
    , "苏", "浙", "赣", "桂", "甘", "晋", "蒙"
    , "陕", "吉", "闽", "贵", "粤", "青", "藏"
    , "琼", "使", "布", "艺", "博"]


def getTrainData(posStr, negStr):
    """getTrainData
        根据传入字符找到样本，进行特征提取后，将正负样本返回。

        :argument
            posStr:正类字符
            negStr:负类字符
        :returns
            w1:正类样本训练集
            w2:负累样本训练集
    """
    label = open("label.txt", "r")
    w1 = []
    w2 = []
    for line in label.readlines():
        imagePath = line.split(" ")[0].strip()
        value = line.split(" ")[1].strip()
        image = cv.imread(imagePath)
        img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # 二值化图像，黑白图像，只有0和1,0为0,1为255
        np.set_printoptions(threshold=np.inf)
        ret, binaryImage = cv.threshold(~img_gray, 0, 1, cv.THRESH_BINARY)
        binaryImage = cv.resize(cropImg(binaryImage), (35, 35))
        imgFeature = []
        for i in range(0, 35, 5):
            for j in range(0, 35, 5):
                tempMatrix = binaryImage[i:i + 4, j:j + 4]
                nonZeroCount = np.count_nonzero(tempMatrix)
                imgFeature.append(nonZeroCount / 25)
        # 1*49
        imgFeature = np.array(imgFeature)
        if value == posStr:
            w1.append(imgFeature)
        if value == negStr:
            w2.append(imgFeature)
    return np.array(w1), np.array(w2)


def trainMain():
    """
        训练入口函数，穷举出ASCLL码33-126，以及31个中文字符共158个字符的两两结合的所有组合并进行训练。
    """
    for i in range(33, 158):
        for j in range(i + 1, 158):
            if i < 126:
                posStr = chr(i)
            else:
                posStr = cpText[i - 126]
            if j < 126:
                negStr = chr(j)
            else:
                negStr = cpText[j - 126]
            w1, w2 = getTrainData(posStr, negStr)
            train(w1, w2, i, j)
            print(i, j)


def cropImg(image):
    """
    裁剪传入图片内的字符并返回

    :params image:待裁剪图片

    :return image:裁剪后的图片
    """
    height = image.shape[0]
    width = image.shape[1]
    sumHeight = np.sum(image, axis=1)
    sumWidth = np.sum(image, axis=0)
    left = 0
    right = 0
    top = 0
    bottom = 0
    for i in range(width):
        if sumWidth[i] != 0:
            left = i
            break
    for i in range(width - 1, -1, -1):
        if sumWidth[i] != 0:
            right = i
            break
    for i in range(height):
        if sumHeight[i] != 0:
            top = i
            break
    for i in range(height - 1, -1, -1):
        if sumHeight[i] != 0:
            bottom = i
            break
    return image[top: bottom + 1, left: right + 1]


def train(w1, w2, pos, neg):
    """

    :param w1: 正类样本训练集
    :param w2: 负类样本训练集
    :param pos: 正类字符ASCLL码
    :param neg: 负类字符ASCLL码
    """

    # 初始权重为w
    w = np.zeros((1, 49))
    # 初始偏置置为0
    b = 0
    # 初始学习率
    n = 0.03
    while True:
        # 训练结束flag
        flag = True
        batch = 10
        if pos == 46 and neg == 73 or pos == 46 and neg == 95 or pos == 46 and neg == 108 or pos == 46 and neg == 124:
            batch = 1
        for i in range(batch):
            pred = (np.dot(w, w1[i].T)) + b
            if pred <= 0:
                flag = False
                w = w + (n * w1[i])
                b = b + n
        for j in range(batch):
            pred = (np.dot(w, w2[j].T)) + b
            if pred >= 0:
                flag = False
                w = w - (n * w2[j])
                b = b - n
        if flag:
            break
    np.save("./perceptionData/w_" + str(pos) + "_" + str(neg) + "_", w)
    np.save("./perceptionData/b_" + str(pos) + "_" + str(neg) + "_", b)


def test():
    """
    测试

    """
    testLable = open("testLabel.txt")
    # 预测正确数量
    ok = 0
    # 预测错误数量
    error = 0
    dataSet = testLable.readlines()
    for row in dataSet:
        # 图片路径
        imgPath = row.split(" ")[0]
        # 真实值
        realValue = str(row.split(" ")[1]).strip()
        # 各字符得分
        result = np.zeros(158)
        image = cv.imread(imgPath)
        img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # 二值化图像，黑白图像，只有0和1,0为0,1为255
        ret, testImg = cv.threshold(~img_gray, 0, 1, cv.THRESH_BINARY)
        # 裁剪图片并resize为统一大小 35*35
        testImg = cv.resize(cropImg(testImg), (35, 35))
        x = []
        # 提取特征
        for i in range(0, 35, 5):
            for j in range(0, 35, 5):
                temMatrix = testImg[i:i + 4, j:j + 4]
                sum = np.count_nonzero(temMatrix)
                x.append(sum / 25)
        # 1*49
        x = np.array(x)
        # 使用各w与b识别字符
        for i in range(33, 158):
            for j in range(i + 1, 158):
                w = np.load(os.path.join("./perceptionData/", "w_" + str(i) + "_" + str(j) + "_"".npy"))
                b = np.load(os.path.join("./perceptionData/", "b_" + str(i) + "_" + str(j) + "_"".npy"))
                pred = np.dot(w, x.T) + b
                if pred > 0:
                    result[i] += 1
                else:
                    result[j] += 1
        # 分数最高字符下标
        maxIndex = np.argmax(result)
        if maxIndex > 126:
            # 预测值
            predValue = cpText[maxIndex - 126]
        else:
            predValue = chr(maxIndex)
        print("真实值：", realValue, "预测值：", predValue)
        if predValue == realValue:
            ok += 1
        else:
            error += 1
    print("准确率：", (ok / (ok + error)) * 100, "%")


test()
