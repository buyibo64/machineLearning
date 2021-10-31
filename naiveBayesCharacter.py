import os
import numpy as np
import cv2 as cv

# 先验概率 P(Y = Ck)
pck = round(float(1 / 14), 2)
# 各类字体样本数
category = 14

cpText = ["京", "津", "沪", "渝", "冀", "豫"
    , "云", "辽", "黑", "湘", "皖", "鲁", "新"
    , "苏", "浙", "赣", "桂", "甘", "晋", "蒙"
    , "陕", "吉", "闽", "贵", "粤", "青", "藏"
    , "琼", "使", "布", "艺", "博"]


def train(w, ascll):
    """

    :param w1: 正类样本训练集
    :param w2: 负类样本训练集
    :param pos: 正类字符ASCLL码
    :param neg: 负类字符ASCLL码
    """

    # 图片的特征图(1*49),各个点的频数的可能取值(0-49)
    result = {}
    for i in range(26):
        value = round(float(i / 25), 2)
        frequency = np.zeros(49)
        ones = np.ones_like(frequency)
        for j in range(w.shape[0]):
            for k in range(49):
                if w[j][k] == value:
                    frequency[k] += 1
        # 贝叶斯估计 拉普拉斯平滑
        result[value] = np.array((frequency + ones) / (category + category)).astype(np.float64)
    for key in result.keys():
        np.save("./data3/" + str(ascll) + str(key), result[key])


def getTrainData(target):
    """getTraingData
        根据传入字符找到样本，进行特征提取后，将样本训练集返回。

        :argument
            target:字符
        :returns
            w:样本训练集
    """
    label = open("label.txt")
    w = []
    for line in label.readlines():
        imagePath = line.split(" ")[0].strip()
        value = str(line.split(" ")[1]).strip()
        # 50*50*3 RGB
        image = cv.imread(imagePath)
        # 50*50
        img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # 二值化图像，黑白图像，只有0和1,0为0,1为255
        np.set_printoptions(threshold=np.inf)
        # 50*50
        ret, imgviewx2 = cv.threshold(~img_gray, 0, 1, cv.THRESH_BINARY)
        # 35*35
        imgviewx2 = cv.resize(cropImg(imgviewx2), (35, 35))
        # 7*7
        imgFeature = []
        # 5*5的遍历
        for i in range(0, 35, 5):
            for j in range(0, 35, 5):
                temMatrix = imgviewx2[i:i + 4, j:j + 4]
                sum = np.count_nonzero(temMatrix)
                imgFeature.append(round(float(sum / 25), 2))
        # 1*49
        imgFeature = np.array(imgFeature)
        if value == target:
            w.append(imgFeature)
    # axis = k;压缩k维方向
    # mean1 = np.mean(w1, axis=0)
    # print(mean1)
    return np.array(w)


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


def trainMain():
    """
        训练入口函数，穷举出ASCLL码33-126，以及31个中文字符共158个字符的两两结合的所有组合并进行训练。
    """
    for i in range(33, 159):
        if i <= 126:
            trainStr = chr(i)
        else:
            trainStr = cpText[i - 126 - 1]
        w = getTrainData(trainStr)
        train(w, i)
        print(i, trainStr)


def test():
    """
      测试

      """
    testLabel = open("testLabel.txt")
    ok = 0
    error = 0
    for row in testLabel.readlines():
        imgPath = row.split(" ")[0]
        realValue = str(row.split(" ")[1]).strip()
        image = cv.imread(imgPath)
        img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # 二值化图像，黑白图像，只有0和1,0为0,1为255
        ret, testImg = cv.threshold(~img_gray, 0, 1, cv.THRESH_BINARY)
        testImg = cv.resize(cropImg(testImg), (35, 35))
        imgFeature = []
        for i in range(0, 35, 5):
            for j in range(0, 35, 5):
                temMatrix = testImg[i:i + 4, j:j + 4]
                sum = np.count_nonzero(temMatrix)
                imgFeature.append(round(float(sum / 25), 2))
        # Y = ascll
        result = np.zeros(159)
        for ascll in range(33, 159):
            # X = item Y = ascll
            multiplyPxy = pck
            denominator = 0
            for index in range(49):
                pxy = np.load(os.path.join("./data3/", str(ascll) + str(imgFeature[index]) + ".npy"))
                multiplyPxy *= pxy[index]
                denominator += multiplyPxy
            result[ascll] = multiplyPxy / denominator
        maxIndex = np.argmax(result)
        if maxIndex > 126:
            predValue = cpText[maxIndex - 127]
        else:
            predValue = chr(maxIndex)
        if predValue == realValue:
            ok += 1
        else:
            error += 1
        print("真实值:", realValue, "预测值:", predValue)
    print("准确率:", ok/ok+error)

# trainMain()

test()

