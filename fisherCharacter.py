import os
import cv2 as cv
import numpy as np

cpText = ["京", "津", "沪", "渝", "冀", "豫"
    , "云", "辽", "黑", "湘", "皖", "鲁", "新"
    , "苏", "浙", "赣", "桂", "甘", "晋", "蒙"
    , "陕", "吉", "闽", "贵", "粤", "青", "藏"
    , "琼", "使", "布", "艺", "博"]


def getTraingData(target, broTarget):
    """getTraingData
        根据传入字符找到样本，进行特征提取后，将正负样本返回。

        :argument
            posStr:正类字符
            negStr:负类字符
        :returns
            w1:正类样本训练集
            w2:负累样本训练集
    """
    label = open("label.txt")
    w1 = []
    w2 = []
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
        ret, imgviewx2 = cv.threshold(~img_gray, 0, 1, cv.THRESH_BINARY )
        # 35*35
        imgviewx2 = cv.resize(cropImg(imgviewx2), (35, 35))
        # 7*7
        imgFeatrue = []
        # 5*5的遍历
        for i in range(0, 35, 5):
            for j in range(0, 35, 5):
                temMatrix = imgviewx2[i:i+4, j:j+4]
                sum = np.count_nonzero(temMatrix)
                imgFeatrue.append(sum/25)
        # 1*49
        imgFeatrue = np.array(imgFeatrue)

        if value == target:
            w1.append(imgFeatrue)
        elif value == broTarget:
            w2.append(imgFeatrue)
    nw1 = np.array(w1)
    nw2 = np.array(w2)
    # axis = k;压缩k维方向
    # mean1 = np.mean(w1, axis=0)
    # print(mean1)
    return nw1, nw2


def train(w1, w2, pos, neg):
    """

    :param w1: 正类样本训练集
    :param w2: 负类样本训练集
    :param pos: 正类字符ASCLL码
    :param neg: 负类字符ASCLL码
    """
    # 均值
    m1 = w1.mean(axis=0)
    m2 = w2.mean(axis=0)

    s1 = np.matrix(np.zeros([49, 49]))
    s2 = np.matrix(np.zeros([49, 49]))

    for i in range(w1.shape[0]):
        martixSub = np.matrix(w1[i] - m1)
        # 49*1 * 1*49 = 49*49
        s1 += martixSub.T * martixSub
    # 负类样本
    for i in range(w2.shape[0]):
        martixSub = np.matrix(w2[i] - m2)
        s2 += martixSub.T * martixSub
    w = np.dot(np.linalg.pinv(s1+s2), (m1 - m2))
    print(pos,neg)
    np.save("./data2/w_"+str(pos)+"_"+str(neg)+"_", w)
    np.save("./data2/m1_"+str(pos)+"_"+str(neg)+"_", m1)
    np.save("./data2/m2_"+str(pos)+"_"+str(neg)+"_", m2)
    # return w, m1, m2


def trainMain():
    """
        训练入口函数，穷举出ASCLL码33-126，以及31个中文字符共158个字符的两两结合的所有组合并进行训练。
    """
    for i in range(0, 32):
        for j in range(i+1, 32):
            w1, w2 = getTraingData(cpText[i], cpText[j])
            train(w1, w2, cpText[i], cpText[j])
        print(cpText[i])


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


def test():
    """
    测试

    """
    testLable = open("testLabel.txt")
    ok = 0
    error = 0
    last = ''
    dataSet = testLable.readlines()
    for currentLine in range(941, 1004):
        row = dataSet[currentLine]
        imgPath = row.split(" ")[0]
        realValue = str(row.split(" ")[1]).strip()
        if last == realValue:
            continue
        last = realValue
        image = cv.imread(imgPath)
        img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # 二值化图像，黑白图像，只有0和1,0为0,1为255
        ret, testImg = cv.threshold(~img_gray, 0, 1, cv.THRESH_BINARY)
        testImg = cv.resize(cropImg(testImg), (35, 35))
        imgFeatrue = []
        for i in range(0, 35, 5):
            for j in range(0, 35, 5):
                temMatrix = testImg[i:i + 4, j:j + 4]
                sum = np.count_nonzero(temMatrix)
                imgFeatrue.append(sum / 25)
        # 1*49
        x = np.array(imgFeatrue)
        result = np.zeros(158)
        for i in range(33, 158):
            for j in range(i+1, 158):
                lstr = ""
                rstr = ""
                if i > 126:
                    lstr = cpText[i-127]
                else:
                    lstr = str(i)
                if j > 126:
                    rstr = cpText[j-127]
                else:
                    rstr = str(j)
                tempm1 = []
                tempm2 = []
                w = np.load(os.path.join("./data2/", "w_"+lstr+"_"+rstr+"_"".npy"))
                m1 = np.load(os.path.join("./data2/", "m1_"+lstr+"_"+rstr+"_"".npy"))
                m2 = np.load(os.path.join("./data2/", "m2_"+lstr+"_"+rstr+"_"".npy"))
                tempm1.append(m1)
                tempm2.append(m2)
                nm1 = np.array(tempm1)
                nm2 = np.array(tempm2)
                y = np.reshape(np.dot(w, x.T), -1)
                pos = np.reshape(np.dot(w, nm1.T), -1)
                neg = np.reshape(np.dot(w, nm2.T), -1)
                if np.absolute(y-pos) < np.absolute(y-neg):
                    result[i] += 1
                else:
                    result[j] += 1
        maxIndex = np.argmax(result)
        predValue = ""
        if maxIndex > 126:
            predValue = cpText[maxIndex-127]
        else:
            predValue = chr(maxIndex)
        print("真实值：", realValue, "预测值：", predValue)
        if predValue == realValue:
            ok += 1
        else:
            error += 1
    print("准确率：", (ok/(ok+error))*100, "%")


test()


