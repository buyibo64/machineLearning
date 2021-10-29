import numpy as np
import cv2 as cv

# 先验概率 P(Y = Ck)
pck = 1 / 94


def train():
    labelText = open("label.txt")
    dataSet = labelText.readlines()
    size = len(dataSet)
    probability = np.zeros((94, 2500))
    temp = np.array([])
    for i in range(94):
        sumPix = np.zeros((50, 50))
        for j in range(109):
            line = dataSet[(i * 109)+j]
            imagePath = line.split(" ")[0]
            value = str(line.split(" ")[1]).strip()
            image = cv.imread(imagePath)
            img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            # 二值化图像，黑白图像，只有0和1,0为0,1为255
            ret, binaryImage = cv.threshold(~img_gray, 0, 1, cv.THRESH_BINARY)
            sumPix += binaryImage
            temp = sumPix.reshape(-1)
        for k in range(len(temp)):
            # 拉普拉斯平滑
            probability[i, k] = temp[k]+109/(3 * 109)
    return probability




train()
