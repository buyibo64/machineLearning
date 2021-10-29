from PIL import Image, ImageDraw, ImageFont
from io import StringIO
import random
import os
import numpy as np


def createCharImage():
    labelText = open("testLabel.txt", "w")
    # invalid = [18,19,,41,117,53,55,58,59,62,63,64,65,92,93,94,95]
    for i in range(33, 127):
        temp = chr(i)
        for j in range(0, 10):
            # 每个字体生成一个字符
            # 纯白255
            img = Image.new("RGB", (50, 50), (255, 255, 255))
            randomFont = np.random.randint(1, 110)
            font = ImageFont.truetype(os.path.join("font", "ttf/ttf (" + str(randomFont) + ").ttf"), random.randint(23, 28))
            drawImg = ImageDraw.Draw(img)
            area = (random.randint(10, 15), 10)
            # 将字体画入图片
            drawImg.text(area, temp, font=font, fill="#000000")
            # 图片名
            imgName = str(i) + "_" + str(j) + ".png"
            # 存label
            labelText.write("./testImg/" + imgName + " " + temp + "\n")
            # 存图片
            img.save("./testImg/" + imgName)
# 生成训练样本汉字
def createChineseImage():
    labelText = open("chineseTrainLabel.txt", "a")
    cpText = ["京", "津", "沪", "渝", "冀", "豫"
              , "云", "辽", "黑", "湘", "皖", "鲁", "新"
              , "苏", "浙", "赣", "桂", "甘", "晋", "蒙"
              , "陕", "吉", "闽", "贵", "粤", "青", "藏"
              , "琼", "使", "布", "艺", "博"]
    for i in range(len(cpText)):
        for fontIndex in range(18):
            img = Image.new("RGB", (50, 50), (255, 255, 255))
            font = ImageFont.truetype(os.path.join("font", "cttf/ttf (" + str(fontIndex+1) + ").ttf"), random.randint(23, 28))
            drawImg = ImageDraw.Draw(img)
            area = (random.randint(10, 15), 10)
            # 将字体画入图片
            drawImg.text(area, cpText[i], font=font, fill="#000000")
            # 图片名
            imgName = str(i) + "_" + str(fontIndex) + ".png"
            # 存label
            labelText.write("./chineseTrainImg/" + imgName + " " + cpText[i] + "\n")
            # 存图片
            img.save("./chineseTrainImg/" + imgName)

# 生成测试样本汉字
def createTestChineseImage():
    labelText = open("testLabel.txt", "a")
    cpText = ["京", "津", "沪", "渝", "冀", "豫"
              , "云", "辽", "黑", "湘", "皖", "鲁", "新"
              , "苏", "浙", "赣", "桂", "甘", "晋", "蒙"
              , "陕", "吉", "闽", "贵", "粤", "青", "藏"
              , "琼", "使", "布", "艺", "博"]
    for i in range(len(cpText)):
        for fontIndex in range(2):
            index = np.random.randint(1, 18)
            img = Image.new("RGB", (50, 50), (255, 255, 255))
            font = ImageFont.truetype(os.path.join("font", "cttf/ttf (" + str(index) + ").ttf"), random.randint(23, 28))
            drawImg = ImageDraw.Draw(img)
            area = (random.randint(10, 15), 10)
            # 将字体画入图片
            drawImg.text(area, cpText[i], font=font, fill="#000000")
            # 图片名
            imgName = cpText[i] + "_" + str(fontIndex) + ".png"
            # 存label
            labelText.write("./testImg/" + imgName + " " + cpText[i] + "\n")
            # 存图片
            img.save("./testImg/" + imgName)

# createCharImage()

createTestChineseImage()


# print(np.random.randint(1, 110))
# if __name__ == '__main__':
#     createCharImage();
