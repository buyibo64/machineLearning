import matplotlib.pyplot as plt
import numpy as np


def drawBar():
    x = ['Breast', 'Bupa', 'CMC', 'Diabets','Echoncardiogram',
         'Haberman', 'Heart_disease', 'Heart-statlog', 'Hepatitis', 'Ionosphere',
         'PimaIndian', 'Seeds', 'Sonar', 'Spect', 'Wdbc']
    # 时间
    # svm = [0.01482, 0.01208, 0.10128, 0.03723, 0.00179,
    #        0.00446, 0.00637, 0.00477, 0.00172, 0.00863,
    #        0.03241, 0.00094, 0.00352, 0.00641, 0.00748]
    # LDA = [0.00173, 0.00101, 0.00280, 0.00177, 0.00104,
    #        0.00078, 0.00107, 0.00184, 0.00120, 0.00303,
    #        0.00965, 0.00311, 0.00276, 0.00278, 0.00532]
    # OSVC = [0.00461, 0.00345, 0.04558, 0.02487, 0.00059,
    #         0.00164, 0.00429, 0.00382, 0.00161, 0.00299,
    #         0.02382, 0.00060, 0.00377, 0.00452, 0.00568]
    # THSVM = [0.00658, 0.00307, 0.04861,0.01148, 0.00107,
    #          0.00162, 0.00208, 0.00576, 0.00168, 0.00307,
    #          0.01486, 0.00361, 0.00185, 0.00251, 0.00547]
    # MMHS_SVM = [0.00327, 0.0028, 0.04756, 0.00777, 0.00081,
    #             0.00082, 0.00174, 0.00183, 0.00136, 0.00236,
    #             0.00879, 0.00251, 0.00144, 0.00181, 0.00454]

    # 支持向量个数
    svm = [323, 239,1005, 538, 93,
           208, 206, 189, 110, 200,
           538, 114, 146, 188, 347]
    LDA = [479, 242, 1032, 538, 93,
           215, 207, 189, 110, 200,
           538, 147, 146, 188, 400]
    OSVC = [212, 241, 595, 538, 93,
            189, 207, 189, 110, 103,
            538, 97, 137, 188, 368]
    THSVM = [479, 242, 1032, 538, 93,
             215, 207, 189, 110, 200,
             538, 147, 146, 188, 400]
    MMHS_SVM = [23, 234, 1012, 98, 54,
                35, 163, 74, 87, 120,
                72, 55, 128, 124, 333]

    #精度
    # svm = [96.0784, 64.0485, 79.1837, 65.913, 81.8421,
    #        75.1143, 64.8276, 59.3827, 80, 90.3062,
    #        65.9565, 94.4444, 88.5484, 79.7468, 82.3682]
    # LDA = [95.3431, 62.6214, 64.6712, 76.6087, 83.3684,
    #        74.3956, 82.2989, 83.4568, 82, 85.9615,
    #        70.9130, 93.0159, 76.4516, 67.5949, 83.5794]
    # OSVC = [95.6863, 59.4175, 77.3016, 65.2174, 71.0526,
    #         73.2967, 64.3678, 55.5556, 80, 92.4038,
    #         65.2174, 86.5079, 59.5161, 79.7468, 81.7783]
    # THSVM = [96.5686, 64.6602, 76.9615, 66.8667, 82.1053,
    #          61.8681, 63.908, 60.3704, 66.2222, 90.1987,
    #          66.8261, 89.0476, 83.7097, 78.481, 84.7337]
    # MMHS_SVM = [96.8627, 64.6602, 76.8481, 70.8667, 84.4737,
    #             75.6044, 65.6322, 64.0741, 75.7778, 90.3311,
    #             70.913,  90.7937, 85.3226, 79.3671, 85.503]
    startXLocation = np.arange(5)
    barWidth = 0.15
    width = barWidth / 3
    # fig, ax = plt.subplots(figsize=(30, 15), nrows=2, ncols=2)
    fig, ax = plt.subplots(figsize=(30, 15))
    ax.bar(startXLocation - 2*barWidth, svm[5:10], barWidth, label='SVM')
    ax.bar(startXLocation - barWidth, LDA[5:10], barWidth, label='LDA')
    ax.bar(startXLocation, OSVC[5:10], barWidth, label='OSVC')
    ax.bar(startXLocation + barWidth, THSVM[5:10], barWidth, label='THSVM')
    ax.bar(startXLocation + 2*barWidth, MMHS_SVM[5:10], barWidth, label='MMHS-SVM')
    ax.set_xticks(startXLocation)
    ax.set_xticklabels(x[5:10])
    plt.xlabel("Datasets", fontsize=26)
    # Average Classification Accuracy
    # Number of Support Vectors
    plt.ylabel("Number of Support Vectors", fontsize=26)
    ax.legend( fontsize=26, loc=2, bbox_to_anchor=(0.65, 1))
    plt.xticks(fontsize=26)
    plt.show()
    # plt.bar(x, num_list, width=width, label='boy', fc='y')
    # for i in range(len(x)):
    #     x[i] = x[i] + width
    # plt.bar(x, num_list1, width=width, label='girl',  fc='r')
    # plt.legend()
    # plt.show()


drawBar()
