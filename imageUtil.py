import math
import os

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim

import cv2

# generate n-bit binary data
def to_bin(value, num):
    bin_chars = ""
    temp = value
    for i in range(num):
        bin_char = bin(temp % 2)[-1]
        temp = temp // 2
        bin_chars = bin_char + bin_chars
    return bin_chars.upper()

# image to array
def image2Arr(imagePath, size):
    # read image and convert to gray scale
    img = Image.open(imagePath).convert('L')
    if size != 'NoResize':
        img = img.resize((size, size))
    ls = []
    arr = np.array(img)
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            ls.append(arr[i][j])
    vec = np.array(ls)
    return vec

# array to image
def arr2Image(arr):
    size = int(math.sqrt(len(arr)))
    arr = arr.reshape(size, size)
    img = Image.fromarray(arr)
    return img

# evaluate the mse and ssim between 2 images
def imageEval(imgpath1, imgpath2):
    img1 = cv2.imread(imgpath1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.imread(imgpath2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    mse = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    mse /= float(img1.shape[0] * img1.shape[1])
    sim = ssim(img1, img2)
    # print(mse, sim)
    return mse, sim

# save image
def imageSave(img, prefix, resultHome, params):
    if not os.path.exists(resultHome):
        os.makedirs(resultHome)
    img1 = Image.fromarray(img.astype(np.uint8))
    img2 = img1.convert('RGB')
    file_path = resultHome + prefix + str(int(params * 100)) + '.jpg'
    img2.save(file_path)
    return file_path
    # if img.mode == "F":
    #     img = img.convert('RGB')
    #     file_path = resultHome + prefix + str(int(params * 100)) + '.jpg'
    #     img.save(file_path)
    #     return file_path
    # else:
    #     img = img.convert('RGB')
    #     file_path = resultHome + prefix + str(int(params * 100)) + '.jpg'
    #     img.save(file_path)
    #     return file_path

# image plot
def imgPlot(imgDictList, type):
    n = int(math.sqrt(len(imgDictList)))
    for i in range(len(imgDictList)):
        image = Image.open(imgDictList[i]['img_path'])
        plt.subplot(n, n , i+1)
        plt.imshow(image, cmap='gray')
        plt.title(imgDictList[i]['title'])
    if type == 'diff':
        plt.savefig('result/diff.png')
    else:
        plt.savefig('result/output.png')
    plt.show()

# plot evaluation curve
def plotEvalCurve(imgDictList):
    params = []
    mses = []
    ssims = []
    for i in range(2, len(imgDictList)):
        params.append(imgDictList[i]['param'])
        mses.append(imgDictList[i]['mse'])
        ssims.append(imgDictList[i]['ssim'])

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    ax1.plot(params, mses, color='red', linestyle='-', label='MSE', marker='*')
    ax2.plot(params, ssims, color='blue', linestyle='-', label='SSIM', marker='o')
    fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    ax1.set_xlabel('Params')
    ax1.set_ylabel('MSE')
    ax2.set_ylabel('SSIM')

    plt.savefig('result/eval.png')
    plt.show()

# auto label
def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# overall evaluation -- average SSIM, MSE, Weighted Diverse
def overallPerformanceEval(imgDictList):
    params = []
    mses = []
    ssims = []
    for i in range(2, len(imgDictList)):
        params.append(imgDictList[i]['param'])
        mses.append(imgDictList[i]['mse'])
        ssims.append(imgDictList[i]['ssim'])
    n = len(params)
    sumList = []
    for i in range(len(params)):
        sumList.append(math.sqrt((ssims[i] * (10 ** (3)) - mses[i] * (10 ** (-2))) ** 2) * params[i])

    avgMSE = sum(mses) / n
    avgSSIM = sum(ssims) / n
    weightedDiv = sum(sumList) / n
    return avgMSE, avgSSIM, weightedDiv

# plot comparison distribution
def plotCompareDistribution(keyset, dist1, dist2, labels):
    dist1_sum = sum(dist1.values())
    dist2_sum = sum(dist2.values())
    sorted_dist1 = dict(sorted(dist1.items()))
    sorted_dist2 = dict(sorted(dist2.items()))
    for key in keyset:
        if key in dist1:
            sorted_dist1[key] = round(sorted_dist1[key] / dist1_sum, 3)
        else:
            sorted_dist1[key] = 0.000

    for key in keyset:
        if key in dist2:
            sorted_dist2[key] = round(sorted_dist2[key] / dist2_sum, 3)
        else:
            sorted_dist2[key] = 0.000

    x = np.arange(len(keyset)) # label location
    width = 0.35 # width of bars

    fig, ax = plt.subplots()
    rect1 = ax.bar(x - width / 2, sorted_dist1.values(), width, label=labels[0])
    rect2 = ax.bar(x + width / 2, sorted_dist2.values(), width, label=labels[1])

    ax.set_ylabel('Quasi-Probability')
    ax.set_title('Comparison between Distributions')
    ax.set_xticks(x)
    ax.set_xticklabels(keyset)
    ax.legend()

    autolabel(rect1, ax)
    autolabel(rect2, ax)
    fig.tight_layout()
    plt.grid(which='major', axis='y', zorder=0, linestyle='--')
    plt.show()

# image Fourier Transform


if __name__ == '__main__':
    image1 = Image.open('result/AmpEn_AmpDam/Encoded0.jpg')
    image2 = Image.open('result/AmpEn_AmpDam/ampDamp15.jpg')
    image3 = Image.open('result/AmpEn_AmpDam/Diff15.jpg')
    plt.subplot(1,3,1)
    plt.imshow(image1)
    plt.title('Encoded Image')
    plt.axis(False)
    plt.subplot(1,3,2)
    plt.imshow(image2)
    plt.title('Noised Image')
    plt.axis(False)
    plt.subplot(1,3,3)
    plt.imshow(image3)
    plt.title('Subtracted Image')
    plt.axis(False)
    # list1 = [1,2,3,4]
    # list2 = [2,3,4,5]
    # arr1 = np.array(list1)
    # arr2 = np.array(list2)
    # print(arr2-arr1)
    # keyset = ['00', '01', '10','11']
    # dist1 = {
    #     '00': 1,
    #     '10': 3,
    #     '01': 2,
    #     '11': 4
    # }
    # dist2 = {
    #     '01': 4,
    #     '11': 6,
    #     '10': 5
    # }
    # plotCompareDistribution(keyset, dist1, dist2, '00')
    # vec = image2Arr('img/duck.png')
    # print(len(vec))
    # img = arr2Image(vec)
    # img.show()
    plt.show()