from PIL import Image
import math
import numpy as np
from imageUtil import to_bin

# generate all possible quantum state by n-qubit
def generateKeySet(n):
    # n = # of qubit
    keySet = list(range(0, 2 ** n, 1))
    for i in range(len(keySet)):
        keySet[i] = to_bin(keySet[i], n)
    return keySet

# retrieve img from amplitude encoding
def ampDisReversion(dist, keyset, sqSum, shots, n):
    for key in keyset:
        if key in dist:
            dist[key] = math.sqrt(abs(dist[key] * sqSum / shots))
        else:
            dist[key] = 0
    sorted_dict = dict(sorted(dist.items()))
    sorted_dist = np.array(list(sorted_dict.values()))
    size = int(2 ** (n / 2))
    dist_Mat = sorted_dist.reshape((size, size))
    # img = Image.fromarray(dist_Mat)
    # print(dist_Mat)
    return dist_Mat

# retrieve img from angle encoding
def angleDisReversion(dist, imgArr, shots, reshape):
    pixels = np.zeros((len(imgArr)))
    for item in dist:
        for i, bit in enumerate(item):
            if bit == '0':
                # print(dist[item])
                pixels[i] += dist[item]
    # print(pixels)
    size = int(math.sqrt(len(imgArr)))
    reconstruct = []
    for pixel in pixels:
        color = 2 * np.arccos((pixel / shots) ** (1 / 2)) * 100
        reconstruct.append(color)
    sorted_dist = np.array(reconstruct)
    if reshape == True:
        dist_Mat = sorted_dist.reshape((size, size))
        return dist_Mat
    else:
        return sorted_dist

# retrieve img from dense angle encoding
def denseAngleDisReversion(dist, imgArr, shots, reshape):
    pixelsEven = np.zeros((int(len(imgArr) / 2)))
    for item in dist:
        for i, bit in enumerate(item):
            if bit == '0':
                print(dist[item])
                pixelsEven[i] += dist[item]
    # print(pixels)
    size = int(math.sqrt(len(imgArr)))
    reconstruct = []
    for pixel in pixelsEven:
        color = 2 * np.arccos((pixel / shots) ** (1 / 2)) * 100
        reconstruct.append(color)
    sorted_dist = np.array(reconstruct)
    if reshape == True:
        dist_Mat = sorted_dist.reshape((size, size))
        return dist_Mat
    else:
        return sorted_dist

# retrieve img from basis encoding
def basisEnReversion(pixelList, imgArr):
    transferredPixel = []
    for pixel in pixelList:
        transferredPixel.append(int(pixel, 2))
    processed_pixel = np.array(transferredPixel)
    size = int(math.sqrt(len(imgArr)))
    dist_Mat = processed_pixel.reshape((size, size))
    return dist_Mat

# retrieve img from dense angle encoding
def DenseAngleReversion(dist, imgArr, shots, reshape):
    pass

# retrieve img from dense angle encoding
