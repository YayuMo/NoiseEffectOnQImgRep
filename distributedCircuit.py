# inspired by separate circuit
import math
import re

import numpy as np
from tqdm import tqdm

from circuits import angleEncodingCircuit, simulate, basisEncoding
from imgRev import angleDisReversion, basisEnReversion


# angle distributed simulation
def distributedAngleSimulation(imgArr, partSize, shots, sim):
    partNum = int(len(imgArr) / partSize)
    img = imgArr.reshape((partNum, partSize))
    processedImg = []
    for arr in tqdm(img):
        qc = angleEncodingCircuit(arr)
        dist = simulate(qc, shots, sim)
        img = angleDisReversion(dist, arr, shots, reshape = False)
        processedImg.append(img)
    processedImg = np.array(processedImg)
    size = int(math.sqrt(len(imgArr)))
    return processedImg.reshape((size, size))

# basis distributed simulation
def distributedBasisSimulation(imgArr, partSize, shots, sim):
    partNum = int(len(imgArr) / partSize)
    img = imgArr.reshape((partNum, partSize))
    # print(img)
    processedImg = []
    for arr in tqdm(img):
        qc, encoding = basisEncoding(arr)
        dist = simulate(qc, shots, sim)
        # print(dist)
        key = list(dist.keys())[0]
        keyList = re.findall(r'\w{8}', key)
        processedImg.append(keyList)
    processedImg = np.array(processedImg)
    processedImg = processedImg.reshape((1, len(imgArr)))[0]
    # print(processedImg)
    retrievedImg = basisEnReversion(processedImg, imgArr)
    return retrievedImg

# MCRQI distributed simulation



