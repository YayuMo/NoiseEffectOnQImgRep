# inspired by separate circuit
import math

import numpy as np
from tqdm import tqdm

from circuits import angleEncodingCircuit, simulate
from imgRev import angleDisReversion


def distributedSimulation(imgArr, partSize, shots, sim):
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

