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
    img = Image.fromarray(dist_Mat)
    # print(dist_Mat)
    return img

# retrieve img from angle encoding