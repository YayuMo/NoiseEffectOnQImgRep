import numpy as np
from matplotlib import pyplot as plt
from qiskit.visualization import plot_distribution

from circuits import angleEncodingCircuit, simulate, basisEncoding
from distributedCircuit import distributedAngleSimulation, distributedBasisSimulation
from imageUtil import image2Arr
from imgRev import angleDisReversion, basisEnReversion
from qiskit_backend import constructBackend

if __name__ == '__main__':
    # imgArr = np.array([0, 125, 200, 255])
    imgArr = image2Arr(imagePath='img/duck.png', size=32)
    qc, encoding = basisEncoding(imgArr)
    # qc = angleEncodingCircuit(imgArr) # angle encoding test
    shots = 1
    # qc.draw(output='mpl')
    sim = constructBackend('qasm', 0, qc.num_qubits)
    img = distributedBasisSimulation(imgArr, 2, shots, sim)
    # print(img)
    plt.imshow(img, cmap='gray')
    plt.show()