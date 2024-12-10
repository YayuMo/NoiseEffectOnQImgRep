import numpy as np
from matplotlib import pyplot as plt
from qiskit.visualization import plot_distribution

from circuits import angleEncodingCircuit, simulate, basisEncoding, qramEncoding
from distributedCircuit import distributedAngleSimulation, distributedBasisSimulation
from imageUtil import image2Arr
from imgRev import angleDisReversion, basisEnReversion
from qiskit_backend import constructBackend

if __name__ == '__main__':
    # imgArr = np.array([0, 125, 200, 255])
    imgArr = image2Arr(imagePath='img/duck.png', size=4)
    # qc, encoding = basisEncoding(imgArr)
    # qc = angleEncodingCircuit(imgArr) # angle encoding test
    qc = qramEncoding(imgArr)
    shots = 20000
    # qc.draw(output='mpl')
    sim = constructBackend('stabilizer', 0, qc.num_qubits, [])
    dist = simulate(qc, shots=shots, backend=sim)
    plot_distribution(dist)
    # img = distributedBasisSimulation(imgArr, 2, shots, sim)
    # img = distributedAngleSimulation(imgArr, 4, shots, sim)
    # print(img)
    # plt.imshow(img, cmap='gray')
    plt.show()