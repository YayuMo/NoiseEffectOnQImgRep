import numpy as np
from matplotlib import pyplot as plt
from qiskit.visualization import plot_distribution

from circuits import angleEncodingCircuit, simulate
from distributedCircuit import distributedSimulation
from imageUtil import image2Arr
from imgRev import angleDisReversion
from qiskit_backend import constructBackend

if __name__ == '__main__':
    # imgArr = np.array([0, 125, 200, 255])
    imgArr = image2Arr(imagePath='img/duck.png', size=32)
    qc = angleEncodingCircuit(imgArr)
    shots = 200000
    # qc.draw(output='mpl')
    sim = constructBackend('qasm', 0, qc.num_qubits)
    img = distributedSimulation(imgArr, 8, shots, sim)
    # print(img)
    plt.imshow(img, cmap='gray')
    plt.show()