import numpy as np
from matplotlib import pyplot as plt
from qiskit.visualization import plot_distribution

from circuits import angleEncodingCircuit, simulate
from imgRev import angleDisReversion
from qiskit_backend import constructBackend

if __name__ == '__main__':
    imgArr = np.array([125, 200])
    qc = angleEncodingCircuit(imgArr)
    shots = 20000
    qc.draw(output='mpl')
    sim = constructBackend('qasm', 0, qc.num_qubits)
    dist = simulate(qc, shots, sim)
    print(dist)
    # plot_distribution(dist)
    # plt.show()
    angleDisReversion(dist, imgArr, shots)
