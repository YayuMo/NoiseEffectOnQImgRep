import cmath

import numpy as np
import math
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer, AerSimulator
from qiskit_ibm_runtime import SamplerV2 as Sampler
# from qiskit_ibm_runtime import SamplerV1 as Sampler
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_distribution
from noise_model import constructAmplitudeDampingNoiseModel, constructDepolarizationNoiseModel
import matplotlib

from QRAM import QRAM_circuit
from imageUtil import to_bin, plotCompareDistribution, image2Arr
from imgRev import generateKeySet, ampDisReversion
from qiskit_backend import constructBackend

import matplotlib.pyplot as plt

# Basis Encoding
def basisEncoding(imgArr):
    encodings = ""
    for i in range(len(imgArr)):
        encodings += to_bin(imgArr[i], 8)
    n = len(encodings)
    qc = QuantumCircuit(n, n)
    for index, bit in enumerate(encodings):
        if encodings[index] == '1':
            qc.x(index)
    qc.measure_all()
    # qc.draw(output='mpl')
    # plt.show()
    return qc, encodings

# Angle Encoding -- Qubit Lattice
def angleEncodingCircuit(imgArr):
    # data normalization
    norm = []
    for data in imgArr:
        norm.append((data * math.pi) / 255.0)
    n = len(imgArr)

    # circuit
    qc = QuantumCircuit(n,n)
    for i in range(n):
        qc.ry(norm[i], i)
    qc.measure_all()

    # qc.draw(output='mpl')
    # plt.show()
    return qc

# Dense Angle Encoding
def denseangleEncoding(imgArr):
    # data normalization
    norm = []
    for data in imgArr:
        norm.append((data * math.pi) / 255.0)
    n = int(len(imgArr) / 2)

    # circuit
    qc = QuantumCircuit(n, n)
    for i in range(n):
        qc.ry(norm[2 * i], i)
        qc.p(norm[2 * i + 1], i)
    qc.measure_all()

    # qc.draw(output='mpl')
    # plt.show()
    return qc

# Amplitude Encoding
def amplitudeEncoding(imgArr):
    # data normalization
    norm = np.zeros(len(imgArr), dtype=np.complex_)
    sqSum = 0
    for item in imgArr:
        sqSum += item ** 2
    for i in range(len(imgArr)):
        norm[i] = (imgArr[i] / cmath.sqrt(sqSum))
    # circuit
    n = math.ceil(math.log(len(imgArr), 2))
    index = list(range(0,n,1))
    qc = QuantumCircuit(n, n)
    qc.initialize(norm, index)
    # print(norm)
    qc.measure_all()
    # qc.decompose().decompose().decompose().decompose().decompose().draw(output='mpl')
    # plt.show()
    return qc, sqSum, n

# QuAM Encoding
def quamEncoding(imgArr):
    # print(imgArr)
    # data normalization
    norm = []
    for data in imgArr:
        norm.append(to_bin(data, 8))
    statevector = np.zeros(2 ** len(norm[0]), dtype=np.complex_)
    # print(norm)
    # print(len(norm))
    data_dict = {}
    for i in imgArr:
        data_dict[str(i)] = 0
    for i in imgArr:
        data_dict[str(i)] += 1
    # print(data_dict)
    for item in data_dict:
        statevector[int(item)] = cmath.sqrt(data_dict[item] / len(norm))
        # print(statevector[int(item)])
    # print(statevector)
    # statevector1 = Statevector(statevector)
    # circuit
    n = len(norm[0])
    # print(statevector)
    index = list(range(0, n, 1))
    qc = QuantumCircuit(n, n)
    qc.initialize(statevector, index)
    qc.measure_all()
    # qc.draw(output='mpl')
    # plt.show()
    # print(norm)
    return qc, n

# Efficiet QRAM
def qramEncoding(imgArr):
    n = int(math.log(len(imgArr), 2))
    if n < 8:
        qc = QRAM_circuit(8, 8)
    else:
        qc = QRAM_circuit(n, 8)
    # qb_num = n+m
    return qc

# Angle QRAM - FRQI

# Improved QRAM

# Amplitude QRAM

# simulate
def simulate(qc, shots, backend):
    t_qc = transpile(qc, backend)
    sampler = Sampler(backend)
    sampler.options.default_shots = shots
    result = sampler.run([t_qc]).result()
    # print(result[0].data.keys())
    try:
        dist = result[0].data.meas.get_counts()
    except:
        dist = result[0].data.cl_reg.get_counts()
    # print(dist)
    # plot_distribution(dist)
    # plt.show()
    return dist

if __name__ == '__main__':
    imgArr = np.array([0, 125, 200, 255])
    # imgArr = np.array([200, 255])
    # imgArr = np.array([10])
    # print(basisEncoding(imgArr))
    qc = angleEncodingCircuit(imgArr)
    # qc, encoding = basisEncoding(imgArr)
    # qc = denseangleEncoding(imgArr)
    # qasm = constructBackend('qasm', 0)
    # qc, n = quamEncoding(imgArr)
    # qc = qramEncoding(imgArr)
    # print(qc.num_qubits)
    # qc = transpile(qc, qasm)
    # qc.draw()
    # qc.draw(output='mpl')
    # qc,sqSum, n = amplitudeEncoding(imgArr)
    shots = 1024
    param_meas = 0.1  # amplitude damping parameter
    # # # param_dep = 0.5 # depolarization parameter
    # # param_phase = 0.5 # phase damping parameter
    param_bf = 0.5
    # # # # get_simulator
    ideal_sim = constructBackend('qasm', 0, qc.num_qubits)
    # noise_sim = constructBackend('Depolarization', param_dep)
    noise_sim = constructBackend('Amplitude Damping', param_meas, qc.num_qubits)
    # noise_sim = constructBackend('Phase Damping', param_phase)
    # noise_sim = constructBackend('Bit Flip', param_bf, qc.num_qubits)
    t_qc = transpile(qc, noise_sim)
    qc.draw(output='mpl')
    t_qc.draw(output='mpl')
    dist_ideal = simulate(qc, shots, ideal_sim)
    dist_noise = simulate(t_qc, shots, noise_sim)
    #
    # # keyset = generateKeySet(n)
    # # img_ideal = ampDisReversion(dist_ideal, keyset, sqSum, shots, n)
    # # img_noise = ampDisReversion(dist_noise, keyset, sqSum, shots, n)
    # # plt.subplot(1,2,1)
    # # plt.imshow(img_ideal, cmap='gray')
    # # plt.title('Ideal Image')
    # # plt.subplot(1,2,2)
    # # plt.imshow(img_noise, cmap='gray')
    # # plt.title('Noise Embedded Image')
    # # print(dist_noise)
    #
    # print(dist_noise.keys())
    # print(dist_ideal.keys())
    plot_distribution(dist_ideal, title="Ideal Distribution")
    plot_distribution(dist_noise, title="Noise Distribution")
    #
    # # plotCompareDistribution(keyset, dist_ideal, dist_noise, ['Ideal', 'Noise Embedded'])
    plt.show()

