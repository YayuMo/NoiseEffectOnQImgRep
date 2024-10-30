from qiskit import transpile
from qiskit_aer import Aer
from qiskit_ibm_runtime import QiskitRuntimeService

from circuits import *
from imageUtil import image2Arr
from tqdm import tqdm

import matplotlib.pyplot as plt

def setQiskitBackend(token, backendName):
    service = QiskitRuntimeService(channel='ibm_quantum', token=token)
    return service.backend(backendName)

def circuitEvaluation(imgPath, imgSizeList, encoding, backend):
    qc_depths = []
    qc_nums = []
    qc_vol = []
    if encoding == 'Amplitude Encoding':
        for size in tqdm(imgSizeList):
            vec = image2Arr(imgPath, size)
            qc, sqSum, n = amplitudeEncoding(vec)
            qc_transpiled = transpile(qc, backend, optimization_level=3)
            qc_depths.append(qc_transpiled.depth())
            qc_nums.append(n)

    elif encoding == 'Basis Encoding':
        for size in tqdm(imgSizeList):
            vec = image2Arr(imgPath, size)
            qc,encoding = basisEncoding(vec)
            # qc.draw(output='mpl')
            # plt.show()
            # qc_transpiled = transpile(qc, backend, optimization_level=3)
            # qc_depths.append(qc_transpiled.depth())
            qc_depths.append(qc.depth())
            qc_nums.append(len(encoding))

    elif encoding == 'Angle Encoding':
        for size in tqdm(imgSizeList):
            vec = image2Arr(imgPath, size)
            qc = angleEncodingCircuit(vec)
            # qc_transpiled = transpile(qc, backend, optimization_level=3)
            # qc_depths.append(qc_transpiled.depth())
            qc_depths.append(qc.depth())
            qc_nums.append(len(vec))

    elif encoding == 'Dense Angle Encoding':
        for size in tqdm(imgSizeList):
            vec = image2Arr(imgPath, size)
            qc = denseangleEncoding(vec)
            # qc_transpiled = transpile(qc, backend, optimization_level=3)
            # qc_depths.append(qc_transpiled.depth())
            qc_depths.append(qc.depth())
            qc_nums.append(int(len(vec) /  2))

    elif encoding == 'QUAM Encoding':
        for size in tqdm(imgSizeList):
            vec = image2Arr(imgPath, size)
            qc, n = quamEncoding(vec)
            qc_transpiled = transpile(qc, backend, optimization_level=3)
            qc_depths.append(qc_transpiled.depth())
            qc_nums.append(n)

    elif encoding == 'QRAM Encoding':
        for size in tqdm(imgSizeList):
            vec = image2Arr(imgPath, size)
            qc = qramEncoding(vec)
            qc_depths.append(qc.depth())
            qc_nums.append(qc.num_qubits)


    # plt.subplot(1,2,1)
    # plt.plot(imgSizeList, qc_depths, color = 'red', linestyle='-', label='QC Depth', marker='*')
    # plt.xlabel('Image side size')
    # plt.ylabel('circuit depth')
    # plt.title('Circuit Depth')
    #
    # plt.subplot(1,2,2)
    # plt.plot(imgSizeList, qc_nums, color = 'blue', linestyle='-', label='QC Num', marker='o')
    # plt.xlabel('Image side size')
    # plt.ylabel('number of utilized qubits')
    # plt.title('Qubits number')
    # plt.show()
    return qc_depths, qc_nums

if __name__ == '__main__':
    imgPath = 'img/duck.png'
    # imgPath = 'img/highres.jpg'
    imgSizeList = [2, 4, 8, 16, 32, 64, 128, 256]
    # imgSizeList = [2, 4, 8]
    token = '587ff4182be1ef7257ea53cdc055151efbeec1cafb5e2c7165f1ede2976f9045092d2fa19c20c17a54a389ef2699d485bddd7688a490c4de0d5b4cbf46456e56'
    hardware = setQiskitBackend(token, 'ibm_brisbane')
    sim = Aer.get_backend('qasm_simulator')
    amp_depths, amp_nums = circuitEvaluation(imgPath, imgSizeList, encoding='Amplitude Encoding', backend=hardware)
    dense_depths, dense_nums = circuitEvaluation(imgPath, imgSizeList, encoding='Dense Angle Encoding', backend=hardware)
    quam_depths, quam_nums = circuitEvaluation(imgPath, imgSizeList, encoding='QUAM Encoding', backend=hardware)
    basis_depths, basis_nums = circuitEvaluation(imgPath, imgSizeList, encoding='Basis Encoding', backend=sim)
    angle_depths, angle_nums = circuitEvaluation(imgPath, imgSizeList, encoding='Angle Encoding', backend=sim)
    qram_depths, qram_nums = circuitEvaluation(imgPath, imgSizeList, encoding='QRAM Encoding', backend=sim)

    plt.subplot(1,2,1)
    plt.plot(imgSizeList, basis_depths, color = 'red', linestyle='-', label='Basis Encoding', marker='*')
    plt.plot(imgSizeList, angle_depths, color = 'blue', linestyle='-', label='Angle Encoding', marker='x')
    plt.plot(imgSizeList, dense_depths, color = 'purple', linestyle='-', label='Dense Angle Encoding', marker='D')
    plt.plot(imgSizeList, amp_depths, color = 'yellow', linestyle='-', label='Amplitude Encoding', marker='^')
    plt.plot(imgSizeList, quam_depths, color = 'green', linestyle='-', label='QUAM Encoding', marker='o')
    plt.plot(imgSizeList, qram_depths, color = 'black', linestyle='-', label='QRAM Encoding', marker='+')
    plt.legend(loc='best')
    plt.xlabel('Image side size')
    plt.ylabel('circuit depth')
    plt.yscale('log')
    plt.title('Circuit Depth')

    plt.subplot(1,2,2)
    plt.plot(imgSizeList, basis_nums, color='red', linestyle='-', label='Basis Encoding', marker='*')
    plt.plot(imgSizeList, angle_nums, color='blue', linestyle='-', label='Angle Encoding', marker='x')
    plt.plot(imgSizeList, dense_nums, color='purple', linestyle='-', label='Dense Angle Encoding', marker='D')
    plt.plot(imgSizeList, amp_nums, color='yellow', linestyle='-', label='Amplitude Encoding', marker='^')
    plt.plot(imgSizeList, quam_nums, color='green', linestyle='-', label='QUAM Encoding', marker='o')
    plt.plot(imgSizeList, qram_nums, color = 'black', linestyle='-', label='QRAM Encoding', marker='+')
    plt.legend(loc='best')
    plt.xlabel('Image side size')
    plt.ylabel('number of utilized qubits')
    plt.yscale('log')
    plt.title('Qubits Number')
    plt.show()
