from shutil import posix

from networkx.generators import classic
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile, assemble
from qiskit.circuit.add_control import control
from qiskit.visualization.pulse_v2.layouts import qubit_index_sort
from qiskit_aer import Aer
from qiskit.visualization import plot_distribution
from qiskit.circuit.library import XGate, RYGate

import numpy as np
import math
from tqdm import tqdm

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image

# inverse QFT
def invQFT(qc, n):
    for qubit in range(n // 2):
        qc.swap(qubit, n-qubit-1)
    for j in range(n):
        for m in range(j):
            qc.cp(-math.pi/float(2**(j-m)), m, j)
        qc.h(j)

# BRQI Encoding
def BRQI(image):
    w_bits = int(np.ceil(math.log(image.shape[1],2)))
    h_bits = int(np.ceil(math.log(image.shape[0],2)))
    color_n_b = 8
    color_n_b = int(np.ceil(math.log(color_n_b,2)))
    color = QuantumRegister(1, 'color')
    y_ax = QuantumRegister(w_bits, 'y axis')
    x_ax = QuantumRegister(h_bits, 'x axis')
    bitplane_q = QuantumRegister(color_n_b, 'bitplanes')
    classic = ClassicalRegister(1+w_bits+h_bits+color_n_b, 'classic')
    qc = QuantumCircuit(color, y_ax, x_ax, bitplane_q, classic)

    qc.id(color)
    qc.h(x_ax)
    qc.h(y_ax)
    qc.h(bitplane_q)

    qc.barrier()

    for bitplane in range(8):
        bit_bitplane = "{0:b}".format(bitplane).zfill(color_n_b)
        for n, bit in enumerate(bit_bitplane):
            if bit != '1':
                qc.x(bitplane_q[n])
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                pixel = "{0:b}".format(image[i, j]).zfill(8)
                if pixel[bitplane] == '1':
                    i_bit = "{0:b}".format(i).zfill(h_bits)
                    for i_n, ibit in enumerate(i_bit):
                        if ibit != '1':
                            qc.x(x_ax[i_n])
                    j_bit = "{0:b}".format(j).zfill(w_bits)
                    for j_n, jbit in enumerate(j_bit):
                        if jbit != '1':
                            qc.x(y_ax[j_n])
                    qc.barrier()

                    controls = list(range(color.size, qc.num_qubits))

                    xg = XGate(None).control(len(controls))
                    controls.append(color)
                    qc.append(xg, controls)

                    for j_n, jbit in enumerate(j_bit):
                        if jbit != '1':
                            qc.x(y_ax[j_n])

                    for i_n, ibit in enumerate(i_bit):
                        if ibit != '1':
                            qc.x(x_ax[i_n])

                    qc.barrier()

        for n, bit in enumerate(bit_bitplane):
            if bit != '1':
                qc.x(bitplane_q[n])
    qc.barrier()
    qc.measure(list(reversed(range(qc.num_qubits))), list(range(classic.size)))
    return qc

# FRQI encoding -- Angle QROM
def FRQI(image):
    input_im = image.copy().flatten()
    thetas = np.interp(input_im, (0, 256), (0, np.pi/2))
    coord_q_num = int(np.ceil(math.log(len(input_im), 2)))
    O = QuantumRegister(coord_q_num, 'coordinates')
    c = QuantumRegister(1, 'c_req')
    cr = ClassicalRegister(O.size+c.size, 'cl_reg')

    qc = QuantumCircuit(c, O, cr)
    num_qubits = qc.num_qubits
    qc.id(c)
    qc.h(O)

    controls_ = []
    for i, _ in enumerate(O):
        controls_.extend([O[i]])

    for i, theta in enumerate(thetas):
        qubit_index_bin = "{0:b}".format(i).zfill(coord_q_num)

        for k, qub_ind in enumerate(qubit_index_bin):
            if int(qub_ind):
                qc.x(O[k])

        qc.mcry(theta=2*theta, q_controls=controls_, q_target=c[0])

        for k, qub_ind in enumerate(qubit_index_bin):
            if int(qub_ind):
                qc.x(O[k])

    qc.measure(list(reversed(range(qc.num_qubits))), list(range(cr.size)))

    return qc

# FTQR encoding
def FTQR(image):
    input_im = image.copy().flatten()
    theta = input_im.theta
    v = np.zeros(4)

    pass

# GQIR encoding

# MCRQI encoding -- RGB encoding
def MCRQI(image):
    xqubits = int(math.log(image.shape[0],2))
    yqubits = int(math.log(image.shape[1],2))
    qr = QuantumRegister(xqubits + yqubits + 3) # 3 for RGB qubits
    cr = ClassicalRegister(xqubits + yqubits + 3, 'c')
    qc = QuantumCircuit(qr, cr)

    for k in range(int(np.floor((xqubits + yqubits) / 2))):
        qc.swap(k, xqubits + yqubits -1 - k)

    for i in range(xqubits + yqubits):
        qc.h(i)

    for layer_num, input_im in enumerate(image.T):
        input_im = input_im.flatten()
        input_im = np.interp(input_im, (0, 255), (0, np.pi/2))

        for i, pixel in enumerate(input_im):
            arr = list(range(xqubits + yqubits))
            arr.append(int(xqubits + yqubits + layer_num))
            CMRY = RYGate(2 * pixel).control(xqubits + yqubits)

            to_not = "{0:b}".format(i).zfill(xqubits + yqubits)
            for j, bit in enumerate(to_not):
                if int(bit):
                    qc.x(j)

            qc.barrier()
            qc.append(CMRY, arr)

            if i != len(input_im) - 1 or layer_num != 2:
                for j, bit in enumerate(to_not):
                    if int(bit):
                        qc.x(j)
                qc.barrier()

    for k in range(int(np.floor((xqubits + yqubits) / 2))):
        qc.swap(k, xqubits + yqubits -1 - k)

    qc.swap(-1, -3)

    qc.barrier()

    for i in range(xqubits + yqubits + 3):
        qc.measure(i, i)

    return qc

# NEQR encoding -- QRAM and QROM
def NEQR(image):
    w_bits = int(np.ceil(math.log(image.shape[1],2)))
    h_bits = int(np.ceil(math.log(image.shape[0],2)))

    indx = QuantumRegister(w_bits + h_bits, 'indx')
    intensity = QuantumRegister(8, 'intensity')
    cr = ClassicalRegister(len(indx) + len(intensity), 'cr')
    qc = QuantumCircuit(intensity, indx, cr)
    num_qubits = qc.num_qubits
    input_im = image.copy().flatten()
    qc.id(intensity)
    qc.h(indx)
    for i, pixel in enumerate(input_im):
        pixel_bin = "{0:b}".format(pixel).zfill(len(intensity))
        position = "{0:b}".format(i).zfill(len(indx))
        for j, coord in enumerate(position):
            if int(coord):
                qc.x(qc.num_qubits-j-1)
        for idx, px_value in enumerate(pixel_bin[::-1]):
            if(px_value == '1'):
                control_qubits = list(range(intensity.size, intensity.size + indx.size))
                target_qubit = intensity[idx]
                qc.mcx(control_qubits, target_qubit)
        if i != len(input_im) - 1:
            for j, coord in enumerate(position):
                if int(coord):
                    qc.x(qc.num_qubits-j-1)
        qc.barrier()
    qc.measure(range(qc.num_qubits), range(cr.size))
    return qc

def imageOpen(imagePath, size, cmap):
    # read image and convert to gray scale
    img = Image.open(imagePath).convert(cmap)
    if size != 'NoResize':
        img = img.resize((size, size))
    arr = np.array(img)
    return arr

if __name__ == '__main__':
    img = imageOpen('img/duck.png', 2, cmap='L')
    # img = imageOpen('img/duck.png', 2, cmap='RGB')
    # print(img)
    # img = np.random.uniform(low=0, high=255, size=(2, 2)).astype(int)
    # qc = BRQI(img)
    # qc = FRQI(img)
    # qc = MCRQI(img)
    qc = NEQR(img)
    print(qc.depth())
    # qc.draw(output='mpl')
    # plt.show()