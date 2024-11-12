from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile, assemble
from qiskit.quantum_info.operators import Operator
from qiskit.circuit.add_control import control
from qiskit.pulse import num_qubits
from qiskit.visualization.pulse_v2.layouts import qubit_index_sort
from qiskit_aer import Aer, AerSimulator
from qiskit.visualization import plot_distribution
from qiskit.circuit.library import XGate, RYGate

import numpy as np
import math

from tqdm import tqdm

import matplotlib

from circuits import simulate

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
    w_bits = int(np.ceil(math.log(image.shape[1], 2)))
    h_bits = int(np.ceil(math.log(image.shape[0], 2)))
    if not w_bits: w_bits = 1
    if not h_bits: h_bits = 1
    color_n_b = 8
    color_n_b = int(np.ceil(math.log(color_n_b, 2)))
    color = QuantumRegister(1, 'color')
    y_ax = QuantumRegister(w_bits, 'y axis')
    x_ax = QuantumRegister(h_bits, 'x axis')
    bitplane_q = QuantumRegister(color_n_b, 'bitplanes')
    classic = ClassicalRegister(1 + w_bits + h_bits + color_n_b, 'classic')
    qc = QuantumCircuit(color, y_ax, x_ax, bitplane_q, classic)

    # qc.id(color)
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

                    controls = list(range(color.size,
                                          qc.num_qubits))
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

# BRQI Rev
def Rev_BRQI(image, counts):
    out_pixels = []
    w_bits = int(np.ceil(math.log(image.shape[1], 2)))
    h_bits = int(np.ceil(math.log(image.shape[0], 2)))
    color = QuantumRegister(1, 'color')
    y_ax = QuantumRegister(w_bits, 'y axis')
    x_ax = QuantumRegister(h_bits, 'x axis')
    for item in counts:
        out_pixels.append((int(item[0:color.size], 2),
                           int(item[color.size:color.size + x_ax.size], 2),
                           int(item[color.size + x_ax.size:color.size + x_ax.size + y_ax.size], 2),
                           int(item[color.size + x_ax.size + y_ax.size:qc.num_qubits], 2)
                           ))
    out_image = []
    for k in range(image.shape[0]):
        for j in range(image.shape[1]):
            bits = [i for i in range(len(out_pixels)) if out_pixels[i][1] == k and out_pixels[i][2] == j]
            pixel = np.zeros((8,))
            for bit in bits:
                pixel[out_pixels[bit][3]] = int(out_pixels[bit][0])
            mystring = "".join([str(int(a)) for a in pixel])
            out_image.append(int(mystring, 2))
    out_img = np.array(out_image).reshape(image.shape[1], image.shape[0])

    im_pil = Image.fromarray(out_img.astype(np.uint8))
    img_rotated = im_pil.rotate(-90)
    img_transpose = img_rotated.transpose(Image.FLIP_LEFT_RIGHT)
    out_image = np.array(img_transpose)
    return out_image

# FRQI encoding -- General Angle QROM
def FRQI(image):
    input_im = image.copy().flatten()
    # print(input_im)
    thetas = np.interp(input_im, (0, 256), (0, np.pi/2))
    coord_q_num = int(np.ceil(math.log(len(input_im), 2)))
    O = QuantumRegister(coord_q_num, 'coordinates')
    c = QuantumRegister(1, 'c_reg')
    cr = ClassicalRegister(O.size + c.size, "cl_reg")

    qc_image = QuantumCircuit(c, O, cr)
    # qc_image.id(c)
    qc_image.h(O)

    controls_ = []
    for i, _ in enumerate(O):
        controls_.extend([O[i]])

    for i, theta in enumerate(thetas):
        qubit_index_bin = "{0:b}".format(i).zfill(coord_q_num)

        for k, qub_ind in enumerate(qubit_index_bin):
            if int(qub_ind):
                qc_image.x(O[k])

        # qc_image.barrier()
        # for coord_or_intns in (0,1):
        qc_image.mcry(theta=2 * theta,
                      q_controls=controls_,
                      q_target=c[0])

        # qc_image.barrier()
        for k, qub_ind in enumerate(qubit_index_bin):
            if int(qub_ind):
                qc_image.x(O[k])

        # qc_image.barrier()

    qc_image.measure(list(reversed(range(qc_image.num_qubits))), list(range(cr.size)))

    return qc_image, qc_image.num_qubits

# FRQI Rev
def Rev_FRQI(image, counts):
    input_im = image.copy().flatten()
    classical_colors = []
    for i in range(0, len(input_im)):
        color_list = []
        for item in counts.items():
            key = item[0]
            amount = item[1]
            bin_coord = key[1:]
            int_coord = int(bin_coord, 2)
            if int_coord == i:
                color_list.append((key[0], amount))
        color_amount = 0
        for color, amount in color_list:
            if not int(color):
                color_amount = color_amount + amount
        try:
            color = np.arccos((color_amount / sum(n for _, n in color_list)) ** (1 / 2))
            classical_colors.append(color)
        except ZeroDivisionError:
            print("ZeroDivisionError")
    classical_colors = list(reversed(np.interp(classical_colors, (0, np.pi / 2), (0, 256)).astype(int)))
    imgArr = np.array(classical_colors).reshape(image.shape)
    # print(classical_colors, '\n', input_im)
    return Image.fromarray(imgArr)

# FTQR encoding
def FTQR(image):
    input_im = image.copy().flatten()
    thetas = input_im.copy()
    coord_q_num = int(np.ceil(math.log(len(input_im), 2)))
    v = thetas.copy()
    ph = np.e ** (2 * np.pi * v * 1j / 1024)
    S = np.diag(ph)
    SOp = Operator(S)

    c = QuantumRegister(coord_q_num, 'coordinates')
    cl = ClassicalRegister(coord_q_num, 'cl_reg')
    qc = QuantumCircuit(c, cl)
    qc.h(c)
    qc.append(SOp, c)

    return qc

# GQIR encoding
def GQIR(image):
    x= int(np.ceil(math.log(image.shape[0], 2)))
    y = int(np.ceil(math.log(image.shape[1], 2)))
    q = 8

    color = QuantumRegister(q, 'color')
    y_ax = QuantumRegister(y, 'y axis')
    x_ax = QuantumRegister(x, 'x axis')
    classic = ClassicalRegister(x + y + q, 'classic')
    qc = QuantumCircuit(color, y_ax, x_ax, classic)

    # qc.id(color)
    qc.h(x_ax)
    qc.h(y_ax)
    qc.barrier()
    controls_ = []
    for i, _ in enumerate(x_ax):
        controls_.extend([x_ax[i]])
    for i, _ in enumerate(y_ax):
        controls_.extend([y_ax[i]])
    for xi in range(image.shape[0]):
        xi_bin = "{0:b}".format(xi).zfill(x_ax.size)
        for i, bit in enumerate(xi_bin):
            if not int(bit):
                qc.x(x_ax[i])
        qc.barrier()
        for yi in range(image.shape[1]):
            yi_bin = "{0:b}".format(yi).zfill(y_ax.size)
            for i, bit in enumerate(yi_bin):
                if not int(bit):
                    qc.x(y_ax[i])
            qc.barrier()
            intensity_bin = "{0:b}".format(image[xi, yi]).zfill(len(color))
            xg = XGate(None).control(len(controls_))
            target = []
            for i, bit in enumerate(intensity_bin):
                if int(bit):
                    qc.mcx(controls_, color[i])
            qc.barrier()
            for i, bit in enumerate(yi_bin):
                if not int(bit):
                    qc.x(y_ax[i])
            qc.barrier()
        for i, bit in enumerate(xi_bin):
            if not int(bit):
                qc.x(x_ax[i])
        qc.barrier()

    qc.measure(x_ax, range(x_ax.size))
    qc.measure(y_ax, range(x_ax.size, x_ax.size + y_ax.size))
    qc.measure(color, range(x_ax.size + y_ax.size, x_ax.size + y_ax.size + color.size))
    return qc

# GQIR Rev
def Rev_GQIR(image, counts):
    pass

# MCRQI encoding -- RGB encoding
def MCRQI(image):
    xqbit = int(math.log(image.shape[0],2))
    yqbit = int(math.log(image.shape[1],2))
    qr = QuantumRegister(xqbit + yqbit + 3)  # 3 stands for RGB qubits
    # color = qr[xqbit + yqbit:]
    # coordinate = qr[:xqbit + yqbit]
    cr = ClassicalRegister(xqbit + yqbit + 3, 'c')
    qc = QuantumCircuit(qr, cr)

    for k in range(int(np.floor((xqbit + yqbit) / 2))):
        qc.swap(k, xqbit + yqbit - 1 - k)

    for i in range(xqbit + yqbit):
        qc.h(i)

    for layer_num, input_im in enumerate(image.T):
        input_im = input_im.flatten()
        input_im = np.interp(input_im, (0, 255), (0, np.pi / 2))

        for i, pixel in enumerate(input_im):
            arr = list(range(xqbit + yqbit))
            arr.append(int(xqbit + yqbit + layer_num))
            cMry = RYGate(2 * pixel).control(xqbit + yqbit)

            to_not = "{0:b}".format(i).zfill(xqbit + yqbit)
            for j, bit in enumerate(to_not):
                if int(bit):
                    qc.x(j)
            qc.barrier()
            qc.append(cMry, arr)

            if i != len(input_im) - 1 or layer_num != 2:
                for j, bit in enumerate(to_not):
                    if int(bit):
                        qc.x(j)
                qc.barrier()

    for k in range(int(np.floor((xqbit + yqbit) / 2))):
        qc.swap(k, xqbit + yqbit - 1 - k)

    qc.swap(-1, -3)

    qc.barrier()
    for i in range(xqbit + yqbit + 3):
        qc.measure(i, i)

    return qc, qc.num_qubits

# MCRQI Rev
def Rev_MCRQI(image, counts, to_print=True):
    output_ims = []
    for layer_num, input_im in enumerate(image.T):
        input_im = input_im.flatten()
        nums = []
        for iter in range(len(input_im)):
            flag = 0
            num = []
            for item in counts.items():
                if int(item[0][3:], 2) == iter:
                    num.append((int(item[0][layer_num], 2), item[1]))
            nums.append(num)
        for l, num in enumerate(nums):
            my_set = {x[0] for x in num}
            nums[l] = [(i, sum(x[1] for x in num if x[0] == i)) for i in my_set]
        colors = []
        # for index in range(len(nums)):
        #     if nums[index] == []:
        #         nums[index] = [(0, 0)]
        for num in nums:
            if len(num) == 2:
                if num[0][0] == 0:
                    color = np.arccos((num[0][1] / (num[0][1] + num[1][1])) ** (1 / 2))
                    colors.append(color)
                else:
                    color = np.arccos((num[1][1] / (num[0][1] + num[1][1])) ** (1 / 2))
                    colors.append(color)
            else:
                if num[0][0] == 0:
                    colors.append(0)
                else:
                    colors.append(np.pi / 2)
        output_im = np.interp(colors, (0, np.pi / 2), (0, 255)).astype(int)
        if to_print:
            print(output_im, '\n', (image.T)[layer_num].copy().flatten())
        output_ims.append(output_im.reshape(image[:, :, 0].shape))

    return np.array(output_ims).T

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
    # qc.id(intensity)
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

# NEQR Rev
def Rev_NEQR(image, counts):
    w_bits = int(np.ceil(math.log(image.shape[1],2)))
    h_bits = int(np.ceil(math.log(image.shape[0],2)))
    input_im = image.copy().flatten()
    out_pixels = []
    for item in counts:
        out_pixels.append((int(item[0:w_bits + h_bits], 2), int(item[w_bits + h_bits:], 2)))
    out_image = np.zeros((1, len(input_im)))
    for pixel in out_pixels:
        out_image[0][pixel[0]] = pixel[1]
    out_image = np.reshape(out_image, (image.shape))
    return out_image

# OQIM encoding
def OQIM(image):
    im_list = image.flatten()
    ind_list = sorted(range(len(im_list)), key=lambda i: im_list[i])
    max_index = max(ind_list)
    # in angle: theta = intensity, phi = coordinate
    thetas = np.interp(im_list, (0, 256), (0, np.pi/2))
    phis = np.interp(range(len(im_list)), (0, len(im_list)), (0, np.pi/2))

    num_ind_bits = int(np.ceil(math.log(len(im_list),2)))
    if not num_ind_bits:
        num_ind_bits = 1

    O = QuantumRegister(num_ind_bits, 'o_reg')
    c = QuantumRegister(1, 'c_reg')
    p = QuantumRegister(1, 'p_reg')
    cr = ClassicalRegister(O.size + c.size + p.size, 'cl_reg')

    qc = QuantumCircuit(c, p, O, cr)
    num_qubits = qc.num_qubits
    input_im = image.copy().flatten()
    # qc.id(c)
    qc.h(O)
    qc.h(p)
    controls_ = []
    for i, _ in enumerate(O):
        controls_.extend([O[i]])
    for j, _ in enumerate(p):
        controls_.extend([p[j]])

    for i, (phi, theta) in enumerate(zip(phis, thetas)):
        qubit_index_bin = "{0:b}".format(i).zfill(num_ind_bits)

        for k, qub_ind in enumerate(qubit_index_bin):
            if int(qub_ind):
                qc.x(O[k])

        for coord_or_intns in (0, 1):
            if not coord_or_intns:
                qc.mcry(theta=2 * theta,
                        q_controls=controls_,
                        q_target=c[0])

            else:
                qc.x(p)
                qc.mcry(theta=2 * phi,
                        q_controls=controls_,
                        q_target=c[0])
                if i != len(thetas) - 1:
                    qc.x(p)

        if i != len(thetas) - 1:
            for k, qub_ind in enumerate(qubit_index_bin):
                if int(qub_ind):
                    qc.x(O[k])

    qc.barrier()
    qc.measure(list(reversed(range(qc.num_qubits))), list(range(cr.size)))

    return qc

# OQIM Rev
def Rev_OQIM(image, counts):
    im_list = image.flatten()
    ind_list = sorted(range(len(im_list)), key=lambda k: im_list[k])
    max_index = max(ind_list)
    classical_colors = []
    classical_coords = []
    for i in range(0, max_index + 1):
        color_list = []
        coord_list = []
        for item in counts.items():
            key = item[0]
            amount = item[1]
            bin_coord = key[2:]
            int_coord = int(bin_coord, 2)
            if int_coord == i:
                if not int(key[1]):
                    color_list.append((key[0], amount))
                else:
                    coord_list.append((key[0], amount))
        color_amount = 0
        for color, amount in color_list:
            if not int(color):
                color_amount = color_amount + amount
        try:
            color = np.arccos((color_amount / sum(n for _, n in color_list)) ** (1 / 2))
            classical_colors.append(color)
        except ZeroDivisionError:
            print("ZeroDivisionError")

        coord_amount = 0
        for coord, amount in coord_list:
            if not int(coord):
                coord_amount = coord_amount + amount
        try:
            coord = np.arccos((coord_amount / sum(n for _, n in coord_list)) ** (1 / 2))
            classical_coords.append(coord)
        except ZeroDivisionError:
            print("ZeroDivisionError")
    classical_colors = np.interp(classical_colors, (0, np.pi / 2), (0, 256)).astype(int)
    output_im = classical_colors.reshape(image.shape)

    return output_im

# QSMC_QSNC encoding
def QSMC(image):
    im_list = image.flatten()
    ind_list = sorted(range(len(im_list)), key=lambda k: im_list[k])
    max_index = max(ind_list)
    num_ind_bits = int(np.ceil(math.log(len(im_list), 2)))
    # now in angles: theta = intensity, phi = coordinate
    thetas = np.interp(im_list, (0, 256), (0, np.pi / 2))
    phis = np.interp(range(len(im_list)), (0, len(im_list)), (0, np.pi / 2))
    O = QuantumRegister(num_ind_bits, 'o_reg')
    color = QuantumRegister(1, 'color')
    coordinate = QuantumRegister(1, 'coordinate')
    cr = ClassicalRegister(O.size + color.size + coordinate.size, "cl_reg")

    qc_image = QuantumCircuit(color, coordinate, O, cr)
    # qc_image.id(color)
    # qc_image.id(coordinate)
    qc_image.h(O)
    controls_ = []
    for i, _ in enumerate(O):
        controls_.extend([O[i]])

    for i, (phi, theta) in enumerate(zip(phis, thetas)):
        qubit_index_bin = "{0:b}".format(i).zfill(num_ind_bits)

        for k, qub_ind in enumerate(qubit_index_bin):
            if int(qub_ind):
                qc_image.x(O[k])

        qc_image.barrier()

        for coord_or_intns in (0, 1):
            if not coord_or_intns:
                qc_image.mcry(theta=2 * theta,
                              q_controls=controls_,
                              q_target=color[0])
            else:
                qc_image.mcry(theta=2 * phi,
                              q_controls=controls_,
                              q_target=coordinate[0])

        qc_image.barrier()

        if i != len(thetas) - 1:
            for k, qub_ind in enumerate(qubit_index_bin):
                if int(qub_ind):
                    qc_image.x(O[k])

        qc_image.barrier()

    qc_image.measure(list(reversed(range(qc_image.num_qubits))), list(range(cr.size)))

    return qc_image

# QSMC Rev
def Rev_QSMC(image, counts):
    im_list = image.flatten()
    ind_list = sorted(range(len(im_list)), key=lambda i: im_list[i])
    max_index = max(ind_list)
    classical_colors = []
    classical_coords = []
    for i in range(0, max_index + 1):
        color_list = []
        coord_list = []
        for item in counts.items():
            key = item[0]
            amount = item[1]
            bin_coord = key[2:]
            int_coord = int(bin_coord, 2)
            if int_coord == i:
                color_list.append((key[0], amount))
                coord_list.append((key[1], amount))
        color_amount = 0
        for color, amount in color_list:
            if not int(color):
                color_amount = color_amount + amount
        try:
            color = np.arccos((color_amount / sum(n for _, n in color_list)) ** (1 / 2))
            classical_colors.append(color)
        except ZeroDivisionError:
            print("ZeroDivisionError")

        coord_amount = 0
        for coord, amount in coord_list:
            if not int(coord):
                coord_amount = coord_amount + amount
        try:
            coord = np.arccos((coord_amount / sum(n for _, n in coord_list)) ** (1 / 2))
            classical_coords.append(coord)
        except ZeroDivisionError:
            # coord = 0
            # classical_coords.append(coord)
            print("ZeroDivisionError")
    classical_colors = np.interp(classical_colors, (0, np.pi / 2), (0, 256)).astype(int)
    print('the meauserd colors are \n {} \n the input colors are \n {}'.format(classical_colors, im_list))
    classical_coords = np.interp(classical_coords, (0, np.pi / 2), (0, len(im_list))).astype(int)
    print('the meauserd coordinates are \n {}'.format(classical_coords))
    output_im = classical_colors.reshape(image.shape)
    return output_im

# real-image open
def imageOpen(imagePath, size, cmap):
    # read image and convert to gray scale
    img = Image.open(imagePath).convert(cmap)
    # img = Image.open(imagePath)
    if size != 'NoResize':
        img = img.resize((size, size))
    arr = np.array(img)
    return arr

# random image generate
def imageGenerate(size, prop):
    image = np.zeros((size, size, 3))
    image0List = np.random.randint(size * size, size = prop)
    image1List = np.random.randint(size * size, size = prop)
    image2List = np.random.randint(size * size, size = prop)
    image0 = np.full(size * size, 255)
    image1 = np.full(size * size, 255)
    image2 = np.full(size * size, 255)

    for pixel0 in image0List:
        image0[pixel0] = 0
    for pixel1 in image1List:
        image1[pixel1] = 0
    for pixel2 in image2List:
        image2[pixel2] = 0

    image[:, :, 0] = image0.reshape(size, size)
    image[:, :, 1] = image1.reshape(size, size)
    image[:, :, 2] = image2.reshape(size, size)

    return image

# simulate model 2
def simulate2(qc, shots, backend):
    t_qc = transpile(qc, backend)
    qobj = assemble(t_qc, shots = shots)
    result = backend.run(qobj).result()
    counts = result.get_counts(qc)
    return counts

if __name__ == '__main__':
    img = imageOpen('img/duck.png', 16, cmap='L')
    # img = np.random.uniform(low=0, high=255, size=(16, 16)).astype(int)
    # img = imageOpen('img/duck.png', 2, cmap='RGB')
    # img = imageGenerate(8, 10)
    # print(img)
    # img = np.random.uniform(low=0, high=255, size=(2, 2)).astype(int)
    # qc = BRQI(img)
    qc,n = FRQI(img)
    # qc = FTQR(img)
    # qc = GQIR(img)
    # qc, n = MCRQI(img)
    # qc = NEQR(img)
    # qc = OQIM(img)
    # qc = QSMC(img)
    print(qc.depth())
    # qc.draw(output='mpl')
    shots = 20000
    # print(n)
    aer_sim = Aer.get_backend('qasm_simulator')
    t_qc = transpile(qc, aer_sim)
    qobj = assemble(t_qc, shots = shots)
    # result = aer_sim.run(qobj).result()
    # counts = result.get_counts(qc)
    # print(result)
    dist = simulate(t_qc, shots, aer_sim)
    # print(len(dist))

    # img_rev = Rev_BRQI(img, counts)
    # img_rev = Rev_MCRQI(img, counts, to_print=False)
    img_rev = Rev_FRQI(img, dist)
    # img_rev = Rev_NEQR(img, counts)
    # img_rev = Rev_OQIM(img, counts)
    # img_rev = Rev_QSMC(img, counts)

    plt.subplot(1,2,1)
    plt.title('Original Image')
    plt.imshow(img, cmap='gray')
    plt.subplot(1,2,2)
    plt.title('Quantized Image')
    plt.imshow(img_rev, cmap='gray')
    plt.show()