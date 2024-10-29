import qiskit as qk
import matplotlib
matplotlib.use('TkAgg')
from qiskit.circuit.library import MCMT, SwapGate, MCXGate,ZGate
from qiskit.quantum_info import Operator
# from qiskit.circuit.library import SwapGate
import matplotlib.pyplot as plt
import numpy as np

def buildMultiControlSwap():
    swapmat = [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ]
    return Operator(np.array(swapmat))

def QRAM_circuit(n, m):
    A = qk.QuantumRegister(n, "A")
    D = qk.QuantumRegister(m, "D")
    dq = qk.QuantumRegister(1, "dq")
    qy = qk.QuantumRegister(n + m, "qy")
    r = qk.QuantumRegister(1, "r")
    QRAM = qk.QuantumCircuit(A, D, dq, qy, r)

    # multi_ctrl_swap = MCMT(SwapGate, num_ctrl_qubits=2, num_target_qubits=2)
    # print(multi_ctrl_swap)
    multible_controled_NOT_gate3 = MCXGate(3)

    # for i in range(n):
    #     QRAM.initialize([1,0],A[i])
    #     QRAM.initialize([1,0],D[i])
    # for i in range(n+m):
    #     QRAM.initialize([1,0],qy[i])
    # QRAM.initialize([1,0],dq)
    # QRAM.initialize([1,0],r)
    QRAM.h(A)
    QRAM.cx(qy[:n:], A)
    QRAM.x(A)
    # QRAM.mct(A, dq)
    QRAM.mcx(A, dq)
    QRAM.x(r)
    for i in range(m):
        QRAM.cx(qy[i + n], D[i])
        QRAM.append(multible_controled_NOT_gate3, [1 + ((n + m) * 2), (n + m), n + i, m + i + (m + n + 1)])
        QRAM.cx(qy[i + n], D[i])

    QRAM.x(r)
    for i in range(m):
        QRAM.append(multible_controled_NOT_gate3, [1 + ((n + m) * 2), (n + m), n + i, m + i + (n + m + 1)])

    # QRAM.mct(A, dq)
    QRAM.mcx(A, dq)
    for i in range(n):
        QRAM.x(A[i])
        QRAM.cx(qy[i], A[i])

    # QRAM.measure_all()
    return QRAM

if __name__ == '__main__':

    circuit = QRAM_circuit(3, 3)
    circuit.draw(output="mpl")
    plt.show()
