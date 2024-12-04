import math

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

if __name__ == '__main__':
    qc1 = QuantumCircuit(1, 1)
    qc1.ry(math.pi, 0)
    qc1.p(2 * math.pi, 0)
    state1 = Statevector.from_instruction(qc1)
    qc2 = QuantumCircuit(1, 1)
    qc2.ry(math.pi, 0)
    qc2.p(2 * math.pi, 0)
    qc2.ry(math.pi, 0).inverse()
    state2 = Statevector.from_instruction(qc2)
    print(state1)
    print(state2)