from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import Aer, AerSimulator

from noise_model import *

# construct backend
def constructBackend(method, params, qb_nums):
    if (method == 'statevector'):
        # ideal statevector simulator
        return Aer.get_backend('statevector_simulator')
    elif (method == 'aer'):
        return AerSimulator()
    elif (method == 'qasm'):
        # noisy qasm simulator
        return Aer.get_backend('qasm_simulator')
    elif (method == 'Bit Flip'):
        noise_model = constructBitFlipNoiseModel(params)
        return AerSimulator(noise_model = noise_model)
    elif (method == 'Amplitude Damping'):
        noise_model = constructAmplitudeDampingNoiseModel(params, qb_nums, False)
        return AerSimulator(noise_model = noise_model)
    elif (method == 'Depolarization'):
        noise_model = constructDepolarizationNoiseModel(params)
        return AerSimulator(noise_model = noise_model)
    elif (method == 'Phase Damping'):
        noise_model = constructPhaseDampingNoiseModel(params)
        return AerSimulator(noise_model = noise_model)
    elif (method == 'Theymal'):
        pass
    else:
        print('error')