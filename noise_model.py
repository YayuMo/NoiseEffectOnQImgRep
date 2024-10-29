# Import from Qiskit Aer noise module
import random

from qiskit_aer.noise import amplitude_damping_error, NoiseModel, QuantumError, ReadoutError, pauli_error, depolarizing_error, thermal_relaxation_error, phase_damping_error, phase_amplitude_damping_error

def randomQubitSeriesGenerator(per, num):
    # generate_list
    ls = list(range(num))
    # randomly choose x percentage number in ls
    n = int(per * num)
    res = random.sample(ls, n)
    return res


def constructBitFlipNoiseModel(param_bf):
    # params
    # p_reset = param_bf
    # p_meas = param_bf
    p_gate1 = param_bf

    # QuantumError objects
    # error_reset = pauli_error([('X', p_reset), ('I', 1 - p_reset)])
    # error_meas = pauli_error([('X', p_meas), ('I', 1 - p_meas)])
    error_gate1 = pauli_error([('X', p_gate1), ('I', 1 - p_gate1)])
    error_gate2 = error_gate1.tensor(error_gate1)

    # Add errors to noise model
    noise_bit_flip = NoiseModel()
    # noise_bit_flip.add_all_qubit_quantum_error(error_reset, "reset")
    # noise_bit_flip.add_all_qubit_quantum_error(error_meas, "measure")
    noise_bit_flip.add_all_qubit_quantum_error(error_gate1, ["u1", "u2", "u3"])
    noise_bit_flip.add_all_qubit_quantum_error(error_gate2, ["cx"])

    return noise_bit_flip

# Depolarization Model on X-gates or qubit
def constructDepolarizationNoiseModel(param_dep):
    # construct error
    qerror_dep = depolarizing_error(param_dep, 1)

    # build noise model
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(qerror_dep, ['x'])

    return noise_model

# Amplitude Damping Model on Measurement
def constructAmplitudeDampingNoiseModel(param_meas, qb_nums):
    # construct error
    qerror_meas = amplitude_damping_error(param_meas)
    # build noise model
    noise_model = NoiseModel()
    # noise_model.add_all_qubit_quantum_error(qerror_meas, "measure")
    sampleList = randomQubitSeriesGenerator(0.25, qb_nums)
    for sample in sampleList:
        noise_model.add_quantum_error(qerror_meas, 'measure', [sample])
    return noise_model

# Phase Damping Model on phase gate
def constructPhaseDampingNoiseModel(param_phase):
    # construct error
    qerror_phase = phase_damping_error(param_phase)
    # build noise model
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(qerror_phase, ['u1', 'u2', 'u3'])
    return noise_model

if __name__ == '__main__':
    # constructPhaseDampingNoiseModel(0.05)
    randomQubitSeriesGenerator(0.5, 10)






