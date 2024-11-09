import time
from math import floor, sqrt, pi, log2
import numpy as np
from tangled_adjudicate.schrodinger.sparse_matrices import (create_pauli_matrices_for_full_size_hamiltonian,
                                                            load_schedule_data, create_sparse_hamiltonian,
                                                            compute_eigenvalues_and_eigenvectors)


def initialize_wavefunction(eigenvalues, eigenvectors, n_i, gap_initial):
    if eigenvalues[n_i+1] - eigenvalues[n_i] > gap_initial:
        # here the energy gap is bigger than gap_initial ... so we stay in the initial state
        psi = eigenvectors[:, n_i]
    else:
        # here the energy gap is smaller ... so we mix, but just with the state one higher
        psi = (eigenvectors[:, n_i] + eigenvectors[:, n_i+1])/sqrt(2)
        # question:: what about decay? what about mixing with higher states? this seems to only work if n_i = 0?

    psi = psi[:, np.newaxis]    # this makes psi explicitly a column vector; complex ndarray (2**n_qubits, 1)

    return psi


def evolve_wavefunction(eigenvalues, eigenvectors, psi, tf, s_step, number_of_levels):

    cn = np.squeeze(eigenvectors.conj().T.dot(psi))                         # complex ndarray (number_of_levels, 1)
    cn = np.multiply(cn, np.exp(-1j * eigenvalues * tf * s_step * 2 * pi))  # elementwise multiply => another vector
    psi = np.zeros((eigenvectors.shape[0], 1))
    for k in range(number_of_levels):
        row_vector = cn[k] * eigenvectors[:, k]
        column_vector = row_vector[:, np.newaxis]
        psi = psi + column_vector

    return psi     # complex ndarray (2**n_qubits, 1)


def calculate_correlation_matrix(sz, psi):

    n_qubits = int(log2(sz[0].shape[0]))
    magnetization_list = []
    correlation_matrix = np.zeros((n_qubits, n_qubits))

    for qubit_number in range(n_qubits):
        magnetization_list.append(np.real_if_close(psi.conj().T.dot(sz[qubit_number].dot(psi)))[0][0])

    for n in range(n_qubits - 1):
        for m in range(n + 1, n_qubits):
            correlation_matrix[n, m] = np.real_if_close(psi.conj().T.dot(sz[m].dot(sz[n].dot(psi))))[0][0] - \
                                       magnetization_list[m] * magnetization_list[n]

    return correlation_matrix


def evolve_schrodinger(h, jay, s_min, s_max, tf, n_qubits, verbose=False):

    # tf = real annealing time in nanoseconds
    # s_min = 0.1                           # Initial dimensionless anneal time (s=t/t_f), lower bound 0
    # s_max = 0.5                           # Final dimensionless anneal time (s=t/t_f), upper bound 1
    # Delta and Gamma (envelope functions from schedule) are in units of GHz
    # h and J are dimensionless and in this format:
    #         h = {0: -1, 1: -1}
    #         J = {(0, 1): -1}
    #         IMPORTANT: J[n, m] has m > n
    # n_qubits is the number of qubits

    # parameters, don't touch unless you are a quantum demigod
    n_i = 0                                 # Initial state (= 0 for ground state, 1 for 1st excited, etc.)
    max_step = 0.0005                       # maximum step size
    s_step = max_step                       # initial step size
    gap_min_adaptive = 1e-9                 # Minimum allowed gap for adaptive steps
    n_adaptive = 2                          # Number of levels to consider for adaptive steps
    gap_initial = 0.001                     # Threshold gap for initialization in superposition
    number_of_levels = 2 ** n_qubits        # max number, can reduce if you want

    # load and return (1001,) vectors for delta and big_e for typical D-Wave annealing schedule; if you want to try to
    # exactly match hardware, you will have to use specific data from whatever system you're simulating
    delta_qubit, big_e_qubit = load_schedule_data()

    # Identity and Pauli matrices -- sx, sy, sz are dicts with qubit # as key, should be sparse csr format
    sx, sy, sz = create_pauli_matrices_for_full_size_hamiltonian(n_qubits=n_qubits)

    s = s_min
    energies = []
    probabilities = []
    gap_old = 0
    psi = None
    eigenvectors = None
    correlation_matrix_list = []
    correlation_matrix = np.zeros((n_qubits, n_qubits))   # this is the correlation function we use for gameplay

    start = time.time()

    while s <= s_max:

        if verbose:
            print('computing for s = ', s)

        # eigenenergies and wavefunctions at s

        n_s = 1000 * s                      # checked that s=0.3 gives n_low = 300
        n_low = floor(n_s)                  # and big_e_qubit[n_low] = 0.05022258
        n_high = n_low + 1                  # and delta_qubit[n_low] = 3.208404585

        big_e_s = big_e_qubit[n_low] * (n_high - n_s) + big_e_qubit[n_high] * (n_s - n_low)
        delta = delta_qubit[n_low] * (n_high - n_s) + delta_qubit[n_high] * (n_s - n_low)

        big_h = create_sparse_hamiltonian(n_qubits, h, jay, big_e_s, delta, sx, sz)

        eigenvalues, eigenvectors = compute_eigenvalues_and_eigenvectors(big_h,
                                                                         number_of_levels=number_of_levels,
                                                                         n_qubits=n_qubits,
                                                                         use_eig=False)

        energies.append(eigenvalues)        # eigenvalues is a sorted ndarray of len (2**n_qubits,)

        # Initializing the wavefunction; psi is ndarray (2**n_qubits, 1)
        if s == s_min:
            psi = initialize_wavefunction(eigenvalues, eigenvectors, n_i, gap_initial)

        # evolving the wavefunction; psi is ndarray (2**n_qubits, 1)
        psi = evolve_wavefunction(eigenvalues, eigenvectors, psi, tf, s_step, number_of_levels)

        # calculate correlation function
        correlation_matrix = calculate_correlation_matrix(sz, psi)
        correlation_matrix_list.append(correlation_matrix)

        full_prob = np.squeeze(np.square(np.abs(eigenvectors.conj().T.dot(psi))))
        trunc_prob = [round(full_prob[k], 4) for k in range(len(full_prob))]

        # probability of being in eigenvector N; NOT probabilities of being in the sigma_z basis
        probabilities.append(trunc_prob)    # check P = [P abs(Vn'*psi).^2]
        gap = min(eigenvalues[1:n_adaptive]-eigenvalues[0: n_adaptive - 1])

        # Determining the integration step based on the gap size

        if abs(gap_old - gap) / (gap + 0.00001) > 0.05 and gap > gap_min_adaptive:
            s_step /= 2

        s += s_step
        if s_max-s_step < s < s_max+s_step:
            s = s_max
        gap_old = gap

    if verbose:
        print('This function call took', time.time() - start, 'seconds.')

    final_eigenvalues = energies[-1]      # check
    final_probabilities = probabilities[-1]
    final_eigenvectors = eigenvectors

    if verbose:
        print('final correlation matrix:')
        print(correlation_matrix)

        print('final_eigenvalues:')
        print(["{:.2f}".format(final_eigenvalues[k]) for k in range(len(final_eigenvalues))])

        print('final_eigenvectors:')
        for k in range(final_eigenvectors.shape[0]):
            print(["{:.2f}".format(final_eigenvectors[k][j]) for j in range(final_eigenvectors.shape[1])])

        print('final_probabilities:')
        print(["{:.2f}".format(final_probabilities[k]) for k in range(len(final_probabilities))])

    return correlation_matrix_list[-1]   # only return last element of the list
