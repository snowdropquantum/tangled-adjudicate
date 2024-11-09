""" this is to test sparse matrix operations required for schrodinger """
import os
import numpy as np
from scipy import sparse
from scipy.linalg import eigh, eig
from scipy.sparse.linalg import eigsh


def create_2d_pauli_matrices(verbose=False):
    identity_matrix = sparse.csr_matrix([[1, 0], [0, 1]])
    pauli_z = sparse.csr_matrix([[1, 0], [0, -1]])
    pauli_x = sparse.csr_matrix([[0, 1], [1, 0]])
    pauli_y = sparse.csr_matrix([[0, -1j], [1j, 0]])

    if verbose:
        print('identity_matrix:')
        print(identity_matrix.toarray())
        print('pauli_z:')
        print(pauli_z.toarray())
        print('pauli_x:')
        print(pauli_x.toarray())
        print('pauli_y:')
        print(pauli_y.toarray())

    return identity_matrix, pauli_z, pauli_x, pauli_y


def create_pauli_matrices_for_full_size_hamiltonian(n_qubits, verbose=False):

    # create and load 2x2 sparse pauli matrices
    identity_matrix, pauli_z, pauli_x, pauli_y = create_2d_pauli_matrices()

    # I think it's fine to have these be dicts, where the qubit index is the key
    sz = {}
    sx = {}
    sy = {}

    for m in range(n_qubits):
        ax = 1
        az = 1
        ay = 1
        for n in range(n_qubits):
            if n == m:
                bx = pauli_x
                bz = pauli_z
                by = pauli_y
            else:
                bx = identity_matrix
                bz = identity_matrix
                by = identity_matrix

            ax = sparse.kron(ax, bx, format='csr')
            az = sparse.kron(az, bz, format='csr')
            ay = sparse.kron(ay, by, format='csr')

        sx[m] = ax
        sz[m] = az
        sy[m] = ay

        # note this looks correct for n_qubits = 1 and 2 (and I think 3)
        # but would be good to have a more systematic check
        if verbose:
            print('pauli_z for qubit', m, ':')
            print(sz[m])
            print(sz[m].toarray())
            print('pauli_x for qubit', m, ':')
            print(sx[m])
            print(sx[m].toarray())
            print('pauli_y for qubit', m, ':')
            print(sy[m])
            print(sy[m].toarray())

    return sx, sy, sz


def load_schedule_data(file_path=None, verbose=False):
    # data is a numpy array
    if file_path is None:
        file_path = os.path.join(os.getcwd(), 'schrodinger', 'new_schedule.txt')
    data = np.loadtxt(file_path)       # Import SR8 qubit information

    # these are both 1001 dimensional row vectors
    delta_qubit = data[:, 1] / 2
    big_e_qubit = data[:, 2]

    if verbose:
        print('delta_qubit:', delta_qubit.shape)
        print(delta_qubit)
        print('big_e_qubit:', big_e_qubit.shape)
        print(big_e_qubit)

    return delta_qubit, big_e_qubit


def create_sparse_hamiltonian(n_qubits, h, jay, big_e_s, delta, sx, sz):

    # define hamiltonian -- sparse csr format
    big_h = sparse.csr_matrix((2 ** n_qubits, 2 ** n_qubits))

    for n in range(n_qubits):
        if n in h:
            big_h = big_h - delta * sx[n] + big_e_s * h[n] * sz[n]
        for m in range(n + 1, n_qubits):  # this loops over all m > n
            if (n, m) in jay:
                big_h = big_h + big_e_s * jay[n, m] * sz[n].multiply(sz[m])    # checked for 2 qubits

    return big_h


def process_eigenresults(e_vals, e_vects):
    eigenvalues_unsorted = np.real(e_vals)  # ensures no complex tomfoolery
    sort_index = np.argsort(eigenvalues_unsorted)  # returns the indices of the sort
    eigenvalues = np.sort(eigenvalues_unsorted)  # actually does the sort
    eigenvectors = e_vects[:, sort_index]  # re-orders the columns to match the sort

    return eigenvalues, eigenvectors


def compute_eigenvalues_and_eigenvectors(big_h, number_of_levels, n_qubits, use_eig=False, verbose=False):

    # If use_eig is True, eig is used in all cases.
    # If use_eig is False and number_of_levels == 2**n_qubits, then eigh is used.
    # If use_eig is False and number_of_levels < 2**n_qubits, then eigsh is used.

    if use_eig:
        e_vals, e_vects = eig(big_h.toarray())
        eigenvalues, eigenvectors = process_eigenresults(e_vals, e_vects)
    else:
        if number_of_levels == 2**n_qubits:
            # returns all eigenvalues in ascending order
            # The column eigenvectors[:, i] is the normalized eigenvector corresponding to the eigenvalue evalues[i]
            try:
                #############################################
                evalues, eigenvectors = eigh(big_h.toarray())   # this is taking up most of the time!!!
                #############################################
            except:
                evalues, eigenvectors = eigh(big_h.toarray() + 0.00001 * np.eye(int(number_of_levels)))

            eigenvalues = np.real(evalues)  # ensures no complex tomfoolery
        else:
            # in this case, returns the number_of_levels smallest algebraic eigenvalues; Hamiltonians only have real
            # eigenvalues so this should work, although we'll explicitly make sure they are real
            # The column evectors[:, i] is the normalized eigenvector corresponding to the eigenvalue evalues[i]
            # in this case I don't think the eigenvalues are necessarily sorted already, so we have to do that
            e_vals, e_vects = eigsh(big_h, number_of_levels, which='SA', maxiter=10000, tol=1e-9)
            eigenvalues, eigenvectors = process_eigenresults(e_vals, e_vects)

    if verbose:
        print('final_eigenvalues:')
        print(["{:.2f}".format(eigenvalues[k]) for k in range(len(eigenvalues))])
        print('final_eigenvectors:')
        for k in range(eigenvectors.shape[0]):
            print(["{:.2f}".format(eigenvectors[k][j]) for j in range(eigenvectors.shape[1])])

    return eigenvalues, eigenvectors
