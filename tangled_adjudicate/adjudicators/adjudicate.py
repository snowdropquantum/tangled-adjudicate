""" Adjudicator class for Tangled game states using Schrödinger Equation, Simulated Annealing, and D-Wave hardware """
import random
import neal
# import dimod
import numpy as np

from tangled_adjudicate.utils.utilities import game_state_to_ising_model, game_state_is_terminal, find_isolated_vertices
from tangled_adjudicate.utils.find_graph_automorphisms import get_automorphisms
from tangled_adjudicate.utils.find_hardware_embeddings import get_embeddings
from tangled_adjudicate.utils.parameters import Params
from tangled_adjudicate.schrodinger.schrodinger_functions import evolve_schrodinger

from dwave.system import DWaveSampler, FixedEmbeddingComposite
from dwave.system.testing import MockDWaveSampler


class Adjudicator(object):
    def __init__(self, params):
        self.params = params
        if self.params.USE_QC:   # if using QC, get embeddings and automorphisms
            self.automorphisms = get_automorphisms(self.params.GRAPH_NUMBER)
            self.embeddings = get_embeddings(self.params.GRAPH_NUMBER, self.params.QC_SOLVER_TO_USE)

    def compute_winner_score_and_influence_from_correlation_matrix(self, game_state, correlation_matrix):
        # correlation_matrix is assumed to be symmetric matrix with zeros on diagonal (so that self-correlation of
        # one is not counted) -- this is the standard for computing influence vector
        #
        # returns:
        # winner: if game_state is terminal, string -- one of 'red' (player 1), 'blue' (player 2), 'draw'
        # if game_state not terminal, returns None
        # score: if game_state is terminal, returns a real number which is the score of the game (difference
        # between two players' influences obtained from the influence vector)
        # if game_state not terminal, returns None
        # influence_vector: a vector of real numbers of length == number of vertices; this stores each vertex's
        # influence, which is the sum over all elements of the correlation matrix it is part of

        influence_vector = np.sum(correlation_matrix, axis=0)

        if game_state_is_terminal(game_state):
            score = influence_vector[game_state['player1_node']] - influence_vector[game_state['player2_node']]

            if score > self.params.EPSILON:  # more positive than epsilon, red wins
                winner = 'red'
            else:
                if score < -self.params.EPSILON:
                    winner = 'blue'
                else:
                    winner = 'draw'
        else:
            score = None
            winner = None

        return winner, score, influence_vector

    # all three solver functions input game_state, e.g.:
    #
    # game_state = {'num_nodes': 6, 'edges': [(0, 1, 1), (0, 2, 1), (0, 3, 2), (0, 4, 3), (0, 5, 2), (1, 2, 1),
    # (1, 3, 2), (1, 4, 3), (1, 5, 3), (2, 3, 1), (2, 4, 2), (2, 5, 3), (3, 4, 2), (3, 5, 1), (4, 5, 2)],
    # 'player1_id': 'player1', 'player2_id': 'player2', 'turn_count': 17, 'current_player_index': 1,
    # 'player1_node': 1, 'player2_node': 3}
    #
    # and return a dictionary that contains the following keys:
    #
    # 'game_state': a copy of the input game_state dictionary
    # 'adjudicator': a string, one of 'simulated_annealing', 'quantum_annealing', 'schrodinger_equation'
    # 'winner': if both players have chosen vertices, a string, one of 'red', 'blue', 'draw', otherwise None
    # 'score': if both players have chosen vertices, the difference in influence scores as a real number, otherwise None
    # 'influence_vector': a vector of real numbers of length vertex_count (one real number per vertex in the game graph)
    # 'correlation_matrix': symmetric real-valued matrix of spin-spin correlations with zeros on diagonals
    # 'parameters': a copy of the parameters dictionary

    def simulated_annealing(self, game_state):

        h, jay = game_state_to_ising_model(game_state)
        sampler = neal.SimulatedAnnealingSampler()

        # Approx match: (1) mean energy and (2) rate of local excitations for square-lattice high precision spin glass
        # at 5ns (Advantage2 prototype 2.5)

        # Limits relaxation to local minima. Can vary by model/protocol/QPU. Assumes max(|J|) is scaled to 1.
        beta_max = 3
        # Limits equilibration. Can vary by model/protocol/QPU
        num_sweeps = 16
        beta_range = [1 / np.sqrt(np.sum([Jij ** 2 for Jij in jay.values()]) + 0.001), beta_max]   # 0.001 for J==0
        seed = None   # Choose seed=None if reproducibility is not desired

        # randomize_order=True implements standard symmetry-respecting Metropolis algorithm
        ss = sampler.sample_ising(h, jay, beta_range=beta_range, num_reads=self.params.NUM_READS_SA,
                                  num_sweeps=num_sweeps, randomize_order=True, seed=seed)

        samps = np.array(ss.record.sample, dtype=float)  # casting may not be necessary.

        # creates symmetric matrix with zeros on diagonal (so that self-correlation of one is not counted) -- this is
        # the standard for computing influence vector
        correlation_matrix = (np.einsum('si,sj->ij', samps, samps) / self.params.NUM_READS_SA -
                              np.eye(int(game_state['num_nodes'])))

        winner, score_difference, influence_vector = (
            self.compute_winner_score_and_influence_from_correlation_matrix(game_state, correlation_matrix))

        return_dictionary = {'game_state': game_state, 'adjudicator': 'simulated_annealing',
                             'winner': winner, 'score': score_difference, 'influence_vector': influence_vector,
                             'correlation_matrix': correlation_matrix, 'parameters': self.params}

        return return_dictionary

    def schrodinger_equation(self, game_state):

        h, jay = game_state_to_ising_model(game_state)

        s_min = 0.001   # beginning and ending anneal times
        s_max = 0.999

        correlation_matrix = (
            evolve_schrodinger(h, jay, s_min=s_min, s_max=s_max, tf=self.params.ANNEAL_TIME_IN_NS,
                               n_qubits=game_state['num_nodes']))
        # what's returned here is upper triangular with zeros on the diagonal, so we need to add the transpose
        correlation_matrix = correlation_matrix + correlation_matrix.T

        winner, score_difference, influence_vector = (
            self.compute_winner_score_and_influence_from_correlation_matrix(game_state, correlation_matrix))

        return_dictionary = {'game_state': game_state, 'adjudicator': 'schrodinger_equation',
                             'winner': winner, 'score': score_difference, 'influence_vector': influence_vector,
                             'correlation_matrix': correlation_matrix, 'parameters': self.params}

        return return_dictionary

    def quantum_annealing(self, game_state):

        number_of_embeddings = len(self.embeddings)                 # e.g. P=343
        number_of_problem_variables = game_state['num_nodes']       # e.g. 3

        samples = np.zeros((1, number_of_problem_variables))        # 0th layer to get vstack going, remove at the end
        shim_stats = None
        all_samples = None
        indices_of_flips = None

        if self.params.USE_MOCK_DWAVE_SAMPLER and self.params.USE_SHIM:
            print('D-Wave mock sampler is not set up to use the shimming process, turn shim off if using mock!')

        sampler_kwargs = dict(num_reads=self.params.NUM_READS_QC,
                              answer_mode='raw')

        if self.params.USE_MOCK_DWAVE_SAMPLER:
            base_sampler = MockDWaveSampler(topology_type='zephyr', topology_shape=[6, 4])
        else:
            base_sampler = DWaveSampler(solver=self.params.QC_SOLVER_TO_USE)
            sampler_kwargs.update({'fast_anneal': True,
                                   'annealing_time': self.params.ANNEAL_TIME_IN_NS / 1000})

        if self.params.USE_SHIM:
            shim_stats = {'qubit_magnetizations': [],
                          'average_absolute_value_of_magnetization': [],
                          'all_flux_bias_offsets': []}
            sampler_kwargs.update({'readout_thermalization': 100.,
                                   'auto_scale': False,
                                   'flux_drift_compensation': True,
                                   'flux_biases': [0] * base_sampler.properties['num_qubits']})
            shim_iterations = self.params.SHIM_ITERATIONS
        else:
            shim_iterations = 1    # if we don't shim, just run through shim step only once

        # **********************************************************
        # Step 0: convert game_state to the desired base Ising model
        # **********************************************************

        # for tangled, h_j=0 for all vertices j in the game graph, and J_ij is one of +1, -1, or 0 for all vertex
        # pairs i,j. I named the "base" values (the actual problem defined on the game graph we are asked to solve)
        # base_h (all zero) and base_jay (not all zero).

        base_h, base_jay = game_state_to_ising_model(game_state)

        # this finds any isolated vertices that may be in the graph -- we will replace the samples returned for these
        # at the end with true 50/50 statistics, so we don't have to worry about them

        isolated_vertices = find_isolated_vertices(number_of_problem_variables, base_jay)

        # We now enter a loop where each pass through the loop programs the chip to specific values of h and J but
        # now for the entire chip. We do this by first selecting one automorphism and embedding it in multiple
        # parallel ways across the entire chip, and then optionally applying a gauge transform across all the qubits
        # used. This latter process chooses different random gauges for each of the embedded instances.

        for chip_run_idx in range(self.params.NUMBER_OF_CHIP_RUNS):

            # *******************************************************************
            # Step 1: Randomly select an automorphism and embed it multiple times
            # *******************************************************************

            automorphism_to_use = random.choice(self.automorphisms)  # eg {0:0, 1:2, 2:1}
            inverted_automorphism_to_use = {v: k for k, v in automorphism_to_use.items()}   # swaps key <-> values

            permuted_embedding = []

            for each_embedding in self.embeddings[:number_of_embeddings]:    # each_embedding is like [1093, 1098, 136]; 343 of these for three-vertex graph
                this_embedding = []
                for each_vertex in range(number_of_problem_variables):    # each_vertex ranges from 0 to 2
                    this_embedding.append(each_embedding[inverted_automorphism_to_use[each_vertex]])
                permuted_embedding.append(this_embedding)

            # given that permuted_embedding looks like [[1229, 1235, 563], [872, 242, 866], ...]
            # this next part converts into the format {0: [1229], 1: [1235], 2: [563], 3: [872], 4: [242], 5: [866]}

            embedding_to_use = {}

            for embedding_idx in range(number_of_embeddings):
                for each_vertex in range(number_of_problem_variables):  # up to 0..1037
                    embedding_to_use[number_of_problem_variables * embedding_idx + each_vertex] = \
                        [permuted_embedding[embedding_idx][each_vertex]]

            # *****************************************************************************************************
            # Step 2: Set h, J parameters for full chip using parallel embeddings of a randomly chosen automorphism
            # *****************************************************************************************************

            # compute full_h and full_j which are h, jay values for the entire chip assuming the above automorphism
            # I am calling the problem definition and variable ordering before the automorphism the BLACK or BASE
            # situation. After the automorphism the problem definition and variable labels change -- I'm calling the
            # situation after the automorphism has been applied the BLUE situation.

            full_h = {}
            full_j = {}

            for embedding_idx in range(number_of_embeddings):
                for each_vertex in range(number_of_problem_variables):
                    full_h[number_of_problem_variables * embedding_idx + each_vertex] = 0

            for k, v in base_jay.items():
                edge_under_automorph = (min(automorphism_to_use[k[0]], automorphism_to_use[k[1]]),
                                        max(automorphism_to_use[k[0]], automorphism_to_use[k[1]]))
                full_j[edge_under_automorph] = v
                for j in range(1, number_of_embeddings):
                    full_j[(edge_under_automorph[0] + number_of_problem_variables * j,
                            edge_under_automorph[1] + number_of_problem_variables * j)] = v

            # **************************************************************************
            # Step 3: Choose random gauge, modify h, J parameters for full chip using it
            # **************************************************************************

            # next we optionally apply a random gauge transformation. I call the situation after the gauge
            # transformation has been applied the BLUE with RED STAR situation.

            if self.params.USE_GAUGE_TRANSFORM:
                flip_map = [random.choice([-1, 1]) for _ in full_h]   # random list of +1, -1 values of len # qubits
                indices_of_flips = [i for i, x in enumerate(flip_map) if x == -1]       # the indices of the -1 values

                for edge_key, j_val in full_j.items():              # for each edge and associated J value
                    full_j[edge_key] = j_val * flip_map[edge_key[0]] * flip_map[edge_key[1]]   # Jij -> J_ij g_i g_j

            # *****************************************
            # Step 4: Choose sampler and its parameters
            # *****************************************

            sampler_kwargs.update({'h': full_h,
                                   'J': full_j})

            sampler = FixedEmbeddingComposite(base_sampler, embedding=embedding_to_use)   # applies the embedding

            # *************************************************************************
            # Step 5: Optionally start shimming process in the BLUE with RED STAR basis
            # *************************************************************************

            # all of this in the BLUE with RED STAR basis, ie post automorph, post gauge transform
            for shim_iteration_idx in range(shim_iterations):

                # **************************************
                # Step 6: Generate samples from hardware
                # **************************************

                ss = sampler.sample_ising(**sampler_kwargs)
                all_samples = ss.record.sample

                if self.params.USE_SHIM:

                    # *************************************************************
                    # Step 6a: Compute average values of each qubit == magnetization
                    # *************************************************************

                    magnetization = np.sum(all_samples, axis=0)/self.params.NUM_READS_QC   # BLUE with RED STAR label ordering
                    shim_stats['average_absolute_value_of_magnetization'].append(np.sum([abs(k) for k in magnetization])/len(magnetization))

                    qubit_magnetization = [0] * base_sampler.properties['num_qubits']
                    for k, v in embedding_to_use.items():
                        qubit_magnetization[v[0]] = magnetization[k]        # check

                    shim_stats['qubit_magnetizations'].append(qubit_magnetization)

                    # **************************************
                    # Step 6b: Adjust flux bias offset terms
                    # **************************************

                    for k in range(base_sampler.properties['num_qubits']):
                        sampler_kwargs['flux_biases'][k] -= self.params.ALPHA_PHI * qubit_magnetization[k]

                    shim_stats['all_flux_bias_offsets'].append(sampler_kwargs['flux_biases'])

            # *****************************************************************************************************
            # Step 7: Reverse gauge transform, from BLUE with RED STAR to just BLUE, after shimming process is done
            # *****************************************************************************************************

            if self.params.USE_GAUGE_TRANSFORM:
                all_samples[:, indices_of_flips] = -all_samples[:, indices_of_flips]

            # ***********************************
            # Step 8: Stack samples in BLUE order
            # ***********************************

            # this should make a big fat stack of the results in BLUE variable ordering
            all_samples_processed_blue = all_samples[:, range(number_of_problem_variables)]
            for k in range(1, number_of_embeddings):
                all_samples_processed_blue = np.vstack((all_samples_processed_blue,
                                                        all_samples[:, range(number_of_problem_variables * k,
                                                                             number_of_problem_variables * (k + 1))]))

            # **********************************************************************
            # Step 9: Reorder columns to make them BLACK order instead of BLUE order
            # **********************************************************************

            all_samples_processed_black = all_samples_processed_blue[:, [automorphism_to_use[i] for i in range(all_samples_processed_blue.shape[1])]]

            # *********************************************************
            # Step 10: Add new samples to the stack, all in BLACK order
            # *********************************************************

            samples = np.vstack((samples, all_samples_processed_black))

        # ***************************************************************
        # Step 11: Post process samples stack to extract return variables
        # ***************************************************************

        samples = np.delete(samples, (0), axis=0)  # delete first row of zeros

        # replace columns where there are disconnected variables with truly random samples
        for idx in isolated_vertices:
            samples[:, idx] = np.random.choice([1, -1], size=samples.shape[0])

        sample_count = self.params.NUM_READS_QC * number_of_embeddings * self.params.NUMBER_OF_CHIP_RUNS

        # this is a full matrix with zeros on the diagonal that uses all the samples
        correlation_matrix = \
            (np.einsum('si,sj->ij', samples, samples) / sample_count -
                np.eye(int(game_state['num_nodes'])))

        winner, score_difference, influence_vector = (
            self.compute_winner_score_and_influence_from_correlation_matrix(game_state, correlation_matrix))

        return_dictionary = {'game_state': game_state, 'adjudicator': 'quantum_annealing',
                             'winner': winner, 'score': score_difference, 'influence_vector': influence_vector,
                             'correlation_matrix': correlation_matrix, 'parameters': self.params}

        return return_dictionary
