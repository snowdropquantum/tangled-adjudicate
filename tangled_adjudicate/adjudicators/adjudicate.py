""" Adjudicator class for Tangled game states using Schrödinger Equation, Simulated Annealing, and D-Wave hardware """
import sys
import os
import time
import pickle
import random
import neal
import dimod
import numpy as np
import matplotlib.pyplot as plt

from tangled_adjudicate.utils.utilities import game_state_to_ising_model, game_state_is_terminal, find_isolated_vertices
from tangled_adjudicate.utils.find_graph_automorphisms import get_automorphisms
from tangled_adjudicate.utils.find_hardware_embeddings import get_embeddings
from tangled_adjudicate.schrodinger.schrodinger_functions import evolve_schrodinger
from tangled_adjudicate.utils.parameters import Params

from dwave.system import DWaveSampler, FixedEmbeddingComposite
from dwave.system.testing import MockDWaveSampler


class Adjudicator(object):
    def __init__(self, params):
        self.params = params
        # if we are using the QC, load embeddings and automorphisms from /data; if they are not there,
        # compute them for your processor choice
        if self.params.USE_QC:
            self.automorphisms = get_automorphisms(self.params.GRAPH_NUMBER)
            self.embeddings = get_embeddings(self.params.GRAPH_NUMBER, self.params.QC_SOLVER_TO_USE)

    def compute_winner_score_and_influence_from_correlation_matrix(self, game_state, correlation_matrix):
        # correlation_matrix is assumed to be symmetric matrix with zeros on diagonal (so that self-correlation of
        # one is not counted) -- this is the standard for computing influence vector
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

    def simulated_annealing(self, game_state):

        # this function inputs game_state, e.g.:
        #
        # game_state = {'num_nodes': 6, 'edges': [(0, 1, 1), (0, 2, 1), (0, 3, 2), (0, 4, 3), (0, 5, 2), (1, 2, 1),
        # (1, 3, 2), (1, 4, 3), (1, 5, 3), (2, 3, 1), (2, 4, 2), (2, 5, 3), (3, 4, 2), (3, 5, 1), (4, 5, 2)],
        # 'player1_id': 'player1', 'player2_id': 'player2', 'turn_count': 17, 'current_player_index': 1,
        # 'player1_node': 1, 'player2_node': 3}
        #
        # and returns:
        #
        # winner: string, one of 'red', 'blue', 'draw'
        # score_difference: real number, the score of the game
        # influence_vector: vector of real numbers of length vertex_count, the influence of each vertex

        h, jay = game_state_to_ising_model(game_state)

        sampler = neal.SimulatedAnnealingSampler()

        # Approx match: (1) mean energy and (2) rate of local excitations for square-lattice high precision spin glass
        # at 5ns (Advantage2 prototype 2p3)

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

        number_of_embeddings = len(self.embeddings)                 # e.g. 346
        # number_of_embeddings = 2               # e.g. 346
        number_of_problem_variables = game_state['num_nodes']       # e.g. 3

        samples = np.zeros((1, number_of_problem_variables))        # 0th layer to get vstack going, remove at the end

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
            shim_stats = None

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
        # parallel ways across the entire chip, and then applying a gauge transform across all the qubits used.
        # This latter process chooses different random gauges for each of the embedded instances.

        for chip_run_idx in range(self.params.NUMBER_OF_CHIP_RUNS):

            # *******************************************************************
            # Step 1: Randomly select an automorphism and embed it multiple times
            # *******************************************************************

            automorphism_to_use = random.choice(self.automorphisms)  # eg {0:0, 1:2, 2:1}
            # automorphism_to_use = {0: 2, 1: 0, 2: 1}
            inverted_automorphism_to_use = {v: k for k, v in automorphism_to_use.items()}   # swaps key <-> values

            permuted_embedding = []

            for each_embedding in self.embeddings[:number_of_embeddings]:    # each_embedding is like [1093, 1098, 136]; 343 for three-vertex graph
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

            # next we want to apply a random gauge transformation. I could do this using the
            # SpinReversalTransformComposite function, but I'm not exactly sure how it works, so instead I'm going
            # to just homebrew my own (it's pretty simple, just some bookkeeping to get right). I call the
            # situation after the gauge transformation has been applied the BLUE with RED STAR situation.

            if self.params.USE_GAUGE_TRANSFORM:
                flip_map = [random.choice([-1, 1]) for _ in full_h]   # random list of +1, -1 values of len # qubits
                # flip_map = [-1, 1, -1, 1, 1, -1]   # random list of +1, -1 values of len # qubits
                indices_of_flips = [i for i, x in enumerate(flip_map) if x == -1]       # the indices of the -1 values

                for edge_key, j_val in full_j.items():              # for each edge and associated J value
                    full_j[edge_key] = j_val * flip_map[edge_key[0]] * flip_map[edge_key[1]]   # Jij -> J_ij g_i g_j

            # *****************************************
            # Step 4: Choose sampler and its parameters
            # *****************************************

            sampler_kwargs.update({'h': full_h,
                                   'J': full_j})

            # # I think this is for making sure the variable order doesn't get screwed up later
            # bqm = dimod.BinaryQuadraticModel('SPIN').from_ising(full_h, full_j)

            # applies the desired embedding
            sampler = FixedEmbeddingComposite(base_sampler, embedding=embedding_to_use)

            # ********************************************************************************
            # Step 5: Start linear bias terms shimming process in the BLUE with RED STAR basis
            # ********************************************************************************

            # all of this in the BLUE with RED STAR basis, ie post automorph, post gauge transform
            for shim_iteration_idx in range(shim_iterations):

                # **************************************
                # Step 6: Generate samples from hardware
                # **************************************

                ss = sampler.sample_ising(**sampler_kwargs)
                # ss = dimod.SampleSet.from_samples_bqm(ss, bqm)    # might not be needed -- to check later
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

            all_samples_processed_black = all_samples_processed_blue[:,
                                          [automorphism_to_use[i] for i in range(all_samples_processed_blue.shape[1])]]

            # *********************************************************
            # Step 10: Add new samples to the stack, all in BLACK order
            # *********************************************************

            samples = np.vstack((samples, all_samples_processed_black))

        # ***************************************************************
        # Step 11: Post process samples stack to extract return variables
        # ***************************************************************

        samples = np.delete(samples, (0), axis=0)  # delete first row of zeros

        # replace columns where there are disconnect variables with truly random samples... seems to work?
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

        # return return_dictionary, shim_stats
        return return_dictionary
        #
        #     print()
        # # ********************************
        # # Step 1: Balancing qubits at zero
        # # ********************************
        # #
        # # Because we have all h_j=0, the resultant Ising model is of the form H = \sum_ij J_ij s_i s_j. This
        # # is symmetric under reversal of the signs on all spins +1 <-> -1, which means that every state has a
        # # symmetric state with the spins reversed, which means that the average value of each spin is zero. In
        # # other words, if I collect N samples from this Ising model from hardware, if I look at each spin
        # # independently, if I get an average value <s_j> different from zero this indicates an unwanted analog bias.
        # # D-Wave exposes a parameter called Flux Bias Offsets (FBOs) that can then be used to compensate for this.
        #
        #
        # # number of automorphisms of the game graph: len(self.automorphisms)
        # # number of source to target embeddings found: len(self.embeddings)
        # # number of spin reversal transformations used: self.params.SPIN_REVERSAL_TRANSFORMS
        # # number of samples taken per call to the processor: self.params.NUM_READS
        # # for graph_number 2 ==> these are 6, 346, 10, 100 respectively ==> 2,076,000 samples
        #
        # # self.embeddings[0] = [1093, 1098, 136]
        # num_var = len(self.embeddings[0])  # e.g. 3
        # number_of_automorphisms_P = len(self.automorphisms)  # e.g. 6
        # number_of_embeddings_to_use_U = len(self.embeddings)  # e.g. 346
        #
        # # big_h and big_j are h, jay values for the entire chip assuming the identity automorphism
        #
        # big_h = {}
        # big_j = {}
        #
        # for k in range(number_of_embeddings_to_use_U):
        #     for j in range(num_var):
        #         big_h[num_var * k + j] = 0
        #
        # for k, v in base_jay.items():
        #     big_j[k] = v
        #     for j in range(1, number_of_embeddings_to_use_U):
        #         big_j[(k[0] + num_var * j, k[1] + num_var * j)] = v
        #
        # sampler_kwargs = dict(h=big_h,
        #                       J=big_j,
        #                       num_reads=self.params.NUM_READS_QC,
        #                       answer_mode='raw',
        #                       num_spin_reversal_transforms=self.params.SPIN_REVERSAL_TRANSFORMS)
        #
        # if self.params.USE_MOCK_DWAVE_SAMPLER:
        #     base_sampler = MockDWaveSampler(topology_type='zephyr', topology_shape=[6, 4])
        #
        # else:
        #     base_sampler = DWaveSampler(solver='Advantage2_prototype2.4')
        #     sampler_kwargs.update({'fast_anneal': True,
        #                            'annealing_time': self.params.ANNEAL_TIME_IN_NS / 1000})
        #
        # # I think this is for making sure the variable order doesn't get screwed up later
        # bqm = dimod.BinaryQuadraticModel('SPIN').from_ising(big_h, big_j)
        #
        # samps = np.zeros((1, num_var))   # first layer of zeros to get vstack going, will remove at the end
        #
        # # for each automorphism, we embed using that automorphism into all the available places on the chip, and
        # # collect N samples
        #
        # for automorphism_idx in range(number_of_automorphisms_P):    # ranges from 0 to 5
        #
        #     automorphism_to_use = self.automorphisms[automorphism_idx]    # eg {0:0, 1:2, 2:1}
        #     permuted_embedding = []
        #     for each_embedding in self.embeddings:    # each_embedding is like [1093, 1098, 136]
        #         this_embedding = []
        #         for each_vertex in range(num_var):    # each_vertex ranges from 0 to 2
        #             this_embedding.append(each_embedding[automorphism_to_use[each_vertex]])
        #         permuted_embedding.append(this_embedding)
        #
        #     embedding_to_use = {}
        #
        #     for k in range(number_of_embeddings_to_use_U):
        #         for j in range(num_var):  # up to 0..1037
        #             embedding_to_use[num_var * k + j] = [permuted_embedding[k][j]]
        #
        #     composed_sampler = (
        #         SpinReversalTransformComposite(FixedEmbeddingComposite(base_sampler, embedding=embedding_to_use)))
        #
        #     ss = composed_sampler.sample_ising(**sampler_kwargs)
        #     ss = dimod.SampleSet.from_samples_bqm(ss, bqm)
        #     new_samps = self.ss_to_samps(ss, num_var, number_of_embeddings)
        #     samps = np.vstack((samps, new_samps))   # stack new_samps from this automorphism
        #
        # samps = np.delete(samps, (0), axis=0)   # delete first row of zeros
        #
        # sample_count = self.params.NUM_READS_QC * self.params.SPIN_REVERSAL_TRANSFORMS * number_of_embeddings_to_use_U * number_of_automorphisms_P
        #
        # # this is a full matrix with zeros on the diagonal that uses all the samples
        # correlation_matrix = \
        #     (np.einsum('si,sj->ij', samps, samps) / sample_count -
        #      np.eye(int(game_state['num_nodes'])))
        #
        # winner, score_difference, influence_vector = self.compute_winner_score_and_influence_from_correlation_matrix(game_state, correlation_matrix)
        #
        # return_dictionary = {'game_state': game_state, 'adjudicator': 'quantum_annealing',
        #                      'winner': winner, 'score': score_difference, 'influence_vector': influence_vector,
        #                      'correlation_matrix': correlation_matrix, 'parameters': self.params}
        #
        # return return_dictionary


def test_two_instances():

    precision_digits = 3
    np.set_printoptions(precision=precision_digits)    # just to clean up print output
    np.set_printoptions(suppress=True)                 # remove scientific notation

    params = Params()
    adjudicator = Adjudicator(params=params)

    # solvers_to_use = ['simulated_annealing', 'schrodinger_equation', 'quantum_annealing']
    solvers_to_use = ['quantum_annealing']

    # first game_state is an FM locked thing, so only 1,1,1 and -1,-1,-1 should appear; score should be 0 (draw),
    # correlation matrix should be [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    # second game_state has 2 FM links and one AFM link, with 6 degenerate states; score should be -2/3 (blue),
    # correlation matrix should be [[0, -1/3, +1/3], [-1/3, 0, +1/3], [+1/3, +1/3, 0]]

    # game_states = [{'num_nodes': 3, 'edges': [(0, 1, 2), (0, 2, 2), (1, 2, 2)], 'player1_id': 'player1',
    #                 'player2_id': 'player2', 'turn_count': 17, 'current_player_index': 1, 'player1_node': 1,
    #                 'player2_node': 2},
    #                {'num_nodes': 3, 'edges': [(0, 1, 3), (0, 2, 2), (1, 2, 2)], 'player1_id': 'player1',
    #                 'player2_id': 'player2', 'turn_count': 17, 'current_player_index': 1, 'player1_node': 1,
    #                 'player2_node': 2}]

    game_states = [{'num_nodes': 3, 'edges': [(0, 1, 1), (0, 2, 1), (1, 2, 2)], 'player1_id': 'player1',
                    'player2_id': 'player2', 'turn_count': 17, 'current_player_index': 1, 'player1_node': 1,
                    'player2_node': 2}]

    # game_states = [{'num_nodes': 3, 'edges': [(0, 1, 3), (0, 2, 2), (1, 2, 1)], 'player1_id': 'player1',
    #                 'player2_id': 'player2', 'turn_count': 17, 'current_player_index': 1, 'player1_node': 1,
    #                 'player2_node': 2}]

    # Note: the D-Wave Mock solver will get Instance 2 wrong as it does not implement an unbiased sampler

    for solver_to_use in solvers_to_use:
        cnt = 0
        for game_state in game_states:

            start = time.time()

            results, shim_stats = getattr(adjudicator, solver_to_use)(game_state)

            print('*******************************************')
            print('using the', solver_to_use, 'solver on Instance', cnt + 1)
            print('using parameters M=', params.NUMBER_OF_CHIP_RUNS, 'N=', params.NUM_READS_QC,
                  'USE_SHIM=', params.USE_SHIM, 'USE_GAUGE_TRANSFORM=', params.USE_GAUGE_TRANSFORM,
                  'S=', params.SHIM_ITERATIONS)
            print('run took', round(time.time() - start, precision_digits), 'seconds.')
            print('score was:', round(results['score'], precision_digits))
            print('winner:', results['winner'])
            print('correlation matrix:')
            print(np.array(results['correlation_matrix']))
            print('*******************************************')

            # plot_shim_stats(shim_stats)

            cnt += 1


def run_experiment_for_blog_post(params, file_path):

    number_of_scores_computed_per_parameter_setting = 20

    precision_digits = 3
    np.set_printoptions(precision=precision_digits)    # just to clean up print output
    np.set_printoptions(suppress=True)                 # remove scientific notation

    # first game_state is an FM locked thing, so only 1,1,1 and -1,-1,-1 should appear; score should be 0 (draw),
    # correlation matrix should be [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    # second game_state has 2 FM links and one AFM link, with 6 degenerate states; score should be -2/3 (blue),
    # correlation matrix should be [[0, -1/3, +1/3], [-1/3, 0, +1/3], [+1/3, +1/3, 0]]

    game_states = [{'num_nodes': 3, 'edges': [(0, 1, 2), (0, 2, 2), (1, 2, 2)], 'player1_id': 'player1',
                    'player2_id': 'player2', 'turn_count': 17, 'current_player_index': 1, 'player1_node': 1,
                    'player2_node': 2},
                   {'num_nodes': 3, 'edges': [(0, 1, 3), (0, 2, 2), (1, 2, 2)], 'player1_id': 'player1',
                    'player2_id': 'player2', 'turn_count': 17, 'current_player_index': 1, 'player1_node': 1,
                    'player2_node': 2}]

    # # m_and_n_list = [(50, 1), (10, 5), (5, 10), (1, 50)]
    # m_and_n_list = [(8, 1), (4, 2), (2, 4), (1, 8)]
    # # s_list = [1, 10]
    # s_list = [1, 10]
    #
    # use_gauge_list = [False, True]

    m_and_n_list = [(1, 100)]
    s_list = [1]
    use_gauge_list = [True]

    instance_idx = 1

    score_and_timing_data = {}

    for game_state in game_states:
        score_and_timing_data[instance_idx] = {}

        for m_and_n_values in m_and_n_list:
            score_and_timing_data[instance_idx][m_and_n_values] = {}

            params.NUMBER_OF_CHIP_RUNS = m_and_n_values[0]
            params.NUM_READS_QC = m_and_n_values[1]

            for s in s_list:
                score_and_timing_data[instance_idx][m_and_n_values][s] = {}

                if s == 1:
                    params.USE_SHIM = False
                else:
                    params.USE_SHIM = True
                    params.SHIM_ITERATIONS = s

                for use_gauge in use_gauge_list:
                    score_and_timing_data[instance_idx][m_and_n_values][s][use_gauge] = []

                    params.USE_GAUGE_TRANSFORM = use_gauge
                    adjudicator = Adjudicator(params=params)

                    score_data = []
                    timing_data = []

                    for _ in range(number_of_scores_computed_per_parameter_setting):
                        start = time.time()
                        results = adjudicator.quantum_annealing(game_state)
                        score_data.append(results['score'])
                        timing_data.append(time.time() - start)

                    print('*******************************************')
                    print('with parameters', m_and_n_values, s, use_gauge, 'on Instance', instance_idx)
                    print('average time was', round(sum(timing_data)/number_of_scores_computed_per_parameter_setting, precision_digits), 'seconds.')
                    print('scores were:', np.array(score_data))
                    print('*******************************************')

                    score_and_timing_data[instance_idx][m_and_n_values][s][use_gauge].append(score_data)
                    score_and_timing_data[instance_idx][m_and_n_values][s][use_gauge].append(sum(timing_data)/number_of_scores_computed_per_parameter_setting)
        instance_idx += 1

    with open(file_path, "wb") as fp:
        pickle.dump(score_and_timing_data, fp)

    return score_and_timing_data


def generate_histograms(score_and_timing_data):

    for instance_key, instance_value in score_and_timing_data.items():

        # Create a new 4x4 grid of subplots for each instance
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))

        # Flatten the 4x4 grid for easy iteration
        axes = axes.flatten()

        score_data = []
        timing_data = []
        label_data = []

        for m_and_n_key, m_and_n_value in score_and_timing_data[instance_key].items():
            for s_key, s_value in score_and_timing_data[instance_key][m_and_n_key].items():
                for gauge_key, gauge_value in score_and_timing_data[instance_key][m_and_n_key][s_key].items():
                    score_data.append(score_and_timing_data[instance_key][m_and_n_key][s_key][gauge_key][0])
                    timing_data.append(score_and_timing_data[instance_key][m_and_n_key][s_key][gauge_key][1])
                    label_data.append([['M:' + str(m_and_n_key[0]) + ' N:' + str(m_and_n_key[1])],
                                       ['S:' + str(s_key) + ' G:' + str(gauge_key)]])
        for i, ax in enumerate(axes):
            # Generate some random data (for example, normal distribution)
            data = score_data[i]

            if instance_key == 1:
                ax.hist(data, range=[-0.01, 0.01], bins=101, color='skyblue', edgecolor='black')
                ax.vlines(x=0, ymin=0, ymax=10, colors='green', ls=':', lw=1)
            else:
                ax.hist(data, range=[-0.766, -0.566], bins=101, color='skyblue', edgecolor='black')
                ax.vlines(x=-2/3, ymin=0, ymax=10, colors='green', ls=':', lw=1)

            # Set title and labels for each subplot
            if not i % 4:
                ax.set_ylabel(label_data[i][0][0])
            if i < 4:
                ax.set_title(label_data[i][1][0])
            ax.set_xlabel('Score')

        # Adjust the layout to prevent overlapping
        plt.tight_layout()
        plt.show()


def plot_shim_stats(shim_stats):

    # this was obtained from a run on the 'num_nodes': 3, 'edges': [(0, 1, 3), (0, 2, 2), (1, 2, 2)] instance
    # first 10 iterations no shimming, then 40 more with, last 10 of which are plotted here
    fig, axes = plt.subplots(1, 1, figsize=(10, 5))

    shim_stats['average_absolute_value_of_magnetization'] = [0.10814771622934889, 0.10826239067055393, 0.1102818270165209, 0.11273858114674443, 0.10896404275996113, 0.11055587949465501, 0.11308843537414967, 0.10874829931972789, 0.10925364431486882, 0.11073275024295433, 0.10615743440233238, 0.08552769679300291, 0.07585034013605443, 0.06724198250728863, 0.05812827988338193, 0.05728668610301264, 0.05562487852283772, 0.05186005830903791, 0.0526569484936832, 0.05058697764820214, 0.04690379008746356, 0.05048979591836735, 0.04924586977648203, 0.047471331389698744, 0.04947716229348883, 0.04630320699708455, 0.04893488824101069, 0.05095043731778426, 0.04984450923226434, 0.05010495626822158, 0.04937026239067056, 0.049125364431486886, 0.04987172011661808, 0.04875413022351798, 0.04630903790087464, 0.05062779397473276, 0.048464528668610306, 0.04921671525753159, 0.047152575315840634, 0.050534499514091356, 0.047959183673469394, 0.04811856171039845, 0.04725558794946551, 0.04656948493683188, 0.05002137998056366, 0.05048979591836735, 0.048800777453838685, 0.05110398445092323, 0.05287269193391643, 0.04948299319727892]
    colors = ['orange', 'skyblue']
    data = {0: shim_stats['average_absolute_value_of_magnetization'][:10],
            1: shim_stats['average_absolute_value_of_magnetization'][40:]}
    axes.hist(data[0], range=[0, 0.2], bins=50, color=colors[0], label='First 10 Runs Before Shim', edgecolor='black')
    axes.hist(data[1], range=[0, 0.2], bins=50, color=colors[1], label='Last 10 Runs of Shim', edgecolor='black')

    axes.set_xlabel('Mean Absolute Value of Qubit Magnetizations')
    axes.legend(prop={'size': 10})
    # Adjust the layout to prevent overlapping
    plt.tight_layout()
    plt.show()


def main():

    # plot_shim_stats({})
    test_two_instances()

    # params = Params()
    #
    # file_name_prefix = "graph_" + str(params.GRAPH_NUMBER)
    # data_dir = os.path.join(os.getcwd(), '..', 'data')
    # file_path = os.path.join(data_dir, file_name_prefix + "_blog_post_data_results_good.pkl")
    #
    # if os.path.isfile(file_path):
    #     with open(file_path, "rb") as fp:
    #         score_and_timing_data = pickle.load(fp)
    # else:   # in this case, there are no results yet, so compute them!
    #     score_and_timing_data = run_experiment_for_blog_post(params, file_path)
    #
    # generate_histograms(score_and_timing_data=score_and_timing_data)


if __name__ == "__main__":
    sys.exit(main())
