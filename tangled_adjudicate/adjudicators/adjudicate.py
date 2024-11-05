""" evaluation of Tangled terminal states using simulated annealing """
import neal
import os
import pickle
import dimod
import numpy as np
from utils.utilities import game_state_to_ising_model
from schrodinger.schrodinger_functions import evolve_schrodinger
from dwave.system import DWaveSampler, FixedEmbeddingComposite
from dwave.system.testing import MockDWaveSampler
# from dwave.preprocessing.composites import SpinReversalTransformComposite
from utils.modified_srt import SpinReversalTransformCompositeModified


class Adjudicator(object):
    def __init__(self, params):
        self.params = params
        if self.params.USE_QC:   # if we are using the QC, load embeddings and automorphisms
            data_dir = os.path.join(os.getcwd(), 'data')
            embed_file = 'embeddings_dictionary_graphs_1_through_10_raster_breadth_2_gridsize_6.pkl'
            automorph_file = 'automorphism_dictionary_graphs_1_through_10_max_500000.pkl'
            with open(os.path.join(data_dir, embed_file), "rb") as fp:
                self.embeddings = pickle.load(fp)[self.params.GRAPH_NUMBER]
            with open(os.path.join(data_dir, automorph_file), "rb") as fp:
                self.automorphisms = pickle.load(fp)[self.params.GRAPH_NUMBER]

    def compute_winner_score_and_influence_from_correlation_matrix(self, game_state, correlation_matrix):
        # correlation_matrix is assumed to be symmetric matrix with zeros on diagonal (so that self-correlation of
        # one is not counted) -- this is the standard for computing influence vector

        influence_vector = np.sum(correlation_matrix, axis=0)

        if self.game_state_is_terminal(game_state):
            score_difference = influence_vector[game_state['player1_node']] - influence_vector[game_state['player2_node']]

            if score_difference > self.params.EPSILON:  # more positive than epsilon, red wins
                winner = 'red'
            else:
                if score_difference < -self.params.EPSILON:
                    winner = 'blue'
                else:
                    winner = 'draw'
        else:
            score_difference = None
            winner = None

        return winner, score_difference, influence_vector

    def game_state_is_terminal(self, game_state):
        # a state is terminal if both players have chosen vertices and all edges have been played
        # game_state = {'num_nodes': 6, 'edges': [(0, 1, 1), (0, 2, 1), (0, 3, 2), (0, 4, 3), (0, 5, 2), (1, 2, 1),
        # (1, 3, 2), (1, 4, 3), (1, 5, 3), (2, 3, 1), (2, 4, 2), (2, 5, 3), (3, 4, 2), (3, 5, 1), (4, 5, 2)],
        # 'player1_id': 'player1', 'player2_id': 'player2', 'turn_count': 17, 'current_player_index': 1,
        # 'player1_node': 1, 'player2_node': 3}

        edge_states = [each[2] for each in game_state['edges']]

        if edge_states.count(0) == 0 and game_state['player1_node'] != -1 and game_state['player2_node'] != -1:
            return True
        else:
            return False

    def ss_to_samps(self, ss, num_var, number_of_embeddings_to_use):
        all_samples = ss.record.sample[:, range(num_var)]
        for k in range(1, number_of_embeddings_to_use):
            all_samples = np.vstack((all_samples, ss.record.sample[:, range(num_var * k, num_var * (k + 1))]))
        samps = np.array(all_samples, dtype=float)  # casting may not be necessary.
        return samps

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
        beta_range = [1 / np.sqrt(np.sum([Jij ** 2 for Jij in jay.values()])), beta_max]
        seed = None   # Choose seed=None if reproducibility is not desired

        # randomize_order=True implements standard symmetry-respecting Metropolis algorithm
        ss = sampler.sample_ising(h, jay, beta_range=beta_range, num_reads=self.params.NUM_READS_SA,
                                  num_sweeps=num_sweeps, randomize_order=True, seed=seed)

        samps = np.array(ss.record.sample, dtype=float)  # casting may not be necessary.

        # creates symmetric matrix with zeros on diagonal (so that self-correlation of one is not counted) -- this is
        # the standard for computing influence vector
        correlation_matrix = (np.einsum('si,sj->ij', samps, samps) / self.params.NUM_READS_SA -
                              np.eye(int(game_state['num_nodes'])))

        winner, score_difference, influence_vector = self.compute_winner_score_and_influence_from_correlation_matrix(game_state, correlation_matrix)

        return_dictionary = {'game_state': game_state, 'adjudicator': 'simulated_annealing',
                             'winner': winner, 'score': score_difference, 'influence_vector': influence_vector,
                             'correlation_matrix': correlation_matrix, 'parameters': self.params}

        return return_dictionary

    def schrodinger_equation(self, game_state):

        h, jay = game_state_to_ising_model(game_state)

        s_min = 0.001   # beginning and ending anneal times
        s_max = 0.999

        number_of_levels = 2 ** game_state['num_nodes']

        _, correlation_matrix = (
            evolve_schrodinger(h, jay, s_min=s_min, s_max=s_max, tf=self.params.ANNEAL_TIME_IN_NS,
                               number_of_levels=number_of_levels, n_qubits=game_state['num_nodes'], verbose=False,
                               truncate=True))
        # what's returned here is upper triangular with zeros on the diagonal, so we need to add the transpose
        correlation_matrix = correlation_matrix + correlation_matrix.T

        winner, score_difference, influence_vector = self.compute_winner_score_and_influence_from_correlation_matrix(game_state, correlation_matrix)

        return_dictionary = {'game_state': game_state, 'adjudicator': 'schrodinger_equation',
                             'winner': winner, 'score': score_difference, 'influence_vector': influence_vector,
                             'correlation_matrix': correlation_matrix, 'parameters': self.params}

        return return_dictionary

    def quantum_annealing(self, game_state):

        # todo implement two passes, where first is only one call, second is all of them
        # todo implement self-rolled SRT function

        h, jay = game_state_to_ising_model(game_state)

        # number of automorphisms of the game graph: len(self.automorphisms)
        # number of source to target embeddings found: len(self.embeddings)
        # number of spin reversal transformations used: self.params.SPIN_REVERSAL_TRANSFORMS
        # number of samples taken per call to the processor: self.params.NUM_READS
        # for graph_number 2 ==> these are 6, 346, 10, 100 respectively ==> 2,076,000 samples

        # self.embeddings[0] = [1093, 1098, 136]
        num_var = len(self.embeddings[0])  # e.g. 3
        number_of_automorphisms_P = len(self.automorphisms)  # e.g. 6
        number_of_embeddings_to_use_U = len(self.embeddings)  # e.g. 346

        # big_h and big_j are h, jay values for the entire chip assuming the identity automorphism

        big_h = {}
        big_j = {}

        for k in range(number_of_embeddings_to_use_U):
            for j in range(num_var):
                big_h[num_var * k + j] = 0

        for k, v in jay.items():
            big_j[k] = v
            for j in range(1, number_of_embeddings_to_use_U):
                big_j[(k[0] + num_var * j, k[1] + num_var * j)] = v

        sampler_kwargs = dict(h=big_h,
                              J=big_j,
                              num_reads=self.params.NUM_READS_QC,
                              answer_mode='raw',
                              num_spin_reversal_transforms=self.params.SPIN_REVERSAL_TRANSFORMS)

        if self.params.USE_MOCK_DWAVE_SAMPLER:
            sampler = MockDWaveSampler(topology_type='zephyr', topology_shape=[6, 4])

        else:
            sampler = DWaveSampler(solver='Advantage2_prototype2.4')
            sampler_kwargs.update({'fast_anneal': True,
                                   'annealing_time': self.params.ANNEAL_TIME_IN_NS / 1000})

        # I think this is for making sure the variable order doesn't get screwed up later
        bqm = dimod.BinaryQuadraticModel('SPIN').from_ising(big_h, big_j)

        samps = np.zeros((1, num_var))   # first layer of zeros to get vstack going, will remove at the end

        # for each automorphism, we embed using that automorphism into all the available places on the chip, and
        # collect N samples

        for automorphism_idx in range(number_of_automorphisms_P):    # ranges from 0 to 5

            automorphism_to_use = self.automorphisms[automorphism_idx]    # eg {0:0, 1:2, 2:1}
            permuted_embedding = []
            for each_embedding in self.embeddings:    # each_embedding is like [1093, 1098, 136]
                this_embedding = []
                for each_vertex in range(num_var):    # each_vertex ranges from 0 to 2
                    this_embedding.append(each_embedding[automorphism_to_use[each_vertex]])
                permuted_embedding.append(this_embedding)

            embedding_to_use = {}

            for k in range(number_of_embeddings_to_use_U):
                for j in range(num_var):  # up to 0..1037
                    embedding_to_use[num_var * k + j] = [permuted_embedding[k][j]]

            composed_sampler = SpinReversalTransformCompositeModified(
                FixedEmbeddingComposite(sampler, embedding=embedding_to_use))

            ss = composed_sampler.sample_ising(**sampler_kwargs)
            ss = dimod.SampleSet.from_samples_bqm(ss, bqm)
            new_samps = self.ss_to_samps(ss, num_var, number_of_embeddings_to_use_U)
            samps = np.vstack((samps, new_samps))   # stack new_samps from this automorphism

        samps = np.delete(samps, (0), axis=0)   # delete first row of zeros

        sample_count = self.params.NUM_READS_QC * self.params.SPIN_REVERSAL_TRANSFORMS * number_of_embeddings_to_use_U * number_of_automorphisms_P

        # this is a full matrix with zeros on the diagonal that uses all the samples
        correlation_matrix = \
            (np.einsum('si,sj->ij', samps, samps) / sample_count -
             np.eye(int(game_state['num_nodes'])))

        winner, score_difference, influence_vector = self.compute_winner_score_and_influence_from_correlation_matrix(game_state, correlation_matrix)

        return_dictionary = {'game_state': game_state, 'adjudicator': 'quantum_annealing',
                             'winner': winner, 'score': score_difference, 'influence_vector': influence_vector,
                             'correlation_matrix': correlation_matrix, 'parameters': self.params}

        return return_dictionary
