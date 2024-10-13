""" evaluation of Tangled terminal states using simulated annealing """
import neal
import sys
import time
import numpy as np
from utils.utilities import game_state_to_ising_model


class Adjudicator(object):
    def __init__(self, params):
        self.params = params

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
        ss = sampler.sample_ising(h, jay, beta_range=beta_range, num_reads=self.params.NUM_READS,
                                  num_sweeps=num_sweeps, randomize_order=True, seed=seed)

        samps = np.array(ss.record.sample, dtype=float)  # casting may not be necessary.

        # creates symmetric matrix with zeros on diagonal (so that self-correlation of one is not counted) -- this is
        # the standard for computing influence vector
        correlation_matrix = (np.einsum('si,sj->ij', samps, samps) / self.params.NUM_READS -
                              np.eye(int(game_state['num_nodes'])))

        influence_vector = np.sum(correlation_matrix, axis=0)

        score_difference = influence_vector[game_state['player1_node']] - influence_vector[game_state['player2_node']]

        if score_difference > self.params.EPSILON:  # more positive than epsilon, red wins
            winner = 'red'
        else:
            if score_difference < -self.params.EPSILON:
                winner = 'blue'
            else:
                winner = 'draw'

        return winner, score_difference, influence_vector
