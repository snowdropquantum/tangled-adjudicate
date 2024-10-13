""" evaluation of Tangled terminal states """

import sys
import os
import time
import numpy as np
from adjudicators.adjudicate import Adjudicator
from utils.parameters import Params


def main():
    # game_state = {'num_nodes': 6, 'edges': [(0, 1, 1), (0, 2, 1), (0, 3, 2), (0, 4, 3), (0, 5, 2), (1, 2, 1),
    #                                         (1, 3, 2), (1, 4, 3), (1, 5, 3), (2, 3, 1), (2, 4, 2), (2, 5, 3),
    #                                         (3, 4, 2), (3, 5, 1), (4, 5, 2)],
    #               'player1_id': 'player1', 'player2_id': 'player2', 'turn_count': 17, 'current_player_index': 1,
    #               'player1_node': 1, 'player2_node': 3}
    # example_game_state = {'num_nodes': 4, 'edges': [(0, 1, 0), (0, 2, 3), (0, 3, 2), (1, 2, 0), (1, 3, 2), (2, 3, 3)],
    #                       'player1_id': 'player1', 'player2_id': 'player2', 'turn_count': 8, 'current_player_index': 2,
    #                       'player1_node': 2, 'player2_node': 3}

    params = Params()
    adjudicator = Adjudicator(params)

    example_game_state = {'num_nodes': 3, 'edges': [(0, 1, 3), (0, 2, 2), (1, 2, 2)],
                          'player1_id': 'player1', 'player2_id': 'player2', 'turn_count': 5, 'current_player_index': 1,
                          'player1_node': 1, 'player2_node': 2}

    start = time.time()

    w, score, influence = adjudicator.simulated_annealing(example_game_state)

    print('elapsed time was', time.time() - start, 'seconds.')
    print('winner:', w)
    print('score:', score)
    print('influence vector:', influence)


if __name__ == "__main__":
    sys.exit(main())
