""" evaluation of Tangled terminal states """

import sys
import os
import time
import numpy as np
from adjudicators.adjudicate import Adjudicator
from utils.parameters import Params


def main():
    # this code shows how to use the three different adjudicators
    # there are two example_game_state dictionaries provided, which are terminal states in graph_number 2 and 3
    # respectively, that are of the sort that are closest to the draw line at score = +- 1/2

    precision_digits = 4    # just to clean up print output

    params = Params()
    adjudicator = Adjudicator(params)

    example_game_state = None

    # blue wins, score -2/3; this is one of the states closest to the draw line
    if params.GRAPH_NUMBER == 2:
        example_game_state = {'num_nodes': 3, 'edges': [(0, 1, 3), (0, 2, 2), (1, 2, 2)],
                              'player1_id': 'player1', 'player2_id': 'player2', 'turn_count': 5,
                              'current_player_index': 1, 'player1_node': 1, 'player2_node': 2}

    # red wins, score +2/3; this is one of the states closest to the draw line
    if params.GRAPH_NUMBER == 3:
        example_game_state = {'num_nodes': 4, 'edges': [(0, 1, 3), (0, 2, 1), (0, 3, 3),
                                                        (1, 2, 1), (1, 3, 3), (2, 3, 1)],
                              'player1_id': 'player1', 'player2_id': 'player2', 'turn_count': 8,
                              'current_player_index': 2, 'player1_node': 2, 'player2_node': 3}

    #########################################################################
    # first, simulated annealing...

    start = time.time()

    w, score, influence = adjudicator.simulated_annealing(example_game_state)

    print('elapsed time for simulated annealing was', round(time.time() - start, precision_digits), 'seconds.')
    print('winner:', w)
    if score is None:
        print('score:', score)
    else:
        print('score:', round(score, precision_digits))
    print('influence vector:', [round(influence[k], precision_digits) for k in range(len(influence))])
    print()
    #########################################################################

    #########################################################################
    # second, schrodinger equation...

    start = time.time()

    w, score, influence = adjudicator.schrodinger_equation(example_game_state)

    print('elapsed time for schrodinger equation was', round(time.time() - start, precision_digits), 'seconds.')
    print('winner:', w)
    if score is None:
        print('score:', score)
    else:
        print('score:', round(score, precision_digits))
    print('influence vector:', [round(influence[k], precision_digits) for k in range(len(influence))])
    print()
    #########################################################################

    #########################################################################
    # third, quantum annealing...

    start = time.time()

    w, score, influence = adjudicator.quantum_annealing(example_game_state)

    print('elapsed time for quantum annealing was', round(time.time() - start, precision_digits), 'seconds.')
    print('winner:', w)
    if score is None:
        print('score:', score)
    else:
        print('score:', round(score, precision_digits))
    print('influence vector:', [round(influence[k], precision_digits) for k in range(len(influence))])
    print()
    #########################################################################


if __name__ == "__main__":
    sys.exit(main())
