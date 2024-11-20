""" how to use provided solvers to adjudicate Tangled terminal states """
import pickle
import sys
import os
import ast
import time
import numpy as np

from tangled_adjudicate.adjudicators.adjudicate import Adjudicator
from tangled_adjudicate.utils.parameters import Params
from tangled_adjudicate.utils.game_graph_properties import GraphProperties
from tangled_adjudicate.utils.generate_terminal_states import convert_state_string_to_game_state


def main():
    # this code shows how to use the three different adjudicators
    # there are two example_game_state dictionaries provided, which are terminal states in graph_number 2 and 3
    # respectively, that are of the sort that are closest to the draw line at score = +- 1/2

    solver_list = ['simulated_annealing', 'schrodinger_equation', 'quantum_annealing']

    precision_digits = 4    # just to clean up print output
    np.set_printoptions(suppress=True)   # remove scientific notation

    params = Params()
    adjudicator = Adjudicator(params)

    example_game_state = None

    # draw; score=0; ferromagnetic ring
    if params.GRAPH_NUMBER == 2:
        example_game_state = {'num_nodes': 3, 'edges': [(0, 1, 2), (0, 2, 2), (1, 2, 2)],
                              'player1_id': 'player1', 'player2_id': 'player2', 'turn_count': 5,
                              'current_player_index': 1, 'player1_node': 1, 'player2_node': 2}
    else:
        # red wins, score +2/3; this is one of the states closest to the draw line
        # note that quantum_annealing in this default uses the D-Wave mock software solver and won't give
        # the right answer as its samples aren't unbiased -- if you want the quantum_annealing solver to
        # run on hardware set self.USE_MOCK_DWAVE_SAMPLER = False in /utils/parameters.py and ensure you have
        # hardware access and everything is set up

        if params.GRAPH_NUMBER == 3:
            example_game_state = {'num_nodes': 4, 'edges': [(0, 1, 3), (0, 2, 1), (0, 3, 3),
                                                            (1, 2, 1), (1, 3, 3), (2, 3, 1)],
                                  'player1_id': 'player1', 'player2_id': 'player2', 'turn_count': 8,
                                  'current_player_index': 2, 'player1_node': 2, 'player2_node': 3}
        else:
            print('this introduction only has included game states for graphs 2 and 3. If you want a different'
                  'graph please add a new example_game_state here!')

    for solver_to_use in solver_list:

        start = time.time()

        # equivalent to e.g. results = adjudicator.simulated_annealing(example_game_state)
        results = getattr(adjudicator, solver_to_use)(example_game_state)

        print('elapsed time for', solver_to_use, 'was', round(time.time() - start, precision_digits), 'seconds.')
        print('correlation matrix:')
        print(np.round(results['correlation_matrix'], precision_digits))
        print('winner:', results['winner'])
        if results['score'] is None:
            print('score:', results['score'])
        else:
            print('score:', round(results['score'], precision_digits))
        print('influence vector:', [round(results['influence_vector'][k], precision_digits)
                                    for k in range(len(results['influence_vector']))])
        print()


if __name__ == "__main__":
    sys.exit(main())
