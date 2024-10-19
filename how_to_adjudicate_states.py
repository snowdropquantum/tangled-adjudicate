""" how to use provided solvers to adjudicate Tangled terminal states """
import pickle
import sys
import os
import ast
import time
import numpy as np
from adjudicators.adjudicate import Adjudicator
from utils.parameters import Params
from utils.game_graph_properties import GraphProperties
from utils.generate_terminal_states import convert_state_string_to_game_state


def main():
    # this code shows how to use the three different adjudicators
    # there are two example_game_state dictionaries provided, which are terminal states in graph_number 2 and 3
    # respectively, that are of the sort that are closest to the draw line at score = +- 1/2

    precision_digits = 4    # just to clean up print output
    np.set_printoptions(suppress=True)   # remove scientific notation

    params = Params()
    graph = GraphProperties(params.GRAPH_NUMBER)
    adjudicator = Adjudicator(params)

    example_game_state = None

    # blue wins, score -2/3; this is one of the states closest to the draw line
    if params.GRAPH_NUMBER == 2:
        example_game_state = {'num_nodes': 3, 'edges': [(0, 1, 3), (0, 2, 2), (1, 2, 2)],
                              'player1_id': 'player1', 'player2_id': 'player2', 'turn_count': 5,
                              'current_player_index': 1, 'player1_node': 1, 'player2_node': 2}

    # # red wins, score +2/3; this is one of the states closest to the draw line
    # if params.GRAPH_NUMBER == 3:
    #     example_game_state = {'num_nodes': 4, 'edges': [(0, 1, 3), (0, 2, 1), (0, 3, 3),
    #                                                     (1, 2, 1), (1, 3, 3), (2, 3, 1)],
    #                           'player1_id': 'player1', 'player2_id': 'player2', 'turn_count': 8,
    #                           'current_player_index': 2, 'player1_node': 2, 'player2_node': 3}

    # draw, score 0; this is a test state where the mock D-Wave solver returns -0.500106 (ie very wrong)
    # if params.GRAPH_NUMBER == 3:
    #     example_game_state = {'num_nodes': 4, 'edges': [(0, 1, 1), (0, 2, 1), (0, 3, 2),
    #                                                     (1, 2, 3), (1, 3, 1), (2, 3, 2)],
    #                           'player1_id': 'player1', 'player2_id': 'player2', 'turn_count': 8,
    #                           'current_player_index': 2, 'player1_node': 2, 'player2_node': 3}

    if params.GRAPH_NUMBER == 3:
        data_dir = os.path.join(os.getcwd(), 'data')
        with open(os.path.join(data_dir, 'disagree_keys.pkl'), "rb") as fp:
            disagree_keys = pickle.load(fp)
        game_state = {}
        for each in disagree_keys:
            game_state[each] = convert_state_string_to_game_state(graph, ast.literal_eval(each))
        # example_game_state = {'num_nodes': 4, 'edges': [(0, 1, 3), (0, 2, 3), (0, 3, 1),
        #                                                 (1, 2, 3), (1, 3, 2), (2, 3, 2)],
        #                       'player1_id': 'player1', 'player2_id': 'player2', 'turn_count': 8,
        #                       'current_player_index': 2, 'player1_node': 2, 'player2_node': 3}

        for k, v in game_state.items():
            #########################################################################
            # first, simulated annealing...

            start = time.time()

            results = adjudicator.simulated_annealing(v)

            print('elapsed time for simulated annealing was', round(time.time() - start, precision_digits), 'seconds.')
            print('correlation matrix:')
            print(np.round(results['correlation_matrix'], precision_digits))
            print('winner:', results['winner'])
            if results['score'] is None:
                print('score:', results['score'])
            else:
                print('score:', round(results['score'], precision_digits))
            print('influence vector:', [round(results['influence_vector'][k], precision_digits) for k in range(len(results['influence_vector']))])
            print()
            #########################################################################

            #########################################################################
            # second, schrodinger equation...

            start = time.time()

            results = adjudicator.schrodinger_equation(v)

            print('elapsed time for schrodinger equation was', round(time.time() - start, precision_digits), 'seconds.')
            print('correlation matrix:')
            print(np.round(results['correlation_matrix'], precision_digits))
            print('winner:', results['winner'])
            if results['score'] is None:
                print('score:', results['score'])
            else:
                print('score:', round(results['score'], precision_digits))
            print('influence vector:', [round(results['influence_vector'][k], precision_digits) for k in range(len(results['influence_vector']))])
            print()
            #########################################################################

            #########################################################################
            # third, quantum annealing...

            start = time.time()

            results = adjudicator.quantum_annealing(v)

            print('elapsed time for quantum annealing was', round(time.time() - start, precision_digits), 'seconds.')
            print('correlation matrix:')
            print(np.round(results['correlation_matrix'], precision_digits))
            print('winner:', results['winner'])
            if results['score'] is None:
                print('score:', results['score'])
            else:
                print('score:', round(results['score'], precision_digits))
            print('influence vector:', [round(results['influence_vector'][k], precision_digits) for k in range(len(results['influence_vector']))])
            print()
            #########################################################################


if __name__ == "__main__":
    sys.exit(main())
