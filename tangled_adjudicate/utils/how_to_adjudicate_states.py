""" how to use provided solvers to adjudicate Tangled terminal states """
import sys
import os
import time
import numpy as np

from tangled_adjudicate.adjudicators.simulated_annealing import SimulatedAnnealingAdjudicator
from tangled_adjudicate.adjudicators.quantum_annealing import QuantumAnnealingAdjudicator
from tangled_adjudicate.adjudicators.lookup_table import LookupTableAdjudicator
from tangled_adjudicate.adjudicators.schrodinger import SchrodingerEquationAdjudicator


def main():
    # this code shows how to use the four different adjudicators
    # there are two example_game_state dictionaries provided, which are terminal states in graph_number 2 and 3
    # respectively, that are of the sort that are closest to the draw line at score = +- 1/2

    # set graph_number
    graph_number = 2

    # choose solvers to use
    solver_list = ['simulated_annealing', 'schrodinger_equation', 'quantum_annealing', 'lookup_table']

    precision_digits = 4    # just to clean up print output
    np.set_printoptions(suppress=True)   # remove scientific notation

    args = {'data_dir': os.path.join(os.getcwd(), '..', 'data'),
            'graph_number': graph_number}

    example_game_state = None

    # draw; score=0; ferromagnetic ring
    if graph_number == 2:
        example_game_state = {'num_nodes': 3, 'edges': [(0, 1, 2), (0, 2, 2), (1, 2, 2)],
                              'player1_id': 'player1', 'player2_id': 'player2', 'turn_count': 5,
                              'current_player_index': 1, 'player1_node': 1, 'player2_node': 2}
    else:
        # red wins, score +2/3; this is one of the states closest to the draw line
        # note that quantum_annealing in this default uses the D-Wave mock software solver and won't give
        # the right answer as its samples aren't unbiased -- if you want the quantum_annealing solver to
        # run on hardware set QAParameters.use_mock = False in /adjudicators/quantum_annealing.py and ensure you have
        # hardware access and everything is set up

        if graph_number == 3:
            example_game_state = {'num_nodes': 4, 'edges': [(0, 1, 3), (0, 2, 1), (0, 3, 3),
                                                            (1, 2, 1), (1, 3, 3), (2, 3, 1)],
                                  'player1_id': 'player1', 'player2_id': 'player2', 'turn_count': 8,
                                  'current_player_index': 2, 'player1_node': 2, 'player2_node': 3}
        else:
            print('this introduction only has included game states for graphs 2 and 3. If you want a different'
                  'graph please add a new example_game_state here!')

    for solver_to_use in solver_list:

        start = time.time()

        adjudicator = None

        if solver_to_use == 'simulated_annealing':
            adjudicator = SimulatedAnnealingAdjudicator()
        else:
            if solver_to_use == 'quantum_annealing':
                adjudicator = QuantumAnnealingAdjudicator()
            else:
                if solver_to_use == 'lookup_table':
                    adjudicator = LookupTableAdjudicator()
                else:
                    if solver_to_use == 'schrodinger_equation':
                        adjudicator = SchrodingerEquationAdjudicator()

        adjudicator.setup(**args)
        results = adjudicator.adjudicate(example_game_state)

        print('elapsed time for', solver_to_use, 'was', round(time.time() - start, precision_digits), 'seconds.')

        if results['correlation_matrix'] is None:
            print('correlation matrix:', None)
        else:
            print('correlation matrix:')
            print(np.round(results['correlation_matrix'], precision_digits))

        print('winner:', results['winner'])

        if results['score'] is None:
            print('score:', results['score'])
        else:
            print('score:', round(results['score'], precision_digits))

        if results['influence_vector'] is None:
            print('influence vector:', None)
        else:
            print('influence vector:', [round(results['influence_vector'][k], precision_digits)
                                        for k in range(len(results['influence_vector']))])


if __name__ == "__main__":
    sys.exit(main())
