""" how to use provided solvers to adjudicate Tangled terminal states """
import pickle
import sys
import os
import ast
import time
import numpy as np

from tangled_adjudicate.adjudicators.adjudicate import old_Adjudicator

from tangled_adjudicate.adjudicators.simulated_annealing import SimulatedAnnealingAdjudicator
from tangled_adjudicate.adjudicators.quantum_annealing import QuantumAnnealingAdjudicator
from tangled_adjudicate.adjudicators.lookup_table import LookupTableAdjudicator
from tangled_adjudicate.adjudicators.schrodinger import SchrodingerEquationAdjudicator

from tangled_adjudicate.utils.parameters import Params
from tangled_adjudicate.utils.game_graph_properties import GraphProperties
from tangled_adjudicate.utils.generate_terminal_states import convert_state_string_to_game_state


def main():
    # this code shows how to use the four different adjudicators
    # there are two example_game_state dictionaries provided, which are terminal states in graph_number 2 and 3
    # respectively, that are of the sort that are closest to the draw line at score = +- 1/2

    # solver_list = ['simulated_annealing', 'schrodinger_equation', 'quantum_annealing', 'lookup_table']
    solver_list = ['quantum_annealing']

    precision_digits = 4    # just to clean up print output
    np.set_printoptions(suppress=True)   # remove scientific notation

    params = Params()
    old_adjudicator = old_Adjudicator(params)

    args = {'data_dir': os.path.join(os.getcwd(), '..', 'data'),
            'graph_number': params.GRAPH_NUMBER,
            'solver_name': params.QC_SOLVER_TO_USE}

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

    # if 'simulated_annealing' in solver_list:
    #     sa_adjudicator = SimulatedAnnealingAdjudicator()
    #     sa_adjudicator.setup()
    #     start = time.time()
    #     new_sa_results = sa_adjudicator.adjudicate(example_game_state)
    #     print('elapsed time for simulated_annealing was', round(time.time() - start, precision_digits), 'seconds.')
    #
    # if 'quantum_annealing' in solver_list:
    #     qa_adjudicator = QuantumAnnealingAdjudicator()
    #     qa_adjudicator.setup(**args)
    #     new_qa_results = qa_adjudicator.adjudicate(example_game_state)
    #
    # if 'lookup_table' in solver_list:
    #     lt_adjudicator = LookupTableAdjudicator()
    #     lt_adjudicator.setup(**args)
    #     new_lt_results = lt_adjudicator.adjudicate(example_game_state)
    #
    # if 'schrodinger_equation' in solver_list:
    #     se_adjudicator = SchrodingerEquationAdjudicator()
    #     se_adjudicator.setup()
    #     new_se_results = se_adjudicator.adjudicate(example_game_state)

    for solver_to_use in solver_list:

        start = time.time()

        # equivalent to e.g. results = adjudicator.simulated_annealing(example_game_state)
        old_results = getattr(old_adjudicator, solver_to_use)(example_game_state)

        print('elapsed time for old', solver_to_use, 'was', round(time.time() - start, precision_digits), 'seconds.')

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
        new_results = adjudicator.adjudicate(example_game_state)

        print('elapsed time for new', solver_to_use, 'was', round(time.time() - start, precision_digits), 'seconds.')

        if old_results['correlation_matrix'] is None:
            print('old correlation matrix:', None)
        else:
            print('old correlation matrix:')
            print(np.round(old_results['correlation_matrix'], precision_digits))

        if new_results['correlation_matrix'] is None:
            print('new correlation matrix:', None)
        else:
            print('new correlation matrix:')
            print(np.round(new_results['correlation_matrix'], precision_digits))

        print('old winner:', old_results['winner'])
        print('new winner:', new_results['winner'])

        if old_results['score'] is None:
            print('old score:', old_results['score'])
        else:
            print('old score:', round(old_results['score'], precision_digits))

        if new_results['score'] is None:
            print('new score:', new_results['score'])
        else:
            print('new score:', round(new_results['score'], precision_digits))

        if old_results['influence_vector'] is None:
            print('old influence vector:', None)
        else:
            print('old influence vector:', [round(old_results['influence_vector'][k], precision_digits)
                                        for k in range(len(old_results['influence_vector']))])

        if new_results['influence_vector'] is None:
            print('new influence vector:', None)
        else:
            print('new influence vector:', [round(new_results['influence_vector'][k], precision_digits)
                                        for k in range(len(new_results['influence_vector']))])

        print()
    print()


if __name__ == "__main__":
    sys.exit(main())
