""" how to use provided solvers to adjudicate Tangled terminal states """
import sys
import os
import time
import numpy as np

from tangled_adjudicate.utils.game_graph_properties import GraphProperties

from tangled_adjudicate.adjudicators.simulated_annealing import SimulatedAnnealingAdjudicator
from tangled_adjudicate.adjudicators.quantum_annealing import QuantumAnnealingAdjudicator
from tangled_adjudicate.adjudicators.lookup_table import LookupTableAdjudicator
from tangled_adjudicate.adjudicators.schrodinger import SchrodingerEquationAdjudicator


def main():
    # this code shows how to use the four different adjudicators

    # set graph_number
    graph_number = 11   # 11 is P_3, 2 is K_3, 20 is diamond graph, 19 is barbell graph

    # get graph properties
    graph = GraphProperties(graph_number=graph_number)

    # choose solver(s) to use
    # solver_list = ['simulated_annealing', 'schrodinger_equation', 'lookup_table', 'quantum_annealing']
    solver_list = ['simulated_annealing', 'schrodinger_equation', 'lookup_table']

    precision_digits = 4    # just to clean up print output
    np.set_printoptions(suppress=True)   # remove scientific notation

    args = {'data_dir': os.path.join(os.getcwd(), '..', 'data'),
            'graph_number': graph_number,
            'lookup_args': {'solver': 'quantum_annealing',   # one of 'simulated_annealing', 'schrodinger_equation', 'quantum_annealing'; must have already been created to use this solver
                            'epsilon': 0.5,
                            'anneal_time': 350,
                            'num_reads': 100000}}
    example_game_state = None

    # these are in the order introduced in the Phase 1 paper; yes, I realize my numbering scheme is ludicrous

    if graph_number == 11:  # P_3 graph, S = [2,3], score = +2 (red)
        example_game_state = {'num_nodes': 3, 'edges': [(0, 1, 2), (1, 2, 3)],
                              'player1_id': 'player1', 'player2_id': 'player2', 'turn_count': 4,
                              'current_player_index': 1, 'player1_node': graph.vertex_ownership[graph_number][0],
                              'player2_node': graph.vertex_ownership[graph_number][1]}

    if graph_number == 2:   # K_3 graph, S = [3,3,3], score = 0 (draw)
        example_game_state = {'num_nodes': 3, 'edges': [(0, 1, 3), (0, 2, 3), (1, 2, 3)],
                              'player1_id': 'player1', 'player2_id': 'player2', 'turn_count': 5,
                              'current_player_index': 1, 'player1_node': graph.vertex_ownership[graph_number][0],
                              'player2_node': graph.vertex_ownership[graph_number][1]}

    if graph_number == 20:  # Diamond graph; S = [1,2,2,2,3]; SA scores +4/3, QA & SE scores = +2 (red)
        example_game_state = {'num_nodes': 4, 'edges': [(0, 1, 1), (0, 3, 2),
                                                        (1, 2, 2), (1, 3, 2), (2, 3, 3)],
                              'player1_id': 'player1', 'player2_id': 'player2', 'turn_count': 8,
                              'current_player_index': 2, 'player1_node': graph.vertex_ownership[graph_number][0],
                              'player2_node': graph.vertex_ownership[graph_number][1]}

    if graph_number == 19:   # Barbell graph; S =[3,2,2,3,2,3,2]; SA scores -4/9, QA & SE scores +1/2
        example_game_state = {'num_nodes': 6, 'edges': [(0, 1, 3), (0, 2, 2), (1, 2, 2),
                                                        (2, 5, 3), (3, 4, 2), (3, 5, 3),
                                                        (4, 5, 2)],
                              'player1_id': 'player1', 'player2_id': 'player2', 'turn_count': 9,
                              'current_player_index': 1, 'player1_node': graph.vertex_ownership[graph_number][0],
                              'player2_node': graph.vertex_ownership[graph_number][1]}

    if example_game_state is None:
        sys.exit(print('this introduction only includes game states for graphs 2, 11, 19, and 20. '
                       'If you want a different graph please add a new example_game_state here!'))

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
