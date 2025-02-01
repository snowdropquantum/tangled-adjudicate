""" generate and adjudicate all Tangled terminal states for tiny graphs """
import sys
import os
import time
import pickle
import numpy as np

from tangled_adjudicate.adjudicators.simulated_annealing import SimulatedAnnealingAdjudicator
from tangled_adjudicate.adjudicators.quantum_annealing import QuantumAnnealingAdjudicator
from tangled_adjudicate.adjudicators.lookup_table import LookupTableAdjudicator
from tangled_adjudicate.adjudicators.schrodinger import SchrodingerEquationAdjudicator

from tangled_adjudicate.utils.generate_terminal_states import generate_all_tangled_terminal_states


def generate_adjudication_results_for_all_terminal_states(graph_number, solver_to_use):
    # uses up to three different adjudicators provided to evaluate all unique terminal states for tiny graphs
    # (in the default here, graphs 2 and 3). Note this only works for tiny graphs as the number of terminal states
    # grows like 3 ** edge_count.
    # solver_to_use is a string, one of 'simulated_annealing', 'schrodinger_equation', or 'quantum_annealing'
    # the results are stored in a dictionary whose keys are the solvers. When you call this function using a solver
    # that hasn't been called yet, it adds that key and its results. If you call it in a case where there are
    # already results, it will ask you if you want to overwrite them.

    if solver_to_use not in ['simulated_annealing', 'schrodinger_equation', 'quantum_annealing', 'lookup_table']:
        sys.exit(print('the solver' + solver_to_use + 'is not in the allowed list -- please take a look!'))

    precision_digits = 4                        # just to clean up print output
    np.set_printoptions(suppress=True)          # remove scientific notation

    adjudicator = None

    args = {'data_dir': os.path.join(os.getcwd(), '..', 'data'),
            'graph_number': graph_number}

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

    game_states = generate_all_tangled_terminal_states(graph_number)

    data_dir = os.path.join(os.getcwd(), '..', 'data')
    file_path = os.path.join(data_dir, "graph_" + str(graph_number) + "_terminal_states_adjudication_results.pkl")

    if os.path.isfile(file_path):
        with open(file_path, "rb") as fp:
            adjudication_results = pickle.load(fp)
    else:   # in this case, there are no results at all and we start fresh
        adjudication_results = {}

    # at this point, either we have loaded some adjudication_results from an existing file, or we have a new empty dict
    if solver_to_use in adjudication_results:       # this means we loaded this in already
        user_input = input('results already exist for ' + solver_to_use + ', overwrite (y/n)?')
        if user_input.lower() != 'y':
            return None

    # now we proceed to compute and store result
    print('beginning adjudication using the ' + solver_to_use + ' solver...')
    start = time.time()
    adjudication_results[solver_to_use] = {}

    for k, v in game_states.items():
        adjudication_results[solver_to_use][k] = adjudicator.adjudicate(v['game_state'])

    print('elapsed time was', round(time.time() - start, precision_digits), 'seconds.')

    # store it -- this should leave any previously loaded solver results intact
    with open(file_path, "wb") as fp:
        pickle.dump(adjudication_results, fp)


def main():

    # note: generating all schrodinger_equation adjudication results for graph 3 or bigger takes forever
    # I spot checked new subclass version and all spot checks were good

    graph_number = 2
    solver_list = ['simulated_annealing', 'schrodinger_equation', 'quantum_annealing', 'lookup_table']

    for solver_to_use in solver_list:
        generate_adjudication_results_for_all_terminal_states(graph_number, solver_to_use)


if __name__ == "__main__":
    sys.exit(main())
