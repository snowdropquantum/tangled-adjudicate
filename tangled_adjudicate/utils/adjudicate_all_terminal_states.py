""" generate and adjudicate all Tangled terminal states for tiny graphs """

import sys
import os
import time
import pickle
import numpy as np

from tangled_adjudicate.adjudicators.adjudicate import Adjudicator
from tangled_adjudicate.utils.generate_terminal_states import generate_all_tangled_terminal_states
from tangled_adjudicate.utils.parameters import Params


def generate_adjudication_results_for_all_terminal_states(solver_to_use):
    # uses up to three different adjudicators provided to evaluate all unique terminal states for tiny graphs
    # (in the default here, graphs 2 and 3). Note this only works for tiny graphs as the number of terminal states
    # grows like 3 ** edge_count.
    # solver_to_use is a string, one of 'simulated_annealing', 'schrodinger_equation', or 'quantum_annealing'
    # the results are stored in a dictionary whose keys are the solvers. When you call this function using a solver
    # that hasn't been called yet, it adds that key and its results. If you call it in a case where there are
    # already results, it will ask you if you want to overwrite them.

    if solver_to_use not in ['simulated_annealing', 'schrodinger_equation', 'quantum_annealing']:
        sys.exit(print('the solver' + solver_to_use + 'is not in the allowed list -- please take a look!'))

    precision_digits = 4                        # just to clean up print output
    np.set_printoptions(suppress=True)          # remove scientific notation

    params = Params()                           # your graph_number will be set here, make sure it's what you want!
    adjudicator = Adjudicator(params)
    game_states = generate_all_tangled_terminal_states(params.GRAPH_NUMBER)

    file_name_prefix = "graph_" + str(params.GRAPH_NUMBER)
    data_dir = os.path.join(os.getcwd(), '..', 'data')
    file_path = os.path.join(data_dir, file_name_prefix + "_terminal_states_adjudication_results.pkl")

    if os.path.isfile(file_path):
        with open(file_path, "rb") as fp:
            adjudication_results = pickle.load(fp)
    else:   # in this case, there are no results at all and we start fresh
        adjudication_results = {}

    # at this point, either we have loaded some adjudication_results from an existing file, or we have a new empty dict
    if solver_to_use in adjudication_results:       # this means we loaded this in already
        user_input = input('results already exist, overwrite (y/n)?')
        if user_input.lower() != 'y':
            sys.exit(print('exiting!'))

    # now we proceed to compute and store result
    print('beginning adjudication using the ' + solver_to_use + ' solver...')
    start = time.time()
    adjudication_results[solver_to_use] = {}
    for k, v in game_states.items():
        adjudication_results[solver_to_use][k] = getattr(adjudicator, solver_to_use)(v['game_state'])
    print('elapsed time was', round(time.time() - start, precision_digits), 'seconds.')

    # store it -- this should leave any previously loaded solver results intact
    with open(file_path, "wb") as fp:
        pickle.dump(adjudication_results, fp)


def main():

    solver_list = ['simulated_annealing', 'schrodinger_equation', 'quantum_annealing']

    for solver_to_use in solver_list:
        generate_adjudication_results_for_all_terminal_states(solver_to_use)


if __name__ == "__main__":
    sys.exit(main())
