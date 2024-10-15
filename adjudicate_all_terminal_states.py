""" generate and adjudicate all Tangled terminal states for tiny graphs """

import sys
import os
import time
import pickle
import numpy as np
from adjudicators.adjudicate import Adjudicator
from utils.generate_terminal_states import write_unique_states_to_disk
from utils.parameters import Params


def main():
    # this code uses the three different adjudicators provided to evaluate all unique terminal states for tiny
    # graphs (in this case, graphs 2 and 3). Run /utils/generate_terminal_states.py first for the graph you want
    # to use (also you will need the automorphisms of the graph, which you can get by running
    # /utils/find_graph_automorphisms.py -- I've included the ones for graphs 2 and 3, you can generate for different
    # graphs if you want, but note this only works for tiny graphs as the number of terminal states grows like
    # 3 ** edge_count

    precision_digits = 4    # just to clean up print output
    np.set_printoptions(suppress=True)   # remove scientific notation

    params = Params()
    adjudicator = Adjudicator(params)

    file_name_prefix = "graph_" + str(params.GRAPH_NUMBER)

    data_dir = os.path.join(os.getcwd(), 'data')

    my_file = os.path.join(data_dir, file_name_prefix + "_unique_terminal_states_adjudication_results.pkl")

    if os.path.isfile(my_file):
        with open(my_file, "rb") as fp:
            results = pickle.load(fp)
    else:
        with open(os.path.join(data_dir, file_name_prefix + '_unique_terminal_states.pkl'), "rb") as fp:
            game_states = pickle.load(fp)

        results = {}

        for k, v in game_states.items():
            results[k] = {}
            results[k]['simulated_annealing'] = []
            results[k]['schrodinger_equation'] = []
            results[k]['quantum_annealing'] = []

        start = time.time()
        for k, v in game_states.items():
            results[k]['simulated_annealing'].append(adjudicator.simulated_annealing(v['game_state']))
        print('elapsed time for simulated annealing was', round(time.time() - start, precision_digits), 'seconds.')

        start = time.time()
        for k, v in game_states.items():
            results[k]['schrodinger_equation'].append(adjudicator.schrodinger_equation(v['game_state']))
        print('elapsed time for schrodinger equation was', round(time.time() - start, precision_digits), 'seconds.')

        start = time.time()
        for k, v in game_states.items():
            results[k]['quantum_annealing'].append(adjudicator.quantum_annealing(v['game_state']))
        print('elapsed time for quantum annealing was', round(time.time() - start, precision_digits), 'seconds.')

        with open(os.path.join(data_dir, file_name_prefix + "_unique_terminal_states_adjudication_results.pkl"), "wb") as fp:
            pickle.dump(results, fp)

    print()


if __name__ == "__main__":
    sys.exit(main())
