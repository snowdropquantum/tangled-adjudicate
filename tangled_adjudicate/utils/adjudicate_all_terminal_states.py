""" adjudicate all Tangled terminal states for small graphs """
import sys
import os
import time
import pickle
import cProfile
import pstats
from concurrent.futures.process import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
from tqdm import tqdm

from tangled_adjudicate.adjudicators.simulated_annealing import SimulatedAnnealingAdjudicator
from tangled_adjudicate.adjudicators.quantum_annealing import QuantumAnnealingAdjudicator
from tangled_adjudicate.adjudicators.schrodinger import SchrodingerEquationAdjudicator

from tangled_adjudicate.utils.game_graph_properties import GraphProperties
from tangled_adjudicate.utils.utilities import convert_my_game_state_to_erik_game_state


def safe_nested_access_and_create(d, keys, default_value=None):
    """
    Safely access a nested dictionary and create missing keys.

    Args:
        d: The dictionary to access/modify
        keys: List of keys to navigate through
        default_value: Value to set if the full path doesn't exist

    Returns:
        The value at the nested location, or default_value if created
    """
    if not keys:
        return d

    current = d
    for i, key in enumerate(keys[:-1]):
        # If we're not at the final key, we need to ensure the path exists
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]

    # For the final key, get or set the value
    last_key = keys[-1]
    if last_key not in current:
        current[last_key] = default_value

    return current[last_key]


def parallel_generate_adjudication_results(graph_number, solver_to_use, params, variant='fixed'):
    ############################################################################################################
    # output file is
    # os.path.join(data_dir, "graph_" + str(graph_number) + "_adjudicated_unique_terminal_states_for_paper.pkl")
    ############################################################################################################
    #
    # it stores a dict adjudication_results[first_key]['simulated_annealing'][num_reads] = results
    #                  adjudication_results[first_key]['schrodinger_equation'][anneal_time] = results
    #                  adjudication_results[first_key]['quantum_annealing'][num_reads][anneal_time] = results
    # first_key is one of 'fixed', 'free'
    # num_reads, anneal_time are ints

    # solver_to_use is a string, one of 'simulated_annealing', 'schrodinger_equation', 'quantum_annealing'.
    # When you call this function using a solver / parameter that hasn't been called yet, it adds that key and its
    # results. If you call it in a case where there are already results, it will ask you if you want to overwrite them.

    graph = GraphProperties(graph_number)

    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
    data_dir = os.path.join(script_dir, '..', 'data')

    # load in already computed lists of unique terminal states -- free and fixed are different files
    if variant == 'fixed':
        file_path = os.path.join(data_dir, "graph_" + str(graph_number) + "_unique_terminal_states_fixed_vertices.pkl")
    else:
        if variant == 'free':
            file_path = os.path.join(data_dir, "graph_" + str(graph_number) + "_unique_terminal_states.pkl")
        else:
            sys.exit(print('variant needs to be either fixed or free!'))

    if os.path.exists(file_path):
        with open(file_path, "rb") as fp:
            game_states = pickle.load(fp)
    else:
        sys.exit(print('you need to run generate_terminal_states.py first!'))

    if solver_to_use not in ['simulated_annealing', 'schrodinger_equation', 'quantum_annealing']:
        sys.exit(print('the solver' + solver_to_use + 'is not in the allowed list -- please take a look!'))

    precision_digits = 4                        # just to clean up print output
    np.set_printoptions(suppress=True)          # remove scientific notation

    args = {'data_dir': os.path.join(os.getcwd(), '..', 'data'),
            'graph_number': graph_number}

    file_path = os.path.join(data_dir, "graph_" + str(graph_number) + "_adjudicated_unique_terminal_states_for_paper.pkl")

    if os.path.isfile(file_path):
        with open(file_path, "rb") as fp:
            adjudication_results = pickle.load(fp)
    else:   # in this case, there are no results at all and we start fresh
        adjudication_results = {}

    # at this point, either we have loaded some adjudication_results from an existing file, or we have a new empty dict
    if variant in adjudication_results:       # this means we loaded this in already
        if solver_to_use in adjudication_results[variant]:
            if solver_to_use in ['simulated_annealing']:
                if params['num_reads'] in adjudication_results[variant][solver_to_use]:
                    user_input = (
                        input('results already exist for ' + variant + " " +
                              solver_to_use + str(params['num_reads']) + ' , overwrite (y/n)?'))
                    if user_input.lower() != 'y':
                        return None
            if solver_to_use in ['quantum_annealing']:
                if params['num_reads'] in adjudication_results[variant][solver_to_use]:
                    if params['anneal_time'] in adjudication_results[variant][solver_to_use][params['num_reads']]:
                        user_input = input('results already exist for ' + variant + " " + solver_to_use + str(params['num_reads']) + str(params['anneal_time']) + ', overwrite (y/n)?')
                        if user_input.lower() != 'y':
                            return None
            if solver_to_use in ['schrodinger_equation']:
                if params['anneal_time'] in adjudication_results[variant][solver_to_use]:
                    user_input = input('results already exist for ' + variant + " " + solver_to_use + str(params['anneal_time']) + ' , overwrite (y/n)?')
                    if user_input.lower() != 'y':
                        return None

    # initialize adjudicator
    adjudicator = None

    # this should modify adjudication_results in place to add the new key without changing anything else
    if solver_to_use == 'simulated_annealing':
        adjudicator = SimulatedAnnealingAdjudicator()
        args.update({'num_reads': params['num_reads']})
        safe_nested_access_and_create(adjudication_results,
                                      [variant, solver_to_use, params['num_reads']], default_value={})
    else:
        if solver_to_use == 'quantum_annealing':
            adjudicator = QuantumAnnealingAdjudicator()
            args.update({'num_reads': params['num_reads'], 'anneal_time': params['anneal_time']})
            safe_nested_access_and_create(adjudication_results,
                                          [variant, solver_to_use, params['num_reads'], params['anneal_time']],
                                          default_value={})
        else:
            if solver_to_use == 'schrodinger_equation':
                adjudicator = SchrodingerEquationAdjudicator()
                args.update({'anneal_time': params['anneal_time']})
                safe_nested_access_and_create(adjudication_results,
                                              [variant, solver_to_use, params['anneal_time']], default_value={})

    adjudicator.setup(**args)

    # now we proceed to compute and store result
    print('beginning adjudication using the ' + solver_to_use + ' solver with parameters ' + str(args))
    start_adj = time.time()

    if solver_to_use in ['quantum_annealing']:   # no parallelism with quantum annealing -- one at a time
        items = list(game_states.items())
        for item in items:
            k, result = process_game_state(item, variant, graph, adjudicator)
            adjudication_results[variant][solver_to_use][params['num_reads']][params['anneal_time']][k] = result

    else:
        max_workers = 18   # parallelism with simulated annealing or schrodinger equation
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Create a list of argument tuples for each game state
            items = list(game_states.items())

            # Submit all tasks and track with tqdm
            futures = []
            for item in items:
                future = executor.submit(process_game_state, item, variant, graph, adjudicator)
                futures.append(future)

            # Process results as they complete
            for future in tqdm(futures, total=len(futures)):
                k, result = future.result()
                if solver_to_use == 'simulated_annealing':
                    adjudication_results[variant][solver_to_use][params['num_reads']][k] = result
                else:
                    if solver_to_use == 'schrodinger_equation':
                        adjudication_results[variant][solver_to_use][params['anneal_time']][k] = result

    print('elapsed time was', round(time.time() - start_adj, precision_digits), 'seconds.')

    # store it -- this should leave any previously loaded solver results intact
    with open(file_path, "wb") as fp:
        pickle.dump(adjudication_results, fp)


def process_game_state(item, variant, graph, adjudicator):

    k, v = item

    # the fixed variant just stores the state as keys, whereas the free version already has the game state explicit
    if variant == 'fixed':
        game_state = convert_my_game_state_to_erik_game_state(k, graph.vertex_count, graph.edge_list)
    else:
        game_state = v['game_state']

    result = adjudicator.adjudicate(game_state)

    return k, result


def main():

    # given we have computed and stored terminal states, this adjudicates all of them and stores the results

    graph_number = 11     # 19 is barbell graph; 2 is K_3; 11 is RSG
    variant = 'fixed'     # alternative is 'free'

    # list of solvers to use to adjudicate
    solver_list = ['simulated_annealing', 'schrodinger_equation', 'quantum_annealing']

    print('*******************************')
    print('adjudicating all terminal states for graph_number', graph_number)
    print('*******************************')

    # default solver parameters
    params = {'simulated_annealing': {'num_reads': 10000},
              'schrodinger_equation': {'anneal_time': 40},
              'quantum_annealing': {'num_reads': 10000, 'anneal_time': 350}}   # quantum_annealing uses D-Wave hardware

    for solver_to_use in solver_list:
        print('using', solver_to_use, 'solver ...')
        parallel_generate_adjudication_results(graph_number=graph_number,
                                               solver_to_use=solver_to_use,
                                               params=params[solver_to_use],
                                               variant=variant)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # Ensures correct behavior in PyCharm

    start = time.time()

    with cProfile.Profile() as pr:
        main()

    print('total time elapsed:', time.time() - start, 'seconds...')

    stats = pstats.Stats(pr)
    stats.strip_dirs().sort_stats('tottime').print_stats(10)   # show top 10 results
