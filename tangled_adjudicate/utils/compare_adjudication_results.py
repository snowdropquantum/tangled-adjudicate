""" assumes you have computed and stored adjudication results already using adjudicate_all_terminal_states.py """
import sys
import os
import pickle
from itertools import combinations
from collections import defaultdict

from tangled_adjudicate.utils.utilities import evaluate_winner, load_lookup_table
from tangled_adjudicate.utils.game_graph_properties import GraphProperties


def compare_adjudication_results(graph_number, solvers_to_use, lookup_table_solvers_to_use, epsilon, anneal_time, num_reads):
    # this compares results from a set of solvers + parameters
    # solvers_to_use is a list of solvers in ['schrodinger_equation', 'simulated_annealing', 'quantum_annealing', 'lookup_table']
    # lookup_table_solvers_to_use is a list of solvers whose lookup tables you want to use

    graph = GraphProperties(graph_number)

    data_dir = os.path.join(os.getcwd(), '..', 'data')
    raw_file_path = os.path.join(data_dir,
                                 "graph_" + str(graph_number) + "_adjudicated_unique_terminal_states_for_paper.pkl")

    try:
        with open(raw_file_path, 'rb') as fp:
            raw_adjudication_data = pickle.load(fp)
    except Exception as e:
        raise RuntimeError(f"Failed to load raw adjudication data: {str(e)}")

    results = defaultdict(lambda: defaultdict(dict))

    for solver in solvers_to_use:
        if solver == 'simulated_annealing':
            for k, v in raw_adjudication_data['fixed']['simulated_annealing'][num_reads['simulated_annealing']].items():
                results[k][solver] = [v['score'], evaluate_winner(v['score'], epsilon=epsilon)]
        if solver == 'schrodinger_equation':
            for k, v in raw_adjudication_data['fixed']['schrodinger_equation'][anneal_time['schrodinger_equation']].items():
                results[k][solver] = [v['score'], evaluate_winner(v['score'], epsilon=epsilon)]
        if solver == 'quantum_annealing':
            for k, v in raw_adjudication_data['fixed']['quantum_annealing'][num_reads['quantum_annealing']][anneal_time['quantum_annealing']].items():
                results[k][solver] = [v['score'], evaluate_winner(v['score'], epsilon=epsilon)]
        if solver == 'lookup_table':
            for each in lookup_table_solvers_to_use:
                lookup_table = load_lookup_table(data_dir, graph_number, each, epsilon, anneal_time[each], num_reads[each])
                for k, v in lookup_table.items():
                    results[k][solver][each] = [None, v]

    cnt = 0
    current_biggest_score_difference = 0
    big_key = None
    big_max_index = None
    big_min_index = None
    pos_val = None
    neg_val = None

    for k, v in results.items():   # key is '[1, 0, 2, 1, 1]'
        # Only look at the ones where 1 is at red's position and 2 is at blue's position,
        # not the reversed ones which are only used for lookup table + alphazero
        vertices = ''.join(k.strip('[]').split(', ')[:graph.vertex_count])
        pos_1 = vertices.find('1')
        pos_2 = vertices.find('2')
        if pos_1 == graph.vertex_ownership[graph_number][0] and pos_2 == graph.vertex_ownership[graph_number][1]:
            list_of_results = []
            scores = []
            for solver in solvers_to_use:
                if solver == 'lookup_table':
                    for lut_solver in lookup_table_solvers_to_use:
                        list_of_results.append(v[solver][lut_solver][1])
                else:
                    list_of_results.append(v[solver][1])
                    scores.append(v[solver][0])
            if len(set(list_of_results)) > 1:
                print('key ', k, 'has a mismatch; values are ', list_of_results)
                cnt += 1

            most_positive = max(scores)
            most_negative = min(scores)
            biggest_score_difference = most_positive - most_negative

            if biggest_score_difference > current_biggest_score_difference:
                big_key = k
                current_biggest_score_difference = biggest_score_difference
                neg_val = most_negative
                pos_val = most_positive
                big_max_index = scores.index(most_positive)
                big_min_index = scores.index(most_negative)

    print('total adjudication mismatches:', cnt)
    print('key with biggest mismatch:', big_key)
    print('biggest score mismatch:', current_biggest_score_difference)
    print('solvers:', solvers_to_use[big_max_index], solvers_to_use[big_min_index])
    print('respective scores:', pos_val, neg_val)


def main():

    graph_number = 19   # 11 is P_3, 2 is K_3, 20 is diamond graph, 19 is barbell graph
    solvers_to_use = ['simulated_annealing', 'schrodinger_equation', 'quantum_annealing', 'lookup_table']
    lookup_table_solvers_to_use = ['simulated_annealing', 'schrodinger_equation', 'quantum_annealing']
    epsilon = 0.25   # 0.25 for 19, 0.5 for the rest
    anneal_time = {'quantum_annealing': 350, 'schrodinger_equation': 40, 'simulated_annealing': None}
    num_reads = {'quantum_annealing': 100000, 'schrodinger_equation': None, 'simulated_annealing': 100000}

    compare_adjudication_results(graph_number, solvers_to_use, lookup_table_solvers_to_use, epsilon, anneal_time, num_reads)


if __name__ == "__main__":
    sys.exit(main())
