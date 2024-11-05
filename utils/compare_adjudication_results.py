""" assumes you already have computed and stored adjudication results already using
adjudicate_all_terminal_states.py """
import sys
import os
import pickle
import ast
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from utils.game_graph_properties import GraphProperties
from utils.parameters import Params



def compare_adjudication_results(params):

    solvers =['simulated_annealing', 'schrodinger_equation', 'quantum_annealing']

    # load appropriate data file into results dictionary
    data_dir = os.path.join(os.getcwd(), '..', 'data')
    file_name = "graph_" + str(params.GRAPH_NUMBER) + "_unique_terminal_states_adjudication_results.pkl"

    with open(os.path.join(data_dir, file_name), "rb") as fp:
        results = pickle.load(fp)

    if params.GRAPH_NUMBER == 3:   # for graph 3, load in the quantum annealing result separately
        file_name = "graph_" + str(params.GRAPH_NUMBER) + "_unique_terminal_states_adjudication_results_just_qa_just_problems.pkl"

        with open(os.path.join(data_dir, file_name), "rb") as fp:
            results_just_qa = pickle.load(fp)

        for k, v in results_just_qa.items():
            results[k]['quantum_annealing'] = results_just_qa[k]['quantum_annealing']

    game_result = {}
    # iterate over all key, value pairs in results -- key is the string of the original state representation
    disagreement_count = 0
    disagree_keys = []
    for k, v in results.items():
        game_result[k] = {}
        for each_solver in solvers:
            game_result[k][each_solver] = [v[each_solver][0]['winner'], v[each_solver][0]['score']]

        if abs(abs(v['quantum_annealing'][0]['score'])-0.25) < 0.1 or abs(abs(v['simulated_annealing'][0]['score'])-0.25) < 0.1 or abs(abs(v['schrodinger_equation'][0]['score'])-0.25) < 0.1:

            print('solver disagreement about the result for key', k)
            disagree_keys.append(k)
            disagreement_count += 1
    print('out of a total of', len(game_result), 'unique states, number of disagreements was', disagreement_count)

    with open(os.path.join(data_dir, 'disagree_keys.pkl'), "wb") as fp:
        pickle.dump(disagree_keys, fp)

    qa_score = []
    sa_score = []
    se_score = []

    for k, v in game_result.items():
        if k == '[0, 0, 1, 2, 1, 1, 2, 3, 1, 2]':
            print()
        qa_score.append(v['quantum_annealing'][1])
        sa_score.append(v['simulated_annealing'][1])
        se_score.append(v['schrodinger_equation'][1])
        if k in disagree_keys:
            print('key', k, 'has scores of', v['quantum_annealing'][1],
                  v['simulated_annealing'][1], v['schrodinger_equation'][1])
        # if abs(abs(v['simulated_annealing'][1])-0.5) < 0.1:
        if abs(abs(v['quantum_annealing'][1])-0.5) < 0.1 or abs(abs(v['simulated_annealing'][1])-0.5) < 0.1 or abs(abs(v['schrodinger_equation'][1])-0.5) < 0.1:
            print('key', k, 'has scores of', v['quantum_annealing'][1],
                  v['simulated_annealing'][1], v['schrodinger_equation'][1])
    to_plot = [se_score, sa_score, qa_score]

    if params.GRAPH_NUMBER == 2:

        plt.hist(to_plot, range=[-2, 2], bins=200, color=['red', 'blue', 'cyan'], stacked=True)

        plt.text(1, 20, r'Three Vertex Graph', fontsize=12)
        plt.text(1, 18, r'Schrodinger Equation: Red', fontsize=8)
        plt.text(1, 17, r'Simulated Annealing: Blue', fontsize=8)
        plt.text(1, 16, r'Quantum Annealing: Cyan', fontsize=8)
        plt.ylim(0, 25)

        plt.xlabel('Score')
        plt.ylabel('Terminal State Count')

        plt.vlines(x=0.5, ymin=0, ymax=20, colors='green', ls=':', lw=1)
        plt.vlines(x=-0.5, ymin=0, ymax=20, colors='green', ls=':', lw=1)

    if params.GRAPH_NUMBER == 3:

        plt.hist(to_plot, range=[-1, 1], bins=400, color=['red', 'blue', 'cyan'], stacked=True)

        plt.text(.7, 70, r'Four Vertex Graph', fontsize=12)
        plt.text(.7, 65, r'Schrodinger Equation: Red', fontsize=8)
        plt.text(.7, 61, r'Simulated Annealing: Blue', fontsize=8)
        plt.text(.7, 57, r'Quantum Annealing: Cyan', fontsize=8)
        plt.ylim(0, 100)

        plt.xlabel('Score')
        plt.ylabel('Terminal State Count')

        plt.vlines(x=0.5, ymin=0, ymax=70, colors='green', ls=':', lw=1)
        plt.vlines(x=-0.5, ymin=0, ymax=70, colors='green', ls=':', lw=1)

    plt.show()


def compile_results():

    graph_number = 3

    graph = GraphProperties(graph_number=graph_number)

    initial_state = [0] * (graph.vertex_count + len(graph.edge_list))

    file_name_prefix = "graph_" + str(graph_number) + "_root_" + str(initial_state)

    with open(os.path.join(os.getcwd(), '..', 'results', file_name_prefix + "_scores_for_all_terminal_states.txt"), "rb") as fp:
        sorted_results = pickle.load(fp)

    dict_to_write = {}

    for k, v in sorted_results.items():
        dict_to_write[k] = v[0]

    with open(os.path.join(os.getcwd(), '..', 'results', file_name_prefix + "_adjudications_all_terminal_states.txt"), "wb") as fp:
        pickle.dump(dict_to_write, fp)


def compare_results():
    solver_0 = 'SE'
    # solver_1 = 'SA'
    solver_1 = 'QC'

    graph_number = 2

    # smallest epsilon for zero disagreements for SA vs SE
    # for graph 2 0.4400 for 100 ==> 0.11401 for 1000 ==> 0.03201 for 10,000 ==> 0.01161 for 100,000 ==>
    # 0.00703 for 1,000,000 ==> 0.00683 for 10,000,000
    # for graph 3 2.6 for 100 ==> 0.276 for 1000 ==> 0.11521 for 10,000 ==> 0.05036 for 100,000 ==>
    # 0.04817 for 1,000,000 ==> 0.04758 for 10,000,000

    for ta in [5]:
        epsilon_trial = 0.5
        for num_reads in [10000000]:

            print('******************************************************')
            print('evaluating for ta =', ta, 'ns and num_reads =', num_reads)
            print('******************************************************')
            exit_flag = False
            while not exit_flag:
                num_disagreements = compare_adjudication_results(solver_0=solver_0, solver_1=solver_1,
                                                                 num_reads=num_reads, ta=ta,
                                                                 graph_number=graph_number, epsilon=epsilon_trial)
                if num_disagreements > 0:
                    print('number of disagreements for epsilon', epsilon_trial, 'is', num_disagreements)
                    # epsilon_trial += 0.00001
                    epsilon_trial += 0.000001
                    print(epsilon_trial)
                else:
                    print("smallest epsilon for zero disagreements was", round(epsilon_trial, 5))
                    print()
                    exit_flag = True


def main():
    params = Params()
    compare_adjudication_results(params=params)


if __name__ == "__main__":
    sys.exit(main())
