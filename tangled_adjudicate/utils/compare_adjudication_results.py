""" assumes you have computed and stored adjudication results already using adjudicate_all_terminal_states.py """
import sys
import os
import pickle
import matplotlib.pyplot as plt
from itertools import combinations


def compare_adjudication_results(graph_number, solvers_to_use):
    # solvers_to_use is a list of solvers of length either 2 or 3 comprising 2 or 3 of
    # ['simulated_annealing', 'schrodinger_equation', 'quantum_annealing']
    #
    # indexing_solvers = {1: 'simulated_annealing', 2: 'schrodinger_equation', 3: 'quantum_annealing'}

    # load adjudication results obtained from running /utils/adjudicate_all_terminal_states.py
    data_dir = os.path.join(os.getcwd(), '..', 'data')
    file_name = "graph_" + str(graph_number) + "_terminal_states_adjudication_results.pkl"

    with open(os.path.join(data_dir, file_name), "rb") as fp:
        adjudication_results = pickle.load(fp)

    # check to make sure the entries in solvers_to_use have adjudication results already
    for each in solvers_to_use:
        if each not in adjudication_results:
            sys.exit(print('no adjudication results found for solver ' +
                           each + '. Run adjudicate_all_terminal_states.py using this solver first.'))

    # OK so we now have some adjudication results to compare. For each canonical terminal state we will generate a
    # list with three booleans, corresponding to 1v2, 1v3, 2v3 respectively. The boolean is True if the pair agree and
    # False if they don't.

    # initialize game_result dict, load in results from the different solvers requested
    game_result = {}  # this is a dict whose keys are the canonical terminal states
    scores = {}     # holds the scores for all the terminal states in a list

    for k0, value_dict in adjudication_results.items():   # k0 is the solver name string
        for k1, v in value_dict.items():   # k1 is the game state string
            game_result[k1] = []
    for each in solvers_to_use:
        scores[each] = []

    for k0, value_dict in adjudication_results.items():  # k will be solver name string
        if k0 in solvers_to_use:   # if we want to add this, add it
            for k1, v in value_dict.items():
                game_result[k1].append([k0, v['winner'], v['score']])

    comparisons = {}
    for k, v in game_result.items():    # k is game state string
        comparisons[k] = []
        for a, b in combinations(v, 2):
            comparisons[k].append(a[1] == b[1])

    for k, v in comparisons.items():
        if False in v:
            print('key ', k, 'has a mismatch!')

    for k, v in game_result.items():
        for each in v:
            scores[each[0]].append(each[2])

    to_plot = []
    for k, v in scores.items():
        to_plot.append(v)

    red_text = solvers_to_use[0] + ': red'
    blue_text = solvers_to_use[1] + ': blue'

    if len(solvers_to_use) == 3:
        cyan_text = solvers_to_use[2] + ': cyan'

    if graph_number == 2:

        if len(to_plot) == 2:
            plt.hist(to_plot, range=[-2, 2], bins=200, color=['red', 'blue'], stacked=True)
        else:
            plt.hist(to_plot, range=[-2, 2], bins=200, color=['red', 'blue', 'cyan'], stacked=True)


        plt.text(1, 20, r'Three Vertex Graph', fontsize=12)
        plt.text(1, 18, red_text, fontsize=8)
        plt.text(1, 17, blue_text, fontsize=8)
        if len(solvers_to_use) == 3:
            plt.text(1, 16, cyan_text, fontsize=8)

        plt.ylim(0, 25)

        plt.xlabel('Score')
        plt.ylabel('Terminal State Count')

        plt.vlines(x=0.5, ymin=0, ymax=20, colors='green', ls=':', lw=1)
        plt.vlines(x=-0.5, ymin=0, ymax=20, colors='green', ls=':', lw=1)

    if graph_number == 3:

        if len(to_plot) == 2:
            plt.hist(to_plot, range=[-1, 1], bins=400, color=['red', 'blue'], stacked=True)
        else:
            plt.hist(to_plot, range=[-1, 1], bins=400, color=['red', 'blue', 'cyan'], stacked=True)

        plt.text(.7, 70, r'Four Vertex Graph', fontsize=12)
        plt.text(.7, 65, red_text, fontsize=8)
        plt.text(.7, 61, blue_text, fontsize=8)
        if len(solvers_to_use) == 3:
            plt.text(.7, 57, cyan_text, fontsize=8)
        plt.ylim(0, 100)

        plt.xlabel('Score')
        plt.ylabel('Terminal State Count')

        plt.vlines(x=0.5, ymin=0, ymax=70, colors='green', ls=':', lw=1)
        plt.vlines(x=-0.5, ymin=0, ymax=70, colors='green', ls=':', lw=1)

    plt.show()


def main():

    all_solvers = ['simulated_annealing', 'schrodinger_equation', 'quantum_annealing']
    solvers_to_use = ['simulated_annealing', 'schrodinger_equation']

    for graph_number in range(2, 4):
        compare_adjudication_results(graph_number=graph_number, solvers_to_use=solvers_to_use)


if __name__ == "__main__":
    sys.exit(main())
