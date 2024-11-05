""" generates graph automorphisms """

# Generating automorphisms (vertex permutation symmetries) of graphs is important in Tangled for two reasons.
#
# The first is that the game graph symmetries are equivalent board positions, and therefore evaluating a specific
# game board state is the same as evaluating all the equivalent positions at the same time. These symmetries
# can be used to decimate all terminal states down to unique canonical terminal states -- they are used e.g. to
# do data augmentation in the AlphaZero agent (evaluating one terminal state automatically gives you P evaluations,
# where P is the number of automorphisms).
#
# The second is that the same physical qubits can be used to embed P different equivalent graphs in different
# ways. Doing this is important because it 'averages over' the noise that may be present in the physical system.

import sys
import os
import time
import pickle
import networkx as nx
from utils.game_graph_properties import GraphProperties


def automorphisms(graph):
    return list(nx.algorithms.isomorphism.GraphMatcher(graph, graph).isomorphisms_iter())


def limited_automorphisms(graph, max_number=5):

    graph_matcher_object = nx.algorithms.isomorphism.GraphMatcher(graph, graph)

    # Define a generator that stops after yielding n automorphisms
    def limited_isomorphisms(gm, n):
        for i, iso in enumerate(gm.isomorphisms_iter()):
            if i < n:
                yield iso
            else:
                break

    # Get at most max_number automorphisms using the generator
    return list(limited_isomorphisms(graph_matcher_object, max_number))


def generate_list_of_automorphisms(vertex_count, edge_list, max_automorphisms_to_return, verbose=False):
    # this generates a list of automorphism dictionaries of the form
    # [{0: 0, 1: 1, 2: 2}, {0: 0, 2: 1, 1: 2}, {1: 0, 0: 1, 2: 2},
    # {1: 0, 2: 1, 0: 2}, {2: 0, 0: 1, 1: 2}, {2: 0, 1: 1, 0: 2}]

    G = nx.Graph()
    node_list = [k for k in range(vertex_count)]

    G.add_nodes_from(node_list)
    G.add_edges_from(edge_list)

    if max_automorphisms_to_return is not None:
        automorphisms_list = limited_automorphisms(G, max_number=max_automorphisms_to_return)
    else:
        automorphisms_list = automorphisms(G)

    if verbose:
        print(len(automorphisms_list), "automorphisms found.")

    return automorphisms_list


def load_automorphism_dictionary(file_name):

    with open(os.path.join(os.getcwd(), '..', 'data', file_name), "rb") as fp:
        dictionary_of_automorphisms = pickle.load(fp)

    return dictionary_of_automorphisms


def main():
    # generates automorphisms for graphs 1 through 10; default settings take about 10 minutes total to run
    #
    # I let graph 10 run all day with 128GB RAM, and it didn't complete, so I enabled a limit on the number
    # returned, which I set to 500,000. This returns all the automorphisms for graphs 1 through 9. I don't know how
    # many graph 10 has ... but it's a lot more than 500,000! For the game graph, I will likely drop some
    # vertices / edges which will dramatically reduce the number of automorphisms for it -- of course the
    # underlying hardware graph still has them.
    #
    # the dictionary_of_automorphisms object looks like
    # dictionary_of_automorphisms = {1: [{0: 0, 1: 1}, {1: 0, 0: 1}],
    # 2: [{0: 0, 1: 1, 2: 2}, {0: 0, 2: 1, 1: 2}, {1: 0, 0: 1, 2: 2},
    # {1: 0, 2: 1, 0: 2}, {2: 0, 0: 1, 1: 2}, {2: 0, 1: 1, 0: 2}]}

    generate_automorphism_file = True  # if True generates the dictionary in /data directory, else attempts to load it
    automorphism_file_name = 'automorphism_dictionary_graphs_1_through_10_max_500000.pkl'   # name of data file
    max_automorphisms_to_return = 500000   # either an integer, or None if you want all of them

    # checks to see if /data exists; if it doesn't it creates it; if it does, it writes the file to disk
    data_dir = os.path.join(os.getcwd(), '..', 'data')

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    if generate_automorphism_file:
        dictionary_of_automorphisms = {}

        for graph_number in range(1, 11):
            print('********************')
            print('Evaluating graph #', graph_number)
            print('********************')

            start = time.time()

            graph = GraphProperties(graph_number=graph_number)

            dictionary_of_automorphisms[graph_number] = (
                generate_list_of_automorphisms(vertex_count=graph.vertex_count,
                                               edge_list=graph.edge_list,
                                               max_automorphisms_to_return=max_automorphisms_to_return,
                                               verbose=True))

            print('graph_number', graph_number, 'took', time.time() - start, 'seconds.')

        with open(os.path.join(data_dir, automorphism_file_name), "wb") as fp:
            pickle.dump(dictionary_of_automorphisms, fp)

    else:

        dictionary_of_automorphisms = load_automorphism_dictionary(file_name=automorphism_file_name)
        print('this is the loaded dictionary:')
        print(dictionary_of_automorphisms)


if __name__ == "__main__":
    sys.exit(main())
