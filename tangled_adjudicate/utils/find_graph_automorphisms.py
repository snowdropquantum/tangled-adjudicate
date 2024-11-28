""" generates graph automorphisms """

# Generating automorphisms (vertex permutation symmetries) of graphs is important in Tangled for two reasons.
#
# The first is that the game graph symmetries are equivalent board positions, and therefore evaluating a specific
# game board state is the same as evaluating all the equivalent positions at the same time. These symmetries
# can be used to decimate all terminal states down to unique canonical terminal states -- they are used e.g. to
# do data augmentation in the AlphaZero agent (evaluating one terminal state automatically gives you M evaluations,
# where M is the number of automorphisms).
#
# The second is that the same physical qubits can be used to embed M different equivalent graphs in different
# ways. Doing this is important because it 'averages over' the noise that may be present in the physical system.

import os
import time
import pickle

import networkx as nx
from tangled_adjudicate.utils.game_graph_properties import GraphProperties


def automorphisms(graph):
    return list(nx.algorithms.isomorphism.GraphMatcher(graph, graph).isomorphisms_iter())


def limited_automorphisms(graph, max_number=5):
    # if there are too many to compute, you can use this function that will stop after computing max_number of them

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

    graph_to_use = nx.Graph()
    node_list = [k for k in range(vertex_count)]

    graph_to_use.add_nodes_from(node_list)
    graph_to_use.add_edges_from(edge_list)

    if max_automorphisms_to_return is not None:
        automorphisms_list = limited_automorphisms(graph_to_use, max_number=max_automorphisms_to_return)
    else:
        automorphisms_list = automorphisms(graph_to_use)

    if verbose:
        print(len(automorphisms_list), "automorphisms found.")

    return automorphisms_list


def get_automorphisms(graph_number, data_dir):
    # returns automorphisms for the graph corresponding to the graph_number you want (graph_number is the index of
    # your graph as defined in /utils/game_graph_properties.py, default 1 through 10, but you can add new ones if
    # you want -- if you do, I'd suggest indexing your new graphs starting at 11 so you don't accidentally confuse
    # which graph is which)
    #
    # if they exist already, they are loaded, if not, they are computed
    # all the default graphs compute fast except graph 10, where default settings take about 10 minutes total to run
    # I let graph 10 run all day with 128GB RAM, and it didn't complete, so I enabled a limit on the number
    # returned, which I set to 500,000. This returns all the automorphisms for graphs 1 through 9. I don't know how
    # many graph 10 has ... but it's a lot more than 500,000!
    #
    # Note: these are the automorphisms of the source graph -- the processor target graph is not relevant here
    #
    # the list_of_automorphisms object returned is a list of the different automorphisms of your graph, like this:
    #
    # list_of_automorphisms = [{0: 0, 1: 1, 2: 2}, {0: 0, 2: 1, 1: 2}, {1: 0, 0: 1, 2: 2}, {1: 0, 2: 1, 0: 2},
    # {2: 0, 0: 1, 1: 2}, {2: 0, 1: 1, 0: 2}]

    max_automorphisms_to_return = 500000                    # either an integer, or None if you want all of them

    file_name = 'automorphisms_graph_number_' + str(graph_number) + '_max_' + str(max_automorphisms_to_return) + '.pkl'

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    file_path = os.path.join(data_dir, file_name)

    # checks to see if the file is there already; if it is, load it; if not, create it
    if os.path.isfile(file_path):
        with open(file_path, "rb") as fp:
            list_of_automorphisms = pickle.load(fp)
    else:
        print('********************')
        print('automorphism file not found, creating it, this should only happen once per graph ...')
        print('finding automorphisms for graph #', graph_number)
        print('********************')

        start = time.time()

        graph = GraphProperties(graph_number=graph_number)

        list_of_automorphisms = generate_list_of_automorphisms(vertex_count=graph.vertex_count,
                                                               edge_list=graph.edge_list,
                                                               max_automorphisms_to_return=max_automorphisms_to_return,
                                                               verbose=True)

        print('for graph_number', graph_number, 'this took', time.time() - start, 'seconds.')

        with open(os.path.join(data_dir, file_name), "wb") as fp:
            pickle.dump(list_of_automorphisms, fp)

    return list_of_automorphisms
