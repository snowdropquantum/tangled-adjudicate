import sys
import os
import time
import pickle
import networkx as nx
from utils.game_graph_properties import GraphProperties
from utils.parameters import Params


def automorphisms(graph):
    return list(nx.algorithms.isomorphism.GraphMatcher(graph, graph).isomorphisms_iter())


def generate_list_of_automorphisms(vertex_count, edge_list, verbose=False):
    # this generates a list of automorphism dictionaries of the form
    # [{0: 0, 1: 1, 2: 2}, {0: 0, 2: 1, 1: 2}, {1: 0, 0: 1, 2: 2},
    # {1: 0, 2: 1, 0: 2}, {2: 0, 0: 1, 1: 2}, {2: 0, 1: 1, 0: 2}]

    G = nx.Graph()
    node_list = [k for k in range(vertex_count)]

    G.add_nodes_from(node_list)
    G.add_edges_from(edge_list)

    automorphisms_list = automorphisms(G)

    if verbose:
        print(len(automorphisms_list), "automorphisms found.")

    return automorphisms_list


def load_automorphism_dictionary():
    # this is just test code to make sure everything worked properly
    with open(os.path.join(os.getcwd(), '..', 'data', 'automorphism_dictionary.txt'), "rb") as fp:
        dictionary_of_automorphisms = pickle.load(fp)

    print('this is the loaded dictionary:')
    print(dictionary_of_automorphisms)


def main():
    # generates all the automorphisms for graphs 1 through 9, takes about 5 minutes, nearly all for graph 9
    #
    # I let graph 10 run all day with 128GB RAM, and it didn't complete -- think about what to do --
    # the automorphisms of both the source and target (hardware) graph both matter, the former for game
    # graph symmetries, the latter for multiple embeddings into the same physical qubits

    dictionary_of_automorphisms = {}

    for graph_number in range(1, 10):
        print('********************')
        print('Evaluating graph #', graph_number)
        print('********************')

        start = time.time()

        graph = GraphProperties(graph_number=graph_number)

        dictionary_of_automorphisms[graph_number] = generate_list_of_automorphisms(vertex_count=graph.vertex_count,
                                                                                   edge_list=graph.edge_list,
                                                                                   verbose=True)

        print('graph_number', graph_number, 'took', time.time() - start, 'seconds.')

    with open(os.path.join(os.getcwd(), '..', 'data', 'automorphism_dictionary_graphs_1_through_9.txt'), "wb") as fp:
        pickle.dump(dictionary_of_automorphisms, fp)


if __name__ == "__main__":
    sys.exit(main())
