""" generates unique terminal states, states related to them by symmetry, and game_state objects for them """
import sys
import os
import itertools
import pickle
import math
import ast
import numpy as np
from utils.game_graph_properties import GraphProperties
from utils.parameters import Params


def convert_state_string_to_game_state(graph, terminal_state_string):

    vertex_list = terminal_state_string[:graph.vertex_count]
    edge_list = terminal_state_string[graph.vertex_count:]
    edges = [(graph.edge_list[k][0], graph.edge_list[k][1], edge_list[k]) for k in range(len(edge_list))]

    turn_count = vertex_list.count(1) + vertex_list.count(2) + len(edge_list) - edge_list.count(0)

    # if turn_count is even, it's red's turn
    if not turn_count % 2:
        current_player_index = 1
    else:
        current_player_index = 2

    game_state = {'num_nodes': graph.vertex_count, 'edges': edges,
                  'player1_id': 'player1', 'player2_id': 'player2', 'turn_count': turn_count,
                  'current_player_index': current_player_index,
                  'player1_node': vertex_list.index(1), 'player2_node': vertex_list.index(2)}

    return game_state


def write_unique_states_to_disk():
    # this generates all possible terminal game states and groups them into lists where each member of the list is
    # connected by an automorphism. The dictionary game_states has as its key a string with the canonical member of
    # each of these, with the further ['automorphisms'] key being a list of all the states that are symmetries of the
    # canonical key. The key ['game_state'] is the representation of the key as a game_state object.
    #
    # set the Params value to the GRAPH_NUMBER you want and run this code. Note that this requires enumerating all
    # possible terminal states, the number of which is (vertex_count choose 2) * 2 * 3**edge_count, which grows
    # exponentially with edge count. You can do this easily for graph_number 1, 2, 3, 4, but 5 and up get stupidly
    # large.

    params = Params()
    graph = GraphProperties(params.GRAPH_NUMBER)
    file_name_prefix = "graph_" + str(params.GRAPH_NUMBER)

    data_dir = os.path.join(os.getcwd(), '..', 'data')
    automorph_file = 'automorphism_dictionary_graphs_1_through_10_max_500000.pkl'
    with open(os.path.join(data_dir, automorph_file), "rb") as fp:
        automorphs = pickle.load(fp)[params.GRAPH_NUMBER]

    possible_vertex_states = []
    for positions in itertools.permutations(range(graph.vertex_count), 2):
        lst = [0] * graph.vertex_count
        lst[positions[0]] = 1
        lst[positions[1]] = 2
        possible_vertex_states.append(lst)

    possible_vertex_states.sort()

    elements = [1, 2, 3]
    possible_edge_states = list(itertools.product(elements, repeat=graph.edge_count))

    all_states = [j + list(k) for k in possible_edge_states for j in possible_vertex_states]

    same_group_of_states = {}

    for state in all_states:
        only_vertices = state[:graph.vertex_count]
        red_vertex_index = only_vertices.index(1)
        blue_vertex_index = only_vertices.index(2)
        same_group_of_states[str(state)] = []
        for automorph in automorphs:
            new_red_vertex_index = automorph[red_vertex_index]
            new_blue_vertex_index = automorph[blue_vertex_index]
            transformed_each = [0] * graph.vertex_count
            transformed_each[new_red_vertex_index] = 1
            transformed_each[new_blue_vertex_index] = 2

            edge = np.zeros((graph.vertex_count, graph.vertex_count))
            new_edge = np.zeros((graph.vertex_count, graph.vertex_count))
            cnt = graph.vertex_count
            for j in range(graph.vertex_count):
                for i in range(j):
                    edge[i, j] = state[cnt]
                    cnt += 1

            cnt = graph.vertex_count
            for j in range(graph.vertex_count):
                for i in range(j):
                    if automorph[i] < automorph[j]:
                        new_edge[i, j] = edge[automorph[i], automorph[j]]
                    else:
                        new_edge[i, j] = edge[automorph[j], automorph[i]]
                    cnt += 1

            for j in range(graph.vertex_count):
                for i in range(j):
                    transformed_each.append(int(new_edge[i, j]))
            same_group_of_states[str(state)].append(transformed_each)

    good_states = {}
    cnt = 0
    for k, v in same_group_of_states.items():
        if not cnt % (math.comb(graph.vertex_count, 2) * 2):  # 4 choose 2 = 6 * 2 = 12  ..... 3 choose 2 = 3 *2 = 6   math.comb(graph.vertex_count, 2) * 2
            good_states[k] = v
        cnt += 1

    terminal_states = []
    for k, v in good_states.items():
        terminal_states.append(ast.literal_eval(k))

    print('there are', len(terminal_states), 'unique terminal states. Writing to disk ...')

    game_states = {}

    for each in terminal_states:
        game_states[str(each)] = {}
        game_states[str(each)]['game_state'] = convert_state_string_to_game_state(graph, each)
        game_states[str(each)]['automorphisms'] = good_states[str(each)]

    with open(os.path.join(data_dir, file_name_prefix + "_unique_terminal_states.txt"), "wb") as fp:
        pickle.dump(game_states, fp)


def main():
    write_unique_states_to_disk()


if __name__ == "__main__":
    sys.exit(main())
