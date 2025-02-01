""" generates unique terminal states, states related to them by symmetry, and game_state objects for them """
import sys
import os
import itertools
import pickle
import math
import ast
import numpy as np

from tangled_adjudicate.utils.game_graph_properties import GraphProperties
from tangled_adjudicate.utils.find_graph_automorphisms import get_automorphisms
from tangled_adjudicate.utils.utilities import convert_my_game_state_to_erik_game_state


# def convert_state_string_to_game_state(my_state, number_of_vertices, list_of_edge_tuples):
#
#     my_vertices = my_state[:number_of_vertices]
#     my_edges = my_state[number_of_vertices:]
#
#     turn_count = 0
#
#     try:
#         player_1_vertex = my_vertices.index(1)
#         turn_count += 1
#     except ValueError:
#         player_1_vertex = -1
#
#     try:
#         player_2_vertex = my_vertices.index(2)
#         turn_count += 1
#     except ValueError:
#         player_2_vertex = -1
#
#     turn_count += my_edges.count(1) + my_edges.count(2) + my_edges.count(3)
#
#     # if turn_count is even, it's player 1 (red)'s turn
#     current_player_idx = 1 if turn_count % 2 == 0 else 2
#
#     erik_edges = [(list_of_edge_tuples[k][0], list_of_edge_tuples[k][1], my_edges[k]) for k in range(len(my_edges))]
#
#     game_state = {'num_nodes': number_of_vertices, 'edges': erik_edges,
#                   'player1_id': 'player1', 'player2_id': 'player2', 'turn_count': turn_count,
#                   'current_player_index': current_player_idx,
#                   'player1_node': player_1_vertex, 'player2_node': player_2_vertex}
#
#     return game_state


# todo change this to the visualize_and_enumerate code

def generate_all_tangled_terminal_states(graph_number):
    # this loads or generates all possible terminal game states for the graph indexed by graph_number and groups them
    # into lists where each member of the list is connected by an automorphism. Running this function requires either
    # loading or generating an automorphism file.The dictionary game_states has as its key a string with the canonical
    # member of each of these, with the further ['automorphisms'] key being a list of all the states that are symmetries
    # of the canonical key. The key ['game_state'] is the representation of the key as a game_state object.
    #
    # Note that this requires enumerating all possible terminal states, the number of which is
    # (vertex_count choose 2) * 2 * 3**edge_count, which grows exponentially with edge count. You can do this easily
    # for graph_number 1, 2, 3, 4, but 5 and up get stupidly large.

    graph = GraphProperties(graph_number)
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
    data_dir = os.path.join(script_dir, '..', 'data')
    file_path = os.path.join(data_dir, "graph_" + str(graph_number) + "_unique_terminal_states.pkl")

    if os.path.isfile(file_path):   # if the file already exists, just load it
        with open(file_path, "rb") as fp:
            game_states = pickle.load(fp)
    else:
        # add check to make sure you don't ask for something too large
        print('***************************')
        user_input = input('There are ' + str(math.comb(graph.vertex_count, 2) * 2 * 3**graph.edge_count) +
                           ' terminal states -- proceed (y/n)?')
        if user_input.lower() != 'y':
            sys.exit(print('exiting...'))
        print('***************************')

        list_of_automorphisms = get_automorphisms(graph_number, data_dir=data_dir)

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
            for automorph in list_of_automorphisms:
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

        for my_game_state in terminal_states:
            game_states[str(my_game_state)] = {}
            game_states[str(my_game_state)]['game_state'] = convert_my_game_state_to_erik_game_state(my_game_state, graph.vertex_count, graph.edge_list)
            game_states[str(my_game_state)]['automorphisms'] = good_states[str(my_game_state)]

        data_dir = os.path.join(os.getcwd(), '..', 'data')

        with open(os.path.join(data_dir, "graph_" + str(graph_number) + "_unique_terminal_states.pkl"), "wb") as fp:
            pickle.dump(game_states, fp)

    return game_states


def main():

    # this generates all terminal states for graphs 2 and 3
    gs2 = generate_all_tangled_terminal_states(graph_number=2)
    gs3 = generate_all_tangled_terminal_states(graph_number=3)
    print()

if __name__ == "__main__":
    sys.exit(main())
