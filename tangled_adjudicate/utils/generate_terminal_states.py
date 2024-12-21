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


def generate_all_tangled_terminal_states(graph_number):
    # this loads or generates all possible terminal game states for the graph indexed by graph_number and groups them
    # into lists where each member of the list is connected by an automorphism. Running this function requires either
    # loading or generating an automorphism file. The dictionary game_states has as its key a string with the canonical
    # member of each of these, with the further ['automorphisms'] key being a list of all the states that are symmetries
    # of the canonical key. The key ['game_state'] is the representation of the key as a game_state object.
    #
    # Note that this requires enumerating all possible terminal states, the number of which is
    # (vertex_count choose 2) * 2 * 3**edge_count, which grows exponentially with edge count. You can do this easily
    # for graph_number 1, 2, 3, 4, and 11, but the others get stupidly large.

    graph = GraphProperties(graph_number)
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
    data_dir = os.path.join(script_dir, '..', 'data')
    file_path = os.path.join(data_dir,
                             "graph_" + str(graph_number) + "_unique_terminal_states.pkl")

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

        print('there are', len(list_of_automorphisms), 'automorphisms for this graph.')

        possible_vertex_states = []
        for positions in itertools.permutations(range(graph.vertex_count), 2):
            lst = [0] * graph.vertex_count
            lst[positions[0]] = 1
            lst[positions[1]] = 2
            possible_vertex_states.append(lst)

        possible_vertex_states.sort()

        elements = [1, 2, 3]
        possible_edge_states = list(itertools.product(elements, repeat=graph.edge_count))

        # all_states is a list of lists enumerating ALL of the game states
        all_states = [j + list(k) for j in possible_vertex_states for k in possible_edge_states]

        # this next part creates a dictionary where the keys are each of the elements of all_states and the values are
        # lists of all the states connected to the key by an automorphism. Note that different automorphisms can lead
        # to the same state, so at some point the list is converted to a set and then back to a list

        all_states_with_symmetries = {}

        # iterate over all enumerated states
        for state in all_states:

            # # create a key for state
            # all_states_with_symmetries[str(state)] = []

            # create a list for all the symmetric states
            list_of_states_connected_by_symmetry = []

            # get indices of the red and blue vertices
            only_vertices = state[:graph.vertex_count]
            red_vertex_index = only_vertices.index(1)
            blue_vertex_index = only_vertices.index(2)

            # iterate over all automorphisms
            for automorph in list_of_automorphisms:

                # initialize the state we want to compute (transforming state under automorph)
                state_transformed_under_automorph = [0] * graph.vertex_count

                # write transformed vertices into the transformed state -- this finishes the vertex part
                state_transformed_under_automorph[automorph[red_vertex_index]] = 1
                state_transformed_under_automorph[automorph[blue_vertex_index]] = 2

                # now we want to transform the edges under the automorphism
                for edge_idx in range(graph.edge_count):
                    first_vertex = automorph[graph.edge_list[edge_idx][0]]
                    second_vertex = automorph[graph.edge_list[edge_idx][1]]
                    if first_vertex < second_vertex:
                        transformed_edge = (first_vertex, second_vertex)
                    else:
                        transformed_edge = (second_vertex, first_vertex)

                    transformed_edge_idx = graph.edge_list.index(transformed_edge)

                    state_transformed_under_automorph.append(state[graph.vertex_count + transformed_edge_idx])

                list_of_states_connected_by_symmetry.append(str(state_transformed_under_automorph))

            # remove duplicates
            all_states_with_symmetries[str(state)] = list(dict.fromkeys(list_of_states_connected_by_symmetry))

        # we should now have all the states with all their symmetries, with duplicates removed
        # we now want to convert this into a list of unique states, by keeping 'canonical' states and removing
        # keys that are the same state under symmetry.

        # the way I'll do this is to go through each key one by one (they should be ordered) and create 'good key'
        # and 'bad key' lists.

        unique_terminal_states = []
        duplicate_states = []

        for k, v in all_states_with_symmetries.items():
            # the first value in the list of symmetries is the key
            if v[0] not in duplicate_states:
                unique_terminal_states.append(v[0])   # string; these should be ordered
            for value_idx in range(1, len(v)):
                if v[value_idx] not in duplicate_states:
                    duplicate_states.append(v[value_idx])

        print('there are', len(unique_terminal_states), 'unique terminal states. Writing to disk ...')

        game_states = {}

        # unique_terminal_states is a list of strings, like ['[0, 1, 2, 1, 1]', '[0, 1, 2, 1, 2]', '[0, 1, 2, 1, 3]', ...]
        for each in unique_terminal_states:
            game_states[str(each)] = {}
            game_states[str(each)]['game_state'] = convert_state_string_to_game_state(graph, ast.literal_eval(each))
            game_states[str(each)]['automorphisms'] = all_states_with_symmetries[each]

        data_dir = os.path.join(os.getcwd(), '..', 'data')

        with open(os.path.join(data_dir, "graph_" + str(graph_number) + "_unique_terminal_states.pkl"), "wb") as fp:
            pickle.dump(game_states, fp)

    return game_states


def main():

    # this generates all terminal states for graphs 2 and 3
    for graph_number in range(2, 4):
        gs = generate_all_tangled_terminal_states(graph_number)


if __name__ == "__main__":
    sys.exit(main())
