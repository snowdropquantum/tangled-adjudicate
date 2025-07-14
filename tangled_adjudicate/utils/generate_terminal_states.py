""" generates unique terminal states, states related to them by symmetry, and game_state objects for them """
import sys
import os
import itertools
import pickle
import math
import ast
import time

from tqdm import tqdm
import numpy as np
from collections import defaultdict

from tangled_adjudicate.utils.game_graph_properties import GraphProperties
from tangled_adjudicate.utils.find_graph_automorphisms import get_automorphisms
from tangled_adjudicate.utils.utilities import convert_my_game_state_to_erik_game_state


def extract_unique_terminal_states_for_fixed_vertices(graph_number):

    #############################################################################################################
    # output file is a dictionary with unique states as keys, and values are lists of states that are equivalent;
    # if there are no equivalent states, this list is empty
    #############################################################################################################

    graph = GraphProperties(graph_number)

    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
    data_dir = os.path.join(script_dir, '..', 'data')

    file_path = os.path.join(data_dir, "graph_" + str(graph_number) + "_unique_terminal_states.pkl")

    with open(file_path, "rb") as fp:
        game_states = pickle.load(fp)

    file_path = os.path.join(data_dir, "graph_" + str(graph_number) + "_unique_terminal_states_fixed_vertices.pkl")

    user_input = None

    if os.path.isfile(file_path):   # if the file already exists, just load it
        user_input = input("graph_" + str(graph_number) + "_unique_terminal_states_fixed_vertices.pkl already exists, overwrite (y/n)?")

    if user_input is None or user_input.lower() == 'y':
        unique_states = {}
        for k, v in tqdm(game_states.items(), total=len(game_states)):
            key_to_use = None
            for each in v['automorphisms']:   # each is a string like '[0,1,2,1,1,1]'
                possible_state = ast.literal_eval(each)   # possible_state is a list like [0,1,2,1,1,1]
                if possible_state[graph.vertex_ownership[graph_number][0]] == 1 and possible_state[graph.vertex_ownership[graph_number][1]] == 2:
                    if key_to_use is None:
                        key_to_use = str(possible_state)
                        unique_states[key_to_use] = []
                    unique_states[key_to_use].append(str(possible_state))
            if key_to_use is not None:
                my_set = set(unique_states[key_to_use])
                my_set.discard(key_to_use)
                unique_states[key_to_use] = list(my_set)

        print('there are', len(unique_states), 'unique states...')

        with open(file_path, "wb") as fp:
            pickle.dump(unique_states, fp)


def generate_all_tangled_terminal_states(graph_number):
    # this loads or generates all possible terminal game states for the graph indexed by graph_number and groups them
    # into lists where each member of the list is connected by an automorphism. Running this function requires either
    # loading or generating an automorphism file.
    #
    #######################################################################################################
    # The dictionary game_states has as its key a string with the canonical member of each of these,
    # with the further ['automorphisms'] key being a list of all the states that are symmetries
    # of the canonical key. The key ['game_state'] is the representation of the key as a game_state object.
    #######################################################################################################
    #
    # Note that this requires enumerating all possible terminal states, the number of which is
    # (vertex_count choose 2) * 2 * 3**edge_count, which grows exponentially with edge count.
    #
    # graph_number 2 should have 27 keys, and each ['automorphisms'] sub-key should have 6 entries
    # graph_number 3 should have 405 keys, and each ['automorphisms'] sub-key should have 12-24 entries (the reason
    # why there aren't always 24 is that for some of these keys different automorphisms bring you to the same state)
    # graph_number 20 should have 756 keys
    # graph_number 4 should have 1,836 keys
    # graph_number 19 should have 10,449 keys
    # graph_number 18 should have 49,572 keys
    # graph_number 17 should have 623,322 keys
    # graph_number 12 should have 1,030,077 keys
    #
    # for the fixed_vertices variant:
    # graph_number 2 has 27 unique states
    # graph_number 20 has 135 unique states
    # graph_number 3 has 405 unique states
    # graph_number 4 has 378 unique states
    # graph_number 19 has 2,187 unique states
    # graph_number 18 has 19,683 unique states
    # graph_number 17 has 89,694 unique states
    # graph_number 12 has 177,147 unique states

    graph = GraphProperties(graph_number)

    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
    data_dir = os.path.join(script_dir, '..', 'data')
    list_of_automorphisms = get_automorphisms(graph_number, data_dir=data_dir)

    file_path = os.path.join(data_dir, "graph_" + str(graph_number) + "_unique_terminal_states.pkl")

    user_input = None

    if os.path.isfile(file_path):   # file already exists
        user_input = input("graph_" + str(graph_number) + "_unique_terminal_states.pkl already exists, overwrite (y/n)?")

    if user_input is None or user_input.lower() == 'y':
        # add check to make sure you don't ask for something too large
        print('***************************')
        user_input = input('There are ' + str(math.comb(graph.vertex_count, 2) * 2 * 3**graph.edge_count) +
                           ' total non-unique terminal states -- proceed (y/n)?')
        if user_input.lower() != 'y':
            sys.exit(print('exiting...'))
        print('***************************')

        start_of_this = time.time()

        # Create possible_vertex_states as numpy array
        #################################################################################################
        vertex_permutations = list(itertools.permutations(range(graph.vertex_count), 2))
        possible_vertex_states = np.zeros((len(vertex_permutations), graph.vertex_count), dtype=np.uint8)

        for i, positions in enumerate(vertex_permutations):
            possible_vertex_states[i, positions[0]] = 1
            possible_vertex_states[i, positions[1]] = 2

        # Sort the array lexicographically
        possible_vertex_states = possible_vertex_states[
            np.lexsort([possible_vertex_states[:, i] for i in range(graph.vertex_count - 1, -1, -1)])]
        #################################################################################################

        # Create possible_edge_states as numpy array
        #################################################################################################
        elements = np.array([1, 2, 3], dtype=np.uint8)
        edge_product = list(itertools.product(elements, repeat=graph.edge_count))
        possible_edge_states = np.array(edge_product, dtype=np.uint8)
        #################################################################################################

        # Create all_states as a single numpy array
        # First determine the dimensions of the final array
        num_vertex_states = possible_vertex_states.shape[0]
        num_edge_states = possible_edge_states.shape[0]
        total_states = num_vertex_states * num_edge_states
        total_elements = graph.vertex_count + graph.edge_count

        # Pre-allocate the full array
        all_states = np.zeros((total_states, total_elements), dtype=np.uint8)

        # Fill the array efficiently using broadcasting
        for i, edge_state in enumerate(possible_edge_states):
            start_idx = i * num_vertex_states
            end_idx = start_idx + num_vertex_states

            # Copy vertex states
            all_states[start_idx:end_idx, :graph.vertex_count] = possible_vertex_states

            # Set edge states (broadcasting to all corresponding vertex state combinations)
            all_states[start_idx:end_idx, graph.vertex_count:] = edge_state

        ##################
        # all_states is a list of lists enumerating ALL the game states
        # NOTE: this might be VERY BIG
        # all_states = [j + list(k) for k in possible_edge_states for j in possible_vertex_states]

        # this next part creates a dictionary where the keys are each of the elements of all_states and the values are
        # lists of all the states connected to the key by an automorphism. Note that different automorphisms can lead
        # to the same state, so at some point the list is converted to a set and then back to a list
        print('generating all_states took', time.time() - start_of_this, 'seconds...')

        start_of_this = time.time()

        all_states_with_symmetries = {}

        # Pre-compute edge transformation lookup table for each automorphism
        edge_transform_lookup = []

        # this outputs edge_transform_lookup, which is a list indexed by automorphism # that shows how the
        # original graph.edge_list maps under the automorphism

        for automorph in list_of_automorphisms:
            edge_map_local = np.zeros(graph.edge_count, dtype=np.int32)
            for edge_idx in range(graph.edge_count):
                first_vertex = automorph[graph.edge_list[edge_idx][0]]
                second_vertex = automorph[graph.edge_list[edge_idx][1]]
                if first_vertex < second_vertex:
                    transformed_edge = (first_vertex, second_vertex)
                else:
                    transformed_edge = (second_vertex, first_vertex)

                transformed_edge_idx = graph.edge_list.index(transformed_edge)
                # edge_map_local[edge_idx] = transformed_edge_idx
                edge_map_local[transformed_edge_idx] = edge_idx
            edge_transform_lookup.append(edge_map_local)

        # iterate over all enumerated states
        for idx, state in tqdm(enumerate(all_states), total=len(all_states)):

            # Create a list for all states connected to state by automorphisms
            list_of_states_connected_by_symmetry = []

            # Find indices of red and blue vertices (more efficient than .index())
            red_vertex_index = np.where(state[:graph.vertex_count] == 1)[0][0]
            blue_vertex_index = np.where(state[:graph.vertex_count] == 2)[0][0]

            # Iterate over all automorphisms
            for i, automorph in enumerate(list_of_automorphisms):
                # Initialize transformed state array
                state_transformed = np.zeros(graph.vertex_count + graph.edge_count, dtype=np.uint8)

                # Transform vertices
                state_transformed[automorph[red_vertex_index]] = 1
                state_transformed[automorph[blue_vertex_index]] = 2

                # Transform edges (using pre-computed lookup)
                state_transformed[graph.vertex_count:] = state[graph.vertex_count + edge_transform_lookup[i]]

                # Convert to string for dictionary key
                # state_transformed is an ndarray;
                # list_of_states_connected_by_symmetry.append(np.array2string(state_transformed, separator=','))
                list_of_states_connected_by_symmetry.append(state_transformed.tobytes())
            all_states_with_symmetries[state.tobytes()] = list(dict.fromkeys(list_of_states_connected_by_symmetry))

        print('generating all_states_no_symmetries took', time.time() - start_of_this, 'seconds...')
        start = time.time()
        sorted_all_states_with_symmetries = dict(sorted(all_states_with_symmetries.items()))
        print('sorting took', time.time() - start, 'seconds...')

        uniques = []

        start = time.time()

        duplicates = set()  # Using a set for O(1) lookups

        for k, v in sorted_all_states_with_symmetries.items():
            if k not in duplicates:  # O(1) lookup in a set
                uniques.append(k)

            # Add all values except the first to duplicates
            # This is more efficient than appending in a loop
            if len(v) > 1:
                duplicates.update(v[1:])

        print('extracting uniques took', time.time() - start, 'seconds...')

        # note: uniques is a list of strings, so don't have to do ast.literal_eval thing
        # unique_terminal_states = [ast.literal_eval(k) for k in uniques]
        print('there are', len(uniques), 'unique terminal states. Writing to disk ...')

        game_states = {}

        reconstructed = ['['+', '.join(map(str, np.frombuffer(each, dtype=np.uint8).reshape((graph.vertex_count + graph.edge_count,)))) + ']' for each in uniques]

        for idx in range(len(reconstructed)):
            game_states[reconstructed[idx]] = {}
            game_states[reconstructed[idx]]['game_state'] = convert_my_game_state_to_erik_game_state(reconstructed[idx], graph.vertex_count, graph.edge_list)
            game_states[reconstructed[idx]]['automorphisms'] = ['['+', '.join(map(str, np.frombuffer(each, dtype=np.uint8).reshape((graph.vertex_count + graph.edge_count,)))) + ']' for each in all_states_with_symmetries[uniques[idx]]]

        with open(os.path.join(data_dir, "graph_" + str(graph_number) + "_unique_terminal_states.pkl"), "wb") as fp:
            pickle.dump(game_states, fp)


def main():

    graph_number = 3   # graph 11 is the RSG/P_3; graph 2 is K_3; graph 3 is K_4

    ################################################################
    # Generate all unique terminal states for the graph_number given
    ################################################################
    # writes out "graph_" + str(graph_number) + "_unique_terminal_states.pkl" to disk
    generate_all_tangled_terminal_states(graph_number=graph_number)

    ##########################################################################################
    # Generates all unique terminal states for the graph_number given for fixed vertex variant
    ##########################################################################################
    # reads "graph_" + str(graph_number) + "_unique_terminal_states.pkl"
    # writes "graph_" + str(graph_number) + "_unique_terminal_states_fixed_vertices.pkl" to disk
    extract_unique_terminal_states_for_fixed_vertices(graph_number=graph_number)


if __name__ == "__main__":
    sys.exit(main())
