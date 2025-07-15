""" a place to put utility functions """
import os
import pickle
import ast
import gdown
import matplotlib.pyplot as plt
import numpy as np

from tangled_adjudicate.utils.game_graph_properties import GraphProperties


def swap_ones_and_twos(lst):
    # Find indices of 1 and 2
    index_of_one = lst.index(1)
    index_of_two = lst.index(2)

    # Swap the values
    lst[index_of_one] = 2
    lst[index_of_two] = 1

    return lst


def evaluate_winner(score, epsilon):
    if score > epsilon:
        winner = 'red'
    elif score < -epsilon:
        winner = 'blue'
    else:
        winner = 'draw'

    return winner


def load_lookup_table(data_dir, graph_number, solver, epsilon, anneal_time, num_reads):
    # returns a dict whose keys are the strings of the game states and values are 'draw', 'red', or 'blue'
    # lookup_table = {'0, 2, 1, 1, 1, 1': 'draw', '1, 0, 2, 1, 1, 1': 'draw'}

    graph = GraphProperties(graph_number)

    raw_file_path = os.path.join(data_dir,
                                 "graph_" + str(graph_number) + "_adjudicated_unique_terminal_states_for_paper.pkl")

    lookup_table_file_path = os.path.join(data_dir,
                                          "graph_" + str(graph_number) + "_" + str(solver) + "_" + str(epsilon) + "_" + str(anneal_time) + "_" + str(num_reads) + ".pkl")

    if os.path.isfile(lookup_table_file_path):
        with open(lookup_table_file_path, "rb") as fp:
            lookup_table = pickle.load(fp)
            return lookup_table   # we are done!

    # in this case, the table still needs to be extracted
    lookup_table = {}

    try:
        with open(raw_file_path, 'rb') as fp:
            raw_adjudication_data = pickle.load(fp)
    except Exception as e:
        raise RuntimeError(f"Failed to load raw adjudication data: {str(e)}")

    # assume game is type 'fixed'
    # we use epsilon here to compute the winner and not use the existing winner
    if solver in ['simulated_annealing']:
        for k, v in raw_adjudication_data['fixed']['simulated_annealing'][num_reads].items():
            lookup_table[k] = evaluate_winner(v['score'], epsilon=epsilon)
    else:
        if solver in ['schrodinger_equation']:
            for k, v in raw_adjudication_data['fixed']['schrodinger_equation'][anneal_time].items():
                lookup_table[k] = evaluate_winner(v['score'], epsilon=epsilon)
        else:
            if solver in ['quantum_annealing']:
                for k, v in raw_adjudication_data['fixed']['quantum_annealing'][num_reads][anneal_time].items():
                    lookup_table[k] = evaluate_winner(v['score'], epsilon=epsilon)
            else:
                print('something went wrong, solver type not found...')

    # For alphazero, there is the canonical board which is where the pieces actually are. As the agent is playing
    # against itself, there's also a concept of switching places of the pieces. However these switched states are not
    # in the data file. I think we swap where the 1 and 2 are in the vertex spots and swap 'winner' from
    # 'red' <--> 'blue' and keep 'draw' the same

    full_results = {}

    for k, v in lookup_table.items():
        key_list = ast.literal_eval(k)
        new_key = swap_ones_and_twos(key_list[:graph.vertex_count]) + key_list[graph.vertex_count:]
        if v == 'draw':
            full_results[str(new_key)] = 'draw'
        else:
            if v == 'red':
                full_results[str(new_key)] = 'blue'
            else:
                if v == 'blue':
                    full_results[str(new_key)] = 'red'
                else:
                    print('something went wrong with the new key adds in build_results_dict...')

    full_results.update(lookup_table)

    with open(lookup_table_file_path, 'wb') as fp:
        pickle.dump(full_results, fp)

    return full_results


def convert_erik_game_state_to_my_game_state(game_state):
    # extract geordie state from erik state
    edge_state_list = [each[2] for each in game_state['edges']]

    vertex_state_list = [0] * game_state['num_nodes']
    if game_state['player1_node'] != -1:
        vertex_state_list[game_state['player1_node']] = 1
    if game_state['player2_node'] != -1:
        vertex_state_list[game_state['player2_node']] = 2

    my_state = vertex_state_list + edge_state_list

    return my_state


def convert_my_game_state_to_erik_game_state(my_state, number_of_vertices, list_of_edge_tuples):
    # extract erik state from geordie state
    if isinstance(my_state, str):
        my_state = ast.literal_eval(my_state)

    my_vertices = my_state[:number_of_vertices]
    my_edges = my_state[number_of_vertices:]

    turn_count = 0

    try:
        player_1_vertex = my_vertices.index(1)
        turn_count += 1
    except ValueError:
        player_1_vertex = -1

    try:
        player_2_vertex = my_vertices.index(2)
        turn_count += 1
    except ValueError:
        player_2_vertex = -1

    turn_count += my_edges.count(1) + my_edges.count(2) + my_edges.count(3)

    # if turn_count is even, it's player 1 (red)'s turn
    current_player_idx = 1 if turn_count % 2 == 0 else 2

    erik_edges = [(list_of_edge_tuples[k][0], list_of_edge_tuples[k][1], my_edges[k]) for k in range(len(my_edges))]

    game_state = {'num_nodes': number_of_vertices,
                  'edges': erik_edges,
                  'player1_id': 'player1',
                  'player2_id': 'player2',
                  'turn_count': turn_count,
                  'current_player_index': current_player_idx,
                  'player1_node': player_1_vertex,
                  'player2_node': player_2_vertex}

    return game_state


def plots_for_paper(graph_number, anneal_time, energies, probabilities_in_z_basis, sorted_indices_energies, use_zoom=True):
    # generates plots for paper where top figure is terminal state count vs score, next is zoomed in same thing
    # (optional), middle is schrodinger eigenenergies vs s, and bottom is schrodinger probabilities vs s

    # read in adjudication results -- need to run compare_adjudication_results.py first to generate
    data_dir = os.path.join(os.getcwd(), '..', 'data')
    with open(os.path.join(data_dir, "graph_" + str(graph_number) + "_data_for_plots.pkl"), "rb") as fp:
        terminal_state_data = pickle.load(fp)

    to_plot = []
    for k, v in terminal_state_data.items():
        to_plot.append(v)

    if use_zoom:
        num_plots = 4
    else:
        num_plots = 3

    plt.rcParams.update({
        'font.size': 16,  # Default font size
        'axes.labelsize': 16,  # X and Y labels
        'xtick.labelsize': 16,  # X tick labels
        'ytick.labelsize': 16,  # Y tick labels
        'legend.fontsize': 14  # Legend
    })

    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4*num_plots))

    # data for score plots

    graph_data = {11: {'name': '$P_3$ Path Graph', 'top_val': 15, 'width': 2, 'bin_count': 101, 'text_step': 0.5, 'first_offset': 1,
                       'lr_offset': 1, 'prob_y_lim': 0.8, 'num_vert': 3, 'draw_boundary': 0.5, 'y_offset': [2.5, 6]},
                  2: {'name': '$K_3$', 'top_val': 50, 'width': 2, 'bin_count': 101, 'text_step': 1.5, 'first_offset': 4,
                      'lr_offset': 1, 'prob_y_lim': 0.25, 'num_vert': 3, 'draw_boundary': 0.5, 'y_offset': [4.5, 8]},
                  20: {'name': 'Diamond Graph', 'top_val': 250, 'width': 4, 'bin_count': 201, 'text_step': 7.5, 'first_offset': 20,
                       'lr_offset': 2, 'prob_y_lim': 0.3, 'num_vert': 4, 'draw_boundary': 0.5, 'y_offset': [1, 1],
                       'zoom': {'top_val': 10, 'width': 2, 'bin_count': 101, 'text_step': 7.5, 'first_offset': 20, 'lr_offset': 2}},
                  3: {'name': '$K_4$', 'top_val': 800, 'width': 4, 'bin_count': 401, 'text_step': 25, 'first_offset': 70,
                      'lr_offset': 2, 'prob_y_lim': 1, 'num_vert': 4, 'draw_boundary': 0.5, 'y_offset': [2.5, 6],
                      'zoom': {'top_val': 80, 'width': 4, 'bin_count': 401, 'text_step': 2, 'first_offset': 7, 'lr_offset': 2}},
                  19: {'name': 'Barbell', 'top_val': 2500, 'width': 8, 'bin_count': 201, 'text_step': 6, 'first_offset': 12, 'lr_offset': 0.4,
                       'prob_y_lim': 0.2, 'num_vert': 6, 'draw_boundary': 0.25, 'y_offset': [2.5, 6],
                       'zoom': {'top_val': 100, 'width': 1, 'bin_count': 51, 'text_step': 70, 'first_offset': 150, 'lr_offset': 4}},
                  18: {'name': '3-Prism', 'top_val': 20000, 'width': 8, 'bin_count': 801, 'text_step': 500, 'first_offset': 2000,
                       'lr_offset': 4, 'prob_y_lim': 1, 'num_vert': 6, 'draw_boundary': 0.5, 'y_offset': [2.5, 6],
                       'zoom': {'top_val': 1000, 'width': 1, 'bin_count': 201, 'text_step': 25, 'first_offset': 100, 'lr_offset': 0.4}},}

    binary_labels = [
        r'$|\!\!' + format(i, '0' + str(graph_data[graph_number]['num_vert']) + 'b').replace('0', r'\uparrow\!\!\!\!').replace(
            '1', r'\downarrow\!\!\!\!') + r'\; \rangle$'
        for i in range(len(probabilities_in_z_basis[-1]))]

    col = ['gray', 'cyan', 'orange']

    axis_to_use = 0
    axes[axis_to_use].hist(to_plot,
                           range=[-graph_data[graph_number]['width'], graph_data[graph_number]['width']],
                           bins=graph_data[graph_number]['bin_count'],
                           color=col,
                           stacked=True,
                           alpha=0.7,
                           label=['Simulated Annealing', 'Schrodinger Equation', 'Advantage2.1.3'])

    axes[axis_to_use].set_ylabel('Terminal State Count')
    axes[axis_to_use].set_xlabel('Score')
    axes[axis_to_use].grid(True, alpha=0.3)
    axes[axis_to_use].vlines(x=graph_data[graph_number]['draw_boundary'], ymin=0, ymax=graph_data[graph_number]['top_val'],
                             colors='green', ls=':', lw=1)
    axes[axis_to_use].vlines(x=-graph_data[graph_number]['draw_boundary'], ymin=0, ymax=graph_data[graph_number]['top_val'],
                             colors='green', ls=':', lw=1)

    axes[axis_to_use].set_ylim([0, graph_data[graph_number]['top_val']])
    axes[axis_to_use].legend()
    axis_to_use += 1

    if use_zoom:
        axes[axis_to_use].hist(to_plot,
                               range=[-graph_data[graph_number]['zoom']['width'], graph_data[graph_number]['zoom']['width']],
                               bins=graph_data[graph_number]['zoom']['bin_count'],
                               color=col,
                               stacked=True,
                               alpha=0.7)

        axes[axis_to_use].set_ylabel('Terminal State Count')
        axes[axis_to_use].set_xlabel('Score')
        axes[axis_to_use].grid(True, alpha=0.3)
        axes[axis_to_use].vlines(x=graph_data[graph_number]['draw_boundary'], ymin=0,
                                 ymax=graph_data[graph_number]['zoom']['top_val'],
                                 colors='green', ls=':', lw=1)
        axes[axis_to_use].vlines(x=-graph_data[graph_number]['draw_boundary'], ymin=0,
                                 ymax=graph_data[graph_number]['zoom']['top_val'],
                                 colors='green', ls=':', lw=1)

        axes[axis_to_use].set_ylim([0, graph_data[graph_number]['zoom']['top_val']])
        axis_to_use += 1

    # process data for energy plot

    # Flatten the data and assign rank-based colors
    s_expanded = []
    E_flattened = []
    colors = []

    for i, energy_list in enumerate(energies):
        # Sort energies and get their ranks
        sorted_indices = np.argsort(energy_list)  # Indices that would sort the array
        ranks = np.argsort(sorted_indices)  # Rank of each element (0=lowest, 1=second lowest, etc.)

        # Extend the lists
        s_expanded.extend([anneal_time[i]] * len(energy_list))
        E_flattened.extend(energy_list)
        colors.extend(ranks)

    axes[axis_to_use].scatter(s_expanded, E_flattened,
                              c=colors,  # Color by rank
                              s=1,  # Small point size
                              alpha=0.8,
                              cmap='tab10')  # Discrete colormap

    axes[axis_to_use].set_ylabel('Energy [GHz]')
    # axes[axis_to_use].set_xlabel(r'$s = t/t_a$')
    axes[axis_to_use].grid(True, alpha=0.3)

    # Add labels with collision detection
    last_s = anneal_time[-1]
    base_x_offset = 0.1   # 0.08, 0.09
    y_range = np.max(energies[-1]) - np.min(energies[-1])
    y_threshold = 0.05 * y_range  # Define "close" as within 5% of y-range

    used_y_values = []  # Keep track of previously used y-values

    for j in range(energies[-1].size):
        last_energy = energies[-1][j]

        # Check if current last_prob is close to any previously used y-value
        x_offset_multiplier = 0
        for used_y in used_y_values:
            if abs(last_energy - used_y) < y_threshold:
                x_offset_multiplier += 1

        current_x_offset = base_x_offset * x_offset_multiplier
        if last_energy < -5:
            y_offset = 0.01 * y_range + graph_data[graph_number]['y_offset'][0]
        else:
            if last_energy > 5:
                y_offset = 0.01 * y_range - graph_data[graph_number]['y_offset'][1]
            else:
                y_offset = 0.01 * y_range

        # axes[axis_to_use].text(last_s - current_x_offset,
        #                        last_energy + y_offset,
        #                        binary_labels[sorted_indices_energies[j]],
        #                        fontsize=8,
        #                            ha='right',
        #                            va='bottom')

        # Add current y-value to the list of used values
        used_y_values.append(last_energy)

    # probabilities plot
    axis_to_use += 1
    prob_array = np.array(probabilities_in_z_basis)

    for j in range(prob_array.shape[1]):
        cols = [prob_array.shape[1]-1-j]*len(anneal_time)
        axes[axis_to_use].scatter(anneal_time, prob_array[:, j], c=cols, s=1, alpha=0.7, cmap='tab10', vmin=0, vmax=prob_array.shape[1]-1)

    # Add labels with collision detection
    last_s = anneal_time[-1]

    y_range = np.max(prob_array) - np.min(prob_array)
    y_threshold = 0.05 * y_range  # Define "close" as within 5% of y-range

    used_y_values = []  # Keep track of previously used y-values

    for j in range(prob_array.shape[1]):
        last_prob = probabilities_in_z_basis[-1][j]

        # Check if current last_prob is close to any previously used y-value
        x_offset_multiplier = 0
        for used_y in used_y_values:
            if abs(last_prob - used_y) < y_threshold:
                x_offset_multiplier += 1

        current_x_offset = base_x_offset * x_offset_multiplier
        y_offset = 0.025 * y_range

        if last_prob > 0.01:
            axes[axis_to_use].text(last_s - current_x_offset,
                                   last_prob + y_offset,
                                   binary_labels[j],
                                   fontsize=12,
                                   ha='right',
                                   va='bottom')

        # Add current y-value to the list of used values
        used_y_values.append(last_prob)

    axes[axis_to_use].set_ylim([0, graph_data[graph_number]['prob_y_lim']])

    axes[axis_to_use].set_ylabel('Probabilities')
    axes[axis_to_use].set_xlabel(r'$s = t/t_a$')
    axes[axis_to_use].grid(True, alpha=0.3)

    # Share x-axis between last two plots
    axes[axis_to_use].sharex(axes[axis_to_use-1])
    axes[axis_to_use-1].tick_params(labelbottom=False)

    fig.align_ylabels(axes)
    plt.tight_layout()
    plt.show()


# crufty plots

# def plot_energies(s, energies):
#     # plots energies as a function of anneal parameter s
#
#     plt.rcParams.update({
#         'font.size': 24,  # Default font size
#         'axes.labelsize': 24,  # X and Y labels
#         'axes.titlesize': 24,  # Title
#         'xtick.labelsize': 16,  # X tick labels
#         'ytick.labelsize': 16,  # Y tick labels
#         'legend.fontsize': 16  # Legend
#     })
#
#     # Flatten the data and assign rank-based colors
#     s_expanded = []
#     E_flattened = []
#     colors = []
#
#     for i, energy_list in enumerate(energies):
#         # Sort energies and get their ranks
#         sorted_indices = np.argsort(energy_list)  # Indices that would sort the array
#         ranks = np.argsort(sorted_indices)  # Rank of each element (0=lowest, 1=second lowest, etc.)
#
#         # Extend the lists
#         s_expanded.extend([s[i]] * len(energy_list))
#         E_flattened.extend(energy_list)
#         colors.extend(ranks)
#
#     plt.figure(figsize=(10, 3))
#     scatter = plt.scatter(s_expanded, E_flattened,
#                           c=colors,  # Color by rank
#                           s=1,  # Small point size
#                           alpha=0.8,
#                           cmap='tab10')  # Discrete colormap
#
#     # plt.colorbar(scatter, label='Energy Rank (0=lowest)')
#     plt.xlabel(r'$s = t/t_a$')
#     plt.ylabel('Energy [GHz]')
#     # plt.title('Energy vs s (colored by rank within each s)')
#     plt.tight_layout()
#     plt.show()
#
#
# def plot_probabilities(anneal_time, probabilities_in_z_basis):
#     # plots probabilities in Z basis as a function of anneal parameter s
#
#     plt.rcParams.update({
#         'font.size': 24,  # Default font size
#         'axes.labelsize': 24,  # X and Y labels
#         'axes.titlesize': 24,  # Title
#         'xtick.labelsize': 16,  # X tick labels
#         'ytick.labelsize': 16,  # Y tick labels
#         'legend.fontsize': 16  # Legend
#     })
#
#     prob_array = np.array(probabilities_in_z_basis)
#
#     plt.figure(figsize=(10, 4))
#
#     for j in range(prob_array.shape[1]):
#         plt.scatter(anneal_time, prob_array[:, j], s=1, alpha=0.7)
#
#     # Add labels with collision detection
#     last_s = anneal_time[-1]
#     base_x_offset = 0.15
#     y_range = np.max(prob_array) - np.min(prob_array)
#     y_threshold = 0.05 * y_range  # Define "close" as within 5% of y-range
#
#     used_y_values = []  # Keep track of previously used y-values
#     #
#     # binary_labels = ['|' + format(i, '04b').replace('0', '↑').replace('1', '↓') + '>' for i in range(len(probabilities_in_z_basis[-1]))]
#
#     for j in range(prob_array.shape[1]):
#         last_prob = probabilities_in_z_basis[-1][j]
#
#         # Check if current last_prob is close to any previously used y-value
#         x_offset_multiplier = 0
#         for used_y in used_y_values:
#             if abs(last_prob - used_y) < y_threshold:
#                 x_offset_multiplier += 1
#
#         current_x_offset = base_x_offset * x_offset_multiplier
#         y_offset = 0.025 * y_range
#
#         if last_prob > 0.01:
#             plt.text(last_s - current_x_offset,
#                      last_prob + y_offset,
#                      binary_labels[j],
#                      fontsize=16,
#                      ha='right',
#                      va='bottom')
#
#         # Add current y-value to the list of used values
#         used_y_values.append(last_prob)
#
#     plt.ylim([0, 0.3])
#     plt.xlabel('$s=t/t_a$')
#     plt.ylabel('Probabilities')
#     plt.tight_layout()
#     plt.grid(True, alpha=0.3)
#
#     plt.show()