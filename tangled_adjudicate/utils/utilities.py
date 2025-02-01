""" a place to put utility functions """
import gdown
import networkx as nx


def get_tso(graph_number, file_path):
    # get terminal state outcomes
    if graph_number == 2:
        tso_url = 'https://drive.google.com/uc?id=14NcsNuPmHA4fE-Q5Wr4XVRxbYPOTcVNP'
        gdown.download(tso_url, file_path, quiet=False)

    if graph_number == 3:
        tso_url = 'https://drive.google.com/uc?id=1Ob09q0WOHZp4gRd-A5h6af0TNaA3Ek6A'
        gdown.download(tso_url, file_path, quiet=False)


def build_results_dict(results):
    # data is a list of elements like [[[0, 2, 1, 1, 1, 1], 'draw'], [[1, 0, 2, 1, 1, 1], 'draw']]
    # the full data for 3-vertex 3-edge Tangled are in
    # (os.path.join(os.getcwd(), '..', 'data', "three_vertex_terminal_state_outcomes_test.txt")
    results_dict = {}
    for each in results:
        results_dict[str(each[0])] = each[1]
    return results_dict


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
                  # 'edges': [(0, 1, 3), (0, 2, 1), (0, 3, 3), (1, 2, 1), (1, 3, 3), (2, 3, 1)],
                  'edges': erik_edges,
                  'player1_id': 'player1', 'player2_id': 'player2', 'turn_count': turn_count,
                  'current_player_index': current_player_idx, 'player1_node': player_1_vertex,
                  'player2_node': player_2_vertex}

    return game_state
