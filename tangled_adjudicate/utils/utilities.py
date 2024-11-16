""" a place to put utility functions """
import networkx as nx


def game_state_to_ising_model(game_state):
    # maps edge state to J value 0, 1 => J = 0; 2 => J = -1 FM; 3 => J = +1 AFM
    edge_state_map = {0: 0, 1: 0, 2: -1, 3: 1}

    vertex_count = game_state['num_nodes']
    edge_list = [(each[0], each[1]) for each in game_state['edges']]

    h = {}
    jay = {}

    for k in range(vertex_count):
        h[k] = 0

    for k in range(len(edge_list)):
        jay[edge_list[k]] = edge_state_map[game_state['edges'][k][2]]

    return h, jay


def game_state_is_terminal(game_state):
    # a state is terminal if both players have chosen vertices and all edges have been played
    # game_state = {'num_nodes': 6, 'edges': [(0, 1, 1), (0, 2, 1), (0, 3, 2), (0, 4, 3), (0, 5, 2), (1, 2, 1),
    # (1, 3, 2), (1, 4, 3), (1, 5, 3), (2, 3, 1), (2, 4, 2), (2, 5, 3), (3, 4, 2), (3, 5, 1), (4, 5, 2)],
    # 'player1_id': 'player1', 'player2_id': 'player2', 'turn_count': 17, 'current_player_index': 1,
    # 'player1_node': 1, 'player2_node': 3}

    edge_states = [each[2] for each in game_state['edges']]

    if edge_states.count(0) == 0 and game_state['player1_node'] != -1 and game_state['player2_node'] != -1:
        return True
    else:
        return False


def find_isolated_vertices(n_var, base_jay):

    my_graph = nx.Graph()
    my_graph.add_nodes_from([k for k in range(n_var)])
    my_graph.add_edges_from([k for k, v in base_jay.items() if v != 0])

    # Find isolated vertices (vertices with no edges)
    isolated_vertices = list(nx.isolates(my_graph))

    return isolated_vertices
