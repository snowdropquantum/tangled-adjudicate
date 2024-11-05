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
