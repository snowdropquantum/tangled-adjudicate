# tangled-adjudicate
Tangled is a game designed such that determining who wins is a problem where there is a claim of quantum supremacy. You
can read all about it [here](https://www.snowdropquantum.com/blog/play-the-game-8e7fb).

This repo contains what I call **_adjudicators_**, which are functions that take as input a Tangled game state and 
output the correlation matrix and the [influence vector](https://www.snowdropquantum.com/blog/influence) derived from it
for that game state. In the case when the game state is a terminal state, adjudicators also return the score and the 
outcome of the game (win/loss/draw).

Four different adjudicators are provided:

1. A Schrödinger Equation solver, which is only useful for tiny game graphs
2. A Simulated Annealing solver, whose parameters are chosen to mimic D-Wave hardware 
3. A Quantum Annealing solver that uses D-Wave hardware, and
4. A Lookup Table solver that uses pre-computed results (only for complete graphs on 3 or 4 vertices)

In addition to the adjudicators, I provide several utility functions that support the adjudication process.

## Setup

pip install -e .

should work? Need to check it out!

If not, create an empty conda environment with Python 3.10, pip install dwave-ocean-sdk matplotlib, open a terminal and
do the following to set up the D-Wave stuff (you need a Leap account first for this to work, so get one!):

dwave auth login
dwave setup --auth
dwave config create --auto-token

The full D-Wave setup instructions are [here](https://docs.ocean.dwavesys.com/en/stable/docs_cli.html).

## Tangled Game Graph Specification

A Tangled game graph is specified by a graph number, which label specific graphs included here. I've included ten graphs 
numbered 1 through 10. Each graph requires specification of vertex count (how many vertices the graph has) and an 
explicit edge list, which are included for these ten graphs. If you'd like to add a new graph, it's simple! Just add 
it to the GraphProperties class, found in the /utils/game_graph_properties.py file.

## Tangled Game State Specification: Expected Input Format For Adjudicators

The expected input to the adjudicators in this repo are Tangled game states. These are dictionaries with 
the following keys:

* 'num_nodes': an integer, the number of vertices in the game graph
* 'edges': a list of 3-tuples of length edge_count where the first two elements are indices of the vertices that edge 
connects and the third element is: 0 if the edge is unselected; 1 if the edge is gray (no coupling, J=0); 2 if the edge 
is green (ferromagnetic coupling, J=-1); and 3 if the edge is purple (antiferromagnetic coupling, J=+1)
* 'player1_id' and 'player2_id': strings, specifying the names of the agents
* 'turn_count': an integer, the number of turns that have been played to obtain the game state
* 'current_player_index': an integer, either 1 or 2, corresponding to whose turn it is 
* 'player1_node' and 'player2_node': integers, the vertices owned by the players; if none chosen yet, this is -1

Here's an example:

game_state = {'num_nodes': 6, 'edges': [(0, 1, 1), (0, 2, 1), (0, 3, 2), (0, 4, 3), (0, 5, 2), (1, 2, 1),
(1, 3, 2), (1, 4, 3), (1, 5, 3), (2, 3, 1), (2, 4, 2), (2, 5, 3), (3, 4, 2), (3, 5, 1), (4, 5, 2)],
'player1_id': 'player1', 'player2_id': 'player2', 'turn_count': 17, 'current_player_index': 1,
'player1_node': 1, 'player2_node': 3}

If you want to add a new game graph, it's easy to modify the game state dictionary corresponding to it -- just make 
sure the 'num_nodes' is your vertex count and 'edges' includes an entry for each edge in your graph.

## Adjudication Output

Adjudicators output a dictionary with the following keys:

* 'game_state': a copy of the input game_state dictionary
* 'adjudicator': a string, one of 'simulated_annealing', 'quantum_annealing', 'schrodinger_equation', 'lookup_table'
* 'winner': if both players have chosen vertices, a string, one of 'red', 'blue', 'draw', otherwise None
* 'score': if both players have chosen vertices, the difference in influence scores as a real number, otherwise None
* 'influence_vector': a vector of real numbers of length vertex_count (one real number per vertex in the game graph) if available, otherwise None
* 'correlation_matrix': symmetric real-valued matrix of spin-spin correlations with zeros on diagonals if available, otherwise None
* 'parameters': a copy of the parameters dictionary