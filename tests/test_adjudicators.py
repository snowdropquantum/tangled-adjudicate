import os
import random
import pytest
import traceback
from typing import Dict, List, Tuple, Optional

from snowdrop_tangled_game_engine.tangled_game.game import Game
from snowdrop_tangled_game_engine.tangled_game.game_types import Edge, Vertex

from tangled_adjudicate.adjudicators.lookup_table import LookupTableAdjudicator
from tangled_adjudicate.adjudicators.quantum_annealing import QuantumAnnealingAdjudicator
from tangled_adjudicate.adjudicators.simulated_annealing import SimulatedAnnealingAdjudicator
from tangled_adjudicate.adjudicators.schrodinger import SchrodingerEquationAdjudicator


def play_random_game(graph_id: str = "k_3") -> Dict:
    """
    Create a game with the specified graph_id and play it with random legal moves.
    
    Args:
        graph_id: The ID of the graph to use for the game
        
    Returns:
        The final game state as a dictionary
    """
    # Create a new game
    game = Game()
    game.create_game(graph_id=graph_id, player1_invite="player1", player2_invite="player2")
    
    # Join the game
    game.join_game("player1", 1)
    game.join_game("player2", 2)
    
    # Play the game until it's over
    while not game.is_game_over():
        # Get legal moves for the current player
        current_player_id = game.player1_id if game.current_player_index == 1 else game.player2_id
        legal_moves = game.get_legal_moves(current_player_id)
        
        # Filter out quit moves
        legal_moves = [move for move in legal_moves if move[0] != Game.MoveType.QUIT.value]
        
        # Choose a random move
        if legal_moves:
            move = random.choice(legal_moves)
            game.make_move(current_player_id, move[0], move[1], move[2])
    
    # Return the final game state
    return game.get_game_state()


def test_adjudicators():
    """
    Test all adjudicators with a random game on the k_3 graph.
    """
    # Set random seed for reproducibility
    random.seed(42)
    
    # Play a random game
    game_state = play_random_game(graph_id="k_3")
    
    # Print the final game state for debugging
    print("\nFinal Game State:")
    print(f"Player 1 node: {game_state['player1_node']}")
    print(f"Player 2 node: {game_state['player2_node']}")
    print(f"Edges: {game_state['edges']}")
    
    # Create a temporary directory for lookup table data
    temp_dir = os.path.join(os.path.dirname(__file__), "temp_data")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Test each adjudicator
    adjudicators = [
        (LookupTableAdjudicator(), {"data_dir": temp_dir}),
        (SimulatedAnnealingAdjudicator(), {}),
        (SchrodingerEquationAdjudicator(), {}),
        # (QuantumAnnealingAdjudicator(), {"use_mock": True, "graph_number": 2, "data_dir": temp_dir})
    ]
    
    results = []
    
    for adjudicator, params in adjudicators:
        try:
            # Setup the adjudicator
            adjudicator.setup(**params)
            
            # Adjudicate the game state
            result = adjudicator.adjudicate(game_state)
            
            # Store the result
            results.append(result)
            
            # Print the result
            print(f"\n{result['adjudicator']} Result:")
            print(f"Winner: {result['winner']}")
            if result['score'] is not None:
                print(f"Score: {result['score']}")
            
            # Assert that the result has the expected fields
            assert 'winner' in result, f"{result['adjudicator']} result missing 'winner' field"
            assert 'game_state' in result, f"{result['adjudicator']} result missing 'game_state' field"
            assert 'adjudicator' in result, f"{result['adjudicator']} result missing 'adjudicator' field"
            
        except Exception as e:
            # Print the full traceback
            print(f"\nError in {adjudicator.__class__.__name__}:")
            traceback.print_exc()
            pytest.fail(f"Adjudicator {adjudicator.__class__.__name__} failed: {str(e)}")
    
    # Compare results from different adjudicators
    if len(results) > 1:
        print("\nComparing adjudicator results:")
        for i in range(len(results)):
            for j in range(i+1, len(results)):
                adj1 = results[i]['adjudicator']
                adj2 = results[j]['adjudicator']
                winner1 = results[i]['winner']
                winner2 = results[j]['winner']
                
                print(f"{adj1} vs {adj2}: {winner1} vs {winner2}")
                
                # Note: We don't assert equality because different adjudicators might give different results
                # This is just for information


if __name__ == "__main__":
    test_adjudicators()