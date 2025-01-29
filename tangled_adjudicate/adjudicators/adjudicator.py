from abc import ABC, abstractmethod
from typing import Any, TypedDict, List, Tuple, Optional, Dict, Union, Set
import numpy as np
import numpy.typing as npt


class GameState(TypedDict):
    num_nodes: int
    edges: List[Tuple[int, int, int]]  # (node1, node2, weight)
    player1_id: str
    player2_id: str
    turn_count: int
    current_player_index: int
    player1_node: Optional[int]
    player2_node: Optional[int]


class AdjudicationResult(TypedDict):
    game_state: GameState
    adjudicator: str
    winner: Optional[str]  # 'red', 'blue', 'draw', or None
    score: Optional[float]
    influence_vector: Optional[npt.NDArray[np.float64]]
    correlation_matrix: Optional[npt.NDArray[np.float64]]
    parameters: Dict[str, Union[str, int, float, bool]]


class IsingModel(TypedDict):
    h: Dict[int, float]  # Local fields
    j: Dict[Tuple[int, int], float]  # Coupling strengths


class Adjudicator(ABC):
    """Base interface for game state adjudication implementations."""
    
    def __init__(self) -> None:
        """Initialize base adjudicator."""
        self._parameters: Dict[str, Any] = {}
        self.j_map = {0: 0.0,       # edge (i, j) uncolored , J_ij=0
                      1: 0.0,       # edge (i, j) colored gray, J_ij=0
                      2: -1.0,      # edge (i, j) colored green, FM coupling, J_ij=-1.0
                      3: 1.0}       # edge (i, j) colored purple, AFM coupling, J_ij=+1.0
    
    @abstractmethod
    def setup(self, **kwargs) -> None:
        """Optional setup method for implementation-specific initialization."""
        pass
    
    @abstractmethod
    def adjudicate(self, game_state: GameState) -> AdjudicationResult:
        """Adjudicate the given game state."""
        pass

    def _validate_game_state(self, game_state: GameState) -> None:
        """Validate the game state structure and contents."""
        required_keys = {
            'num_nodes', 'edges', 'player1_id', 'player2_id',
            'turn_count', 'current_player_index', 'player1_node', 'player2_node'
        }
        
        if not all(key in game_state for key in required_keys):
            missing_keys = required_keys - set(game_state.keys())
            raise ValueError(f"Game state missing required keys: {missing_keys}")
            
        if game_state['num_nodes'] < 1:
            raise ValueError("Number of nodes must be positive")
            
        for edge in game_state['edges']:
            if len(edge) != 3:
                raise ValueError(f"Invalid edge format: {edge}")
            if not (0 <= edge[0] < game_state['num_nodes'] and 
                   0 <= edge[1] < game_state['num_nodes']):
                raise ValueError(f"Edge vertices out of range: {edge}")

    def _game_state_to_ising(self, game_state: GameState) -> IsingModel:
        """Convert game state to Ising model parameters.
        
        Args:
            game_state: The current game state
            
        Returns:
            IsingModel containing h (local fields) and j (coupling strengths)
        """
        h = {i: 0.0 for i in range(game_state['num_nodes'])}
        j = {}
        
        for edge in game_state['edges']:
            v1, v2, edge_label = edge
            if v1 > v2:
                v1, v2 = v2, v1
            j[(v1, v2)] = float(self.j_map[edge_label])
            
        return IsingModel(h=h, j=j)

    def _find_isolated_vertices(self, game_state: GameState) -> Set[int]:
        """Find vertices with no connections in the graph.
        
        Args:
            game_state: The current game state
            
        Returns:
            Set of isolated vertex indices
        """
        connected_vertices = set()
        for edge in game_state['edges']:
            connected_vertices.add(edge[0])
            connected_vertices.add(edge[1])
            
        all_vertices = set(range(game_state['num_nodes']))
        return all_vertices - connected_vertices

    def _compute_winner_score_and_influence(
        self,
        game_state: GameState,
        correlation_matrix: npt.NDArray[np.float64],
        epsilon: float = 1e-6
    ) -> Tuple[Optional[str], Optional[float], npt.NDArray[np.float64]]:
        """Compute winner, score and influence from correlation matrix."""
        if not isinstance(correlation_matrix, np.ndarray):
            raise ValueError("Correlation matrix must be a numpy array")
            
        if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
            raise ValueError("Correlation matrix must be square")
            
        if correlation_matrix.shape[0] != game_state['num_nodes']:
            raise ValueError("Correlation matrix size must match number of nodes")
            
        influence_vector = np.sum(correlation_matrix, axis=0)
        
        if game_state['player1_node'] is None or game_state['player2_node'] is None:
            return None, None, influence_vector
            
        score = (influence_vector[game_state['player1_node']] - 
                influence_vector[game_state['player2_node']])
        
        if score > epsilon:
            winner = 'red'
        elif score < -epsilon:
            winner = 'blue'
        else:
            winner = 'draw'
            
        return winner, score, influence_vector
