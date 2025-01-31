import os
import pickle
from typing import Dict, Optional
import numpy as np

from ..utils.utilities import (
    convert_erik_game_state_to_my_game_state,
    get_tso,
    build_results_dict
)
from .adjudicator import Adjudicator, GameState, AdjudicationResult


class LookupTableAdjudicator(Adjudicator):
    """Adjudicator implementation using pre-computed lookup tables."""
    
    def __init__(self) -> None:
        """Initialize the lookup table adjudicator."""
        super().__init__()
        self.data_dir: Optional[str] = None
        self.results_dict: Optional[Dict[str, str]] = None
        
    def setup(self, **kwargs) -> None:
        """Configure lookup table parameters.
        
        Args:
            data_dir: Directory containing lookup table data files
            
        Raises:
            ValueError: If parameters are invalid or data directory doesn't exist
        """
        if 'data_dir' in kwargs:
            if not isinstance(kwargs['data_dir'], str):
                raise ValueError("data_dir must be a string")
            if not os.path.isdir(kwargs['data_dir']):
                raise ValueError(f"Directory not found: {kwargs['data_dir']}")
            self.data_dir = kwargs['data_dir']
            
        self._parameters = {'data_dir': self.data_dir}
        
    def _load_lookup_table(self, num_nodes: int) -> None:
        """Load the appropriate lookup table for the given graph size.
        
        Args:
            num_nodes: Number of nodes in the graph
            
        Raises:
            ValueError: If lookup table is not available for this graph size
            RuntimeError: If lookup table file cannot be loaded
        """
        if num_nodes not in [3, 4]:
            raise ValueError(
                "Lookup table only available for complete graphs with 3 or 4 vertices"
            )
            
        if not self.data_dir:
            raise RuntimeError("Data directory not set. Call setup() first.")
            
        graph_number = num_nodes - 1  # Convert from num_nodes to graph_number
        file_path = os.path.join(
            self.data_dir,
            f'graph_{graph_number}_terminal_state_outcomes.pkl'
        )
        
        # Generate lookup table if it doesn't exist
        if not os.path.exists(file_path):
            get_tso(graph_number, file_path)
            
        try:
            with open(file_path, 'rb') as fp:
                results = pickle.load(fp)
            self.results_dict = build_results_dict(results)
        except Exception as e:
            raise RuntimeError(f"Failed to load lookup table: {str(e)}")
        
    def adjudicate(self, game_state: GameState) -> AdjudicationResult:
        """Adjudicate the game state using the lookup table.
        
        Args:
            game_state: The current game state
            
        Returns:
            AdjudicationResult containing the adjudication details
            
        Raises:
            ValueError: If the game state is invalid or unsupported
            RuntimeError: If lookup table is not loaded
        """
        self._validate_game_state(game_state)
        
        # Load lookup table if needed
        if (self.results_dict is None or len(next(iter(self.results_dict.keys()))) != game_state['num_nodes']):
            self._load_lookup_table(game_state['num_nodes'])
            
        if not self.results_dict:
            raise RuntimeError("Failed to load lookup table")
            
        # Convert game state to lookup format
        lookup_state = convert_erik_game_state_to_my_game_state(game_state)
        
        try:
            winner = self.results_dict[str(lookup_state)]
        except KeyError:
            raise ValueError(
                f"Game state not found in lookup table: {lookup_state}"
            )
            
        return AdjudicationResult(
            game_state=game_state,
            adjudicator='lookup_table',
            winner=winner,
            score=None,  # Lookup table doesn't provide scores
            influence_vector=None,
            correlation_matrix=None,
            parameters=self._parameters
        )
    