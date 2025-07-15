import os
import pickle
from typing import Dict, Optional
import numpy as np

from ..utils.utilities import convert_erik_game_state_to_my_game_state, load_lookup_table
from .adjudicator import Adjudicator, GameState, AdjudicationResult


class LookupTableAdjudicator(Adjudicator):
    """Adjudicator implementation using pre-computed lookup tables"""
    
    def __init__(self) -> None:
        """Initialize the lookup table adjudicator"""
        super().__init__()
        self.data_dir: Optional[str] = None
        self.lookup_table: Optional[Dict[str, str]] = None
        self.solver: Optional[str] = None
        self.epsilon: Optional[float] = None
        self.anneal_time: Optional[int] = None
        self.num_reads: Optional[int] = None
        self.graph_number: Optional[int] = None

    def setup(self, **kwargs) -> None:
        """Configure lookup table parameters.
        
        Args:
            data_dir: Directory containing lookup table data files
            lookup_args: Dictionary containing solver, epsilon, anneal_time and num_reads for lookup table
            graph_number: int
        Raises:
            ValueError: If parameters are invalid or data directory doesn't exist
        """
        if 'data_dir' in kwargs:
            if not isinstance(kwargs['data_dir'], str):
                raise ValueError("data_dir must be a string")
            if not os.path.isdir(kwargs['data_dir']):
                raise ValueError(f"Directory not found: {kwargs['data_dir']}")
            self.data_dir = kwargs['data_dir']

        if 'lookup_args' in kwargs:
            if kwargs['lookup_args']['solver'] not in ['simulated_annealing', 'schrodinger_equation', 'quantum_annealing']:
                raise ValueError("solver must be one of 'simulated_annealing', 'schrodinger_equation', 'quantum_annealing'")
            self.solver = kwargs['lookup_args']['solver']
            if not isinstance(kwargs['lookup_args']['epsilon'], float):
                raise ValueError("epsilon must be a float")
            self.epsilon = kwargs['lookup_args']['epsilon']
            if not isinstance(kwargs['lookup_args']['anneal_time'], int):
                raise ValueError("anneal_time must be an int")
            self.anneal_time = kwargs['lookup_args']['anneal_time']
            if not isinstance(kwargs['lookup_args']['num_reads'], int):
                raise ValueError("num_reads must be an int")
            self.num_reads = kwargs['lookup_args']['num_reads']

        if 'graph_number' in kwargs:
            if not isinstance(kwargs['graph_number'], int):
                raise ValueError("graph_number must be an int")
            self.graph_number = kwargs['graph_number']

        self._parameters = {'data_dir': self.data_dir,
                            'solver': self.solver,
                            'epsilon': self.epsilon,
                            'anneal_time': self.anneal_time,
                            'num_reads': self.num_reads,
                            'graph_number': self.graph_number}

    def _get_lookup_table(self) -> None:
        """Load the appropriate lookup table for the given graph_number and solver_used

        Raises:
            RuntimeError: If lookup table file cannot be loaded
        """
        if not self.data_dir:
            raise RuntimeError("Data directory not set. Call setup() first.")

        self.lookup_table = load_lookup_table(data_dir=self.data_dir, graph_number=self.graph_number,
                                              solver=self.solver, epsilon=self.epsilon,
                                              anneal_time=self.anneal_time, num_reads=self.num_reads)

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
        if self.lookup_table is None:
            self._get_lookup_table()
            
        if not self.lookup_table:
            raise RuntimeError("Failed to load lookup table")
            
        # Convert game state to lookup format
        lookup_state = convert_erik_game_state_to_my_game_state(game_state)
        
        try:
            winner = self.lookup_table[str(lookup_state)]
        except KeyError:   # key not in results_dict
            raise RuntimeError("key not in lookup table...")

        return AdjudicationResult(
            game_state=game_state,
            adjudicator='lookup_table',
            winner=winner,
            score=None,                 # Lookup table doesn't provide scores; note though all these are
            influence_vector=None,      # available if we need them from the raw data
            correlation_matrix=None,
            parameters=self._parameters
        )
    