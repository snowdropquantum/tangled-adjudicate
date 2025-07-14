from typing import Dict, Any
import numpy as np
from ..schrodinger.schrodinger_functions import evolve_schrodinger

from .adjudicator import Adjudicator, GameState, AdjudicationResult


class SchrodingerEquationAdjudicator(Adjudicator):
    """Adjudicator implementation using Schrödinger equation evolution."""
    
    def __init__(self) -> None:
        """Initialize the adjudicator with default values."""
        super().__init__()
        self.anneal_time: float = 40.0  # ns
        self.s_min: float = 0.001
        self.s_max: float = 0.999
        
    def setup(self, **kwargs) -> None:
        """Configure the Schrödinger equation parameters.
        
        Args:
            anneal_time: Annealing time in nanoseconds (default: 40.0)
            s_min: Minimum annealing parameter (default: 0.001)
            s_max: Maximum annealing parameter (default: 0.999)
            
        Raises:
            ValueError: If parameters are invalid
        """
        if 'anneal_time' in kwargs:
            if not isinstance(kwargs['anneal_time'], (int, float)) or kwargs['anneal_time'] <= 0:
                raise ValueError("anneal_time must be a positive number")
            self.anneal_time = float(kwargs['anneal_time'])
            
        if 's_min' in kwargs:
            if not isinstance(kwargs['s_min'], (int, float)) or not 0 <= kwargs['s_min'] < 1:
                raise ValueError("s_min must be in [0, 1)")
            self.s_min = float(kwargs['s_min'])
            
        if 's_max' in kwargs:
            if not isinstance(kwargs['s_max'], (int, float)) or not 0 < kwargs['s_max'] <= 1:
                raise ValueError("s_max must be in (0, 1]")
            self.s_max = float(kwargs['s_max'])
            
        if self.s_min >= self.s_max:
            raise ValueError("s_min must be less than s_max")
            
        self._parameters = {
            'anneal_time': self.anneal_time,
            's_min': self.s_min,
            's_max': self.s_max
        }
        
    def adjudicate(self, game_state: GameState) -> AdjudicationResult:
        """Adjudicate the game state using Schrödinger equation evolution.
        
        Args:
            game_state: The current game state
            
        Returns:
            AdjudicationResult containing the adjudication details
            
        Raises:
            ValueError: If the game state is invalid
        """
        self._validate_game_state(game_state)
        
        # Convert game state to Ising model
        ising_model = self._game_state_to_ising(game_state)
        
        # Evolve Schrödinger equation
        correlation_matrix = evolve_schrodinger(
            ising_model['h'],
            ising_model['j'],
            s_min=self.s_min,
            s_max=self.s_max,
            tf=self.anneal_time,
            n_qubits=game_state['num_nodes']
        )
        
        # Make symmetric (evolve_schrodinger returns upper triangular)
        correlation_matrix = correlation_matrix + correlation_matrix.T
        
        # Handle isolated vertices
        isolated_vertices = self._find_isolated_vertices(game_state)
        if isolated_vertices:
            for vertex in isolated_vertices:
                correlation_matrix[:, vertex] = 0
                correlation_matrix[vertex, :] = 0
        
        # Compute results
        winner, score, influence_vector = self._compute_winner_score_and_influence(
            game_state, correlation_matrix
        )
        
        return AdjudicationResult(
            game_state=game_state,
            adjudicator='schrodinger_equation',
            winner=winner,
            score=score,
            influence_vector=influence_vector,
            correlation_matrix=correlation_matrix,
            parameters=self._parameters
        )
    