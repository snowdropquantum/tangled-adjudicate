from typing import Dict, Any, List, Optional
import numpy as np
from dataclasses import dataclass
from dwave.system import DWaveSampler, FixedEmbeddingComposite
from dwave.system.testing import MockDWaveSampler

from ..utils.find_graph_automorphisms import get_automorphisms
from ..utils.find_hardware_embeddings import get_embeddings
from .adjudicator import Adjudicator, GameState, AdjudicationResult

@dataclass
class QAParameters:
    """Parameters for quantum annealing."""
    num_reads: int = 1000
    anneal_time: float = 5.0  # ns
    num_chip_runs: int = 1
    use_gauge_transform: bool = True
    use_shim: bool = False
    shim_iterations: int = 1
    alpha_phi: float = 0.1
    use_mock: bool = False
    solver_name: Optional[str] = None
    graph_number: Optional[int] = None

class QuantumAnnealingAdjudicator(Adjudicator):
    """Adjudicator implementation using D-Wave quantum annealing."""
    
    def __init__(self) -> None:
        """Initialize the quantum annealing adjudicator."""
        super().__init__()
        self.params = QAParameters()
        self.embeddings: List[List[int]] = []
        self.automorphisms: List[Dict[int, int]] = []
        self.sampler: Optional[FixedEmbeddingComposite] = None
        
    def setup(self, **kwargs) -> None:
        """Configure quantum annealing parameters and initialize D-Wave connection.
        
        Args:
            num_reads: Number of annealing reads per run
            anneal_time: Annealing time in nanoseconds
            num_chip_runs: Number of separate chip programming runs
            use_gauge_transform: Whether to apply gauge transformations
            use_shim: Whether to use shimming process
            shim_iterations: Number of shimming iterations if shimming is used
            alpha_phi: Learning rate for flux bias offsets
            use_mock: Whether to use mock D-Wave sampler (for testing)
            solver_name: Name of D-Wave solver to use
            graph_number: Graph number for embedding lookup
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If D-Wave connection fails
        """
        # Update parameters from kwargs
        for key, value in kwargs.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        
        # Validate parameters
        if self.params.num_reads <= 0:
            raise ValueError("num_reads must be positive")
        if self.params.anneal_time <= 0:
            raise ValueError("anneal_time must be positive")
        if self.params.num_chip_runs <= 0:
            raise ValueError("num_chip_runs must be positive")
        if self.params.shim_iterations <= 0:
            raise ValueError("shim_iterations must be positive")
        if self.params.alpha_phi <= 0 or self.params.alpha_phi > 1:
            raise ValueError("alpha_phi must be in (0, 1]")
        
        # Get graph-specific data if graph number provided
        if self.params.graph_number is not None:
            self.automorphisms = get_automorphisms(self.params.graph_number)
            self.embeddings = get_embeddings(
                self.params.graph_number,
                self.params.solver_name
            )
            
        # Initialize sampler
        try:
            if self.params.use_mock:
                base_sampler = MockDWaveSampler(
                    topology_type='zephyr',
                    topology_shape=[6, 4]
                )
            else:
                base_sampler = DWaveSampler(solver=self.params.solver_name)
                
            # Store for later use in adjudicate
            self._base_sampler = base_sampler
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize D-Wave sampler: {str(e)}")
            
        # Store parameters
        self._parameters = self.params.__dict__
        
    def _apply_gauge_transform(
        self,
        samples: np.ndarray,
        flip_indices: List[int]
    ) -> np.ndarray:
        """Apply gauge transformation to samples.
        
        Args:
            samples: Sample array to transform
            flip_indices: Indices where spins should be flipped
            
        Returns:
            Transformed sample array
        """
        samples = samples.copy()
        samples[:, flip_indices] = -samples[:, flip_indices]
        return samples
        
    def _process_embedding(
        self,
        game_state: GameState,
        automorphism: Dict[int, int]
    ) -> Dict[int, List[int]]:
        """Process embedding with given automorphism.
        
        Args:
            game_state: Current game state
            automorphism: Graph automorphism to apply
            
        Returns:
            Processed embedding mapping
        """
        inverted_automorphism = {v: k for k, v in automorphism.items()}
        num_vertices = game_state['num_nodes']
        
        embedding_map = {}
        for embedding_idx, embedding in enumerate(self.embeddings):
            for vertex in range(num_vertices):
                logical_idx = num_vertices * embedding_idx + vertex
                physical_qubit = embedding[inverted_automorphism[vertex]]
                embedding_map[logical_idx] = [physical_qubit]
                
        return embedding_map

    def adjudicate(self, game_state: GameState) -> AdjudicationResult:
        """Adjudicate the game state using quantum annealing.
        
        Args:
            game_state: The current game state
            
        Returns:
            AdjudicationResult containing the adjudication details
            
        Raises:
            ValueError: If the game state is invalid
            RuntimeError: If quantum annealing fails
        """
        if not self._base_sampler:
            raise RuntimeError("Sampler not initialized. Call setup() first.")
            
        self._validate_game_state(game_state)
        
        num_vertices = game_state['num_nodes']
        num_embeddings = len(self.embeddings)
        total_samples = np.zeros((1, num_vertices))  # Initial array for stacking
        
        # Process each chip run
        for _ in range(self.params.num_chip_runs):
            # Select random automorphism
            automorphism = np.random.choice(self.automorphisms)
            embedding_map = self._process_embedding(game_state, automorphism)
            
            # Create sampler with fixed embedding
            sampler = FixedEmbeddingComposite(
                self._base_sampler,
                embedding=embedding_map
            )
            
            # Get Ising model
            ising_model = self._game_state_to_ising(game_state)
            
            # Set up sampling parameters
            sample_kwargs = {
                'num_reads': self.params.num_reads,
                'answer_mode': 'raw'
            }
            
            if not self.params.use_mock:
                sample_kwargs.update({
                    'annealing_time': self.params.anneal_time / 1000,
                    'auto_scale': False
                })
            
            # Perform sampling
            response = sampler.sample_ising(
                ising_model['h'],
                ising_model['j'],
                **sample_kwargs
            )
            
            # Process samples
            samples = np.array(response.record.sample)
            
            # Apply gauge transform if enabled
            if self.params.use_gauge_transform:
                flip_indices = np.random.choice(
                    [0, 1],
                    size=samples.shape[1],
                    p=[0.5, 0.5]
                ).nonzero()[0]
                samples = self._apply_gauge_transform(samples, flip_indices)
            
            # Stack samples for all embeddings
            processed_samples = samples[:, :num_vertices]
            for k in range(1, num_embeddings):
                processed_samples = np.vstack((
                    processed_samples,
                    samples[:, k*num_vertices:(k+1)*num_vertices]
                ))
            
            total_samples = np.vstack((total_samples, processed_samples))
            
        # Remove initial zero row
        total_samples = np.delete(total_samples, 0, axis=0)
        
        # Handle isolated vertices
        isolated_vertices = self._find_isolated_vertices(game_state)
        if isolated_vertices:
            random_samples = np.random.choice(
                [1, -1],
                size=(total_samples.shape[0], len(isolated_vertices))
            )
            for i, vertex in enumerate(isolated_vertices):
                total_samples[:, vertex] = random_samples[:, i]
        
        # Calculate correlation matrix
        sample_count = (self.params.num_reads * num_embeddings *
                       self.params.num_chip_runs)
        correlation_matrix = (
            np.einsum('si,sj->ij', total_samples, total_samples) / sample_count -
            np.eye(num_vertices)
        )
        
        # Compute results
        winner, score, influence_vector = self._compute_winner_score_and_influence(
            game_state, correlation_matrix
        )
        
        return AdjudicationResult(
            game_state=game_state,
            adjudicator='quantum_annealing',
            winner=winner,
            score=score,
            influence_vector=influence_vector,
            correlation_matrix=correlation_matrix,
            parameters=self._parameters
        )