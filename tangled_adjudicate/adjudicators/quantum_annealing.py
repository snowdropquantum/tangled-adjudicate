import os
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
    use_gauge_transform: bool = False
    use_shim: bool = False
    shim_iterations: int = 1
    alpha_phi: float = 0.1
    use_mock: bool = True
    solver_name: str = 'Advantage2_prototype2.6'
    graph_number: Optional[int] = None
    data_dir: Optional[str] = None


class QuantumAnnealingAdjudicator(Adjudicator):
    """Adjudicator implementation using D-Wave quantum annealing."""
    
    def __init__(self) -> None:
        """Initialize the quantum annealing adjudicator."""
        super().__init__()
        self.params = QAParameters()
        self.embeddings: List[List[int]] = []
        self.automorphisms: List[Dict[int, int]] = []
        self.shim_stats: Dict[str] = {}

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

        # load directory for automorphisms & embeddings
        if 'data_dir' in kwargs:
            if not isinstance(kwargs['data_dir'], str):
                raise ValueError("data_dir must be a string")
            if not os.path.isdir(kwargs['data_dir']):
                raise ValueError(f"Directory not found: {kwargs['data_dir']}")
            self.params.data_dir = kwargs['data_dir']

        # we need these so always compute / load in
        self.automorphisms = get_automorphisms(self.params.graph_number, self.params.data_dir)
        self.embeddings = get_embeddings(self.params.graph_number, self.params.solver_name, self.params.data_dir)
            
        # Initialize sampler
        try:
            if self.params.use_mock:
                base_sampler = MockDWaveSampler(topology_type='zephyr', topology_shape=[6, 4])
            else:
                base_sampler = DWaveSampler(solver=self.params.solver_name)
                
            # Store for later use in adjudicate
            self._base_sampler = base_sampler
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize D-Wave sampler: {str(e)}")

        # initialize shim_stats if required
        if self.params.use_shim:
            self.shim_stats = {'qubit_magnetizations': [],
                               'average_absolute_value_of_magnetization': [],
                               'all_flux_bias_offsets': []}

        # Store parameters
        self._parameters = self.params.__dict__

    def _process_embedding(
        self,
        game_state: GameState,
        automorphism: Dict[int, int],
        num_embeddings: int
    ) -> Dict[int, List[int]]:
        """Process embedding with given automorphism.
        
        Args:
            game_state: Current game state
            automorphism: Graph automorphism to apply
            num_embeddings: number of embeddings to use; default is all of them
            
        Returns:
            Processed embedding mapping
        """

        num_vertices = game_state['num_nodes']

        inverted_automorphism_to_use = {v: k for k, v in automorphism.items()}  # swaps key <-> values

        permuted_embedding = []

        for each_embedding in self.embeddings[:num_embeddings]:  # each_embedding is like [1093, 1098, 136]; 343 of these for three-vertex graph
            this_embedding = []
            for each_vertex in range(num_vertices):  # each_vertex ranges from 0 to 2
                this_embedding.append(each_embedding[inverted_automorphism_to_use[each_vertex]])
            permuted_embedding.append(this_embedding)

        # given that permuted_embedding looks like [[1229, 1235, 563], [872, 242, 866], ...]
        # this next part converts into the format {0: [1229], 1: [1235], 2: [563], 3: [872], 4: [242], 5: [866]}

        embedding_map = {}

        for embedding_idx in range(num_embeddings):
            for each_vertex in range(num_vertices):  # up to 0..1037
                embedding_map[num_vertices * embedding_idx + each_vertex] = \
                    [permuted_embedding[embedding_idx][each_vertex]]
                
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

        all_samples = None
        indices_of_flips = None

        # set up sampler kwargs
        if self.params.use_mock and self.params.use_shim:
            print('D-Wave mock sampler is not set up to use the shimming process, turn shim off if using mock!')

        sampler_kwargs = {
                'num_reads': self.params.num_reads,
                'answer_mode': 'raw'
            }

        if not self.params.use_mock:
            sampler_kwargs.update({
                'fast_anneal': True,
                'annealing_time': self.params.anneal_time / 1000,
                'auto_scale': False
            })

        if self.params.use_shim:
            sampler_kwargs.update({'readout_thermalization': 100.,
                                   'auto_scale': False,
                                   'flux_drift_compensation': True,
                                   'flux_biases': [0] * base_sampler.properties['num_qubits']})
            shim_iterations = self.params.shim_iterations
        else:
            shim_iterations = 1    # if we don't shim, just run through shim step only once

        # **********************************************************
        # Step 0: convert game_state to the desired base Ising model
        # **********************************************************

        # for tangled, h_j=0 for all vertices j in the game graph, and J_ij is one of +1, -1, or 0 for all vertex
        # pairs i,j. I named the "base" values (the actual problem defined on the game graph we are asked to solve)
        # base_ising_model.

        base_ising_model = self._game_state_to_ising(game_state)

        # this finds any isolated vertices that may be in the graph -- we will replace the samples returned for these
        # at the end with true 50/50 statistics, so we don't have to worry about them

        isolated_vertices = self._find_isolated_vertices(game_state)

        # We now enter a loop where each pass through the loop programs the chip to specific values of h and J but
        # now for the entire chip. We do this by first selecting one automorphism and embedding it in multiple
        # parallel ways across the entire chip, and then optionally applying a gauge transform across all the qubits
        # used. This latter process chooses different random gauges for each of the embedded instances.

        for _ in range(self.params.num_chip_runs):

            # *******************************************************************
            # Step 1: Randomly select an automorphism and embed it multiple times
            # *******************************************************************

            automorphism = np.random.choice(self.automorphisms)
            embedding_map = self._process_embedding(game_state, automorphism, num_embeddings)

            # *****************************************************************************************************
            # Step 2: Set h, J parameters for full chip using parallel embeddings of a randomly chosen automorphism
            # *****************************************************************************************************

            # compute full_h and full_j which are h, jay values for the entire chip assuming the above automorphism
            # I am calling the problem definition and variable ordering before the automorphism the BLACK or BASE
            # situation. After the automorphism the problem definition and variable labels change -- I'm calling the
            # situation after the automorphism has been applied the BLUE situation.

            full_h = {}
            full_j = {}

            for embedding_idx in range(num_embeddings):
                for each_vertex in range(num_vertices):
                    full_h[num_vertices * embedding_idx + each_vertex] = 0

            for k, v in base_ising_model['j'].items():
                edge_under_automorph = (min(automorphism[k[0]], automorphism[k[1]]),
                                        max(automorphism[k[0]], automorphism[k[1]]))
                full_j[edge_under_automorph] = v
                for j in range(1, num_embeddings):
                    full_j[(edge_under_automorph[0] + num_vertices * j,
                            edge_under_automorph[1] + num_vertices * j)] = v

            # **************************************************************************
            # Step 3: Choose random gauge, modify h, J parameters for full chip using it
            # **************************************************************************

            # next we optionally apply a random gauge transformation. I call the situation after the gauge
            # transformation has been applied the BLUE with RED STAR situation.

            if self.params.use_gauge_transform:
                flip_map = [np.random.choice([-1, 1]) for _ in full_h]   # random list of +1, -1 values of len # qubits
                indices_of_flips = [i for i, x in enumerate(flip_map) if x == -1]       # the indices of the -1 values

                for edge_key, j_val in full_j.items():              # for each edge and associated J value
                    full_j[edge_key] = j_val * flip_map[edge_key[0]] * flip_map[edge_key[1]]   # Jij -> J_ij g_i g_j

            # *****************************************
            # Step 4: Choose sampler and its parameters
            # *****************************************

            sampler_kwargs.update({'h': full_h,
                                   'J': full_j})

            sampler = FixedEmbeddingComposite(self._base_sampler, embedding=embedding_map)   # applies the embedding

            # *************************************************************************
            # Step 5: Optionally start shimming process in the BLUE with RED STAR basis
            # *************************************************************************

            # all of this in the BLUE with RED STAR basis, ie post automorph, post gauge transform
            for shim_iteration_idx in range(shim_iterations):

                # **************************************
                # Step 6: Generate samples from hardware
                # **************************************

                ss = sampler.sample_ising(**sampler_kwargs)
                all_samples = ss.record.sample

                if self.params.use_shim:

                    # *************************************************************
                    # Step 6a: Compute average values of each qubit == magnetization
                    # *************************************************************

                    magnetization = np.sum(all_samples, axis=0)/self.params.num_reads   # BLUE with RED STAR label ordering
                    self.shim_stats['average_absolute_value_of_magnetization'].append(np.sum([abs(k) for k in magnetization])/len(magnetization))

                    qubit_magnetization = [0] * self._base_sampler.properties['num_qubits']
                    for k, v in embedding_map.items():
                        qubit_magnetization[v[0]] = magnetization[k]        # check

                    self.shim_stats['qubit_magnetizations'].append(qubit_magnetization)

                    # **************************************
                    # Step 6b: Adjust flux bias offset terms
                    # **************************************

                    for k in range(self._base_sampler.properties['num_qubits']):
                        sampler_kwargs['flux_biases'][k] -= self.params.alpha_phi * qubit_magnetization[k]

                    self.shim_stats['all_flux_bias_offsets'].append(sampler_kwargs['flux_biases'])

            # *****************************************************************************************************
            # Step 7: Reverse gauge transform, from BLUE with RED STAR to just BLUE, after shimming process is done
            # *****************************************************************************************************

            if self.params.use_gauge_transform:
                all_samples[:, indices_of_flips] = -all_samples[:, indices_of_flips]

            # ***********************************
            # Step 8: Stack samples in BLUE order
            # ***********************************

            # this should make a big fat stack of the results in BLUE variable ordering
            all_samples_processed_blue = all_samples[:, range(num_vertices)]
            for k in range(1, num_embeddings):
                all_samples_processed_blue = np.vstack((all_samples_processed_blue,
                                                        all_samples[:, range(num_vertices * k,
                                                                             num_vertices * (k + 1))]))

            # **********************************************************************
            # Step 9: Reorder columns to make them BLACK order instead of BLUE order
            # **********************************************************************

            all_samples_processed_black = all_samples_processed_blue[:, [automorphism[i] for i in range(all_samples_processed_blue.shape[1])]]

            # *********************************************************
            # Step 10: Add new samples to the stack, all in BLACK order
            # *********************************************************

            total_samples = np.vstack((total_samples, all_samples_processed_black))

        # ***************************************************************
        # Step 11: Post process samples stack to extract return variables
        # ***************************************************************

        total_samples = np.delete(total_samples, (0), axis=0)  # delete first row of zeros

        # replace columns where there are disconnected variables with truly random samples
        for idx in isolated_vertices:
            total_samples[:, idx] = np.random.choice([1, -1], size=total_samples.shape[0])

        sample_count = self.params.num_reads * num_embeddings * self.params.num_chip_runs

        # this is a full matrix with zeros on the diagonal that uses all the samples
        correlation_matrix = \
            (np.einsum('si,sj->ij', total_samples, total_samples) / sample_count -
             np.eye(num_vertices))

        # Compute results
        winner, score, influence_vector = self._compute_winner_score_and_influence(game_state, correlation_matrix)

        return AdjudicationResult(
            game_state=game_state,
            adjudicator='quantum_annealing',
            winner=winner,
            score=score,
            influence_vector=influence_vector,
            correlation_matrix=correlation_matrix,
            parameters=self._parameters
        )
