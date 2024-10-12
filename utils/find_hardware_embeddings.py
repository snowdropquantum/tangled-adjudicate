import time
import os
import pickle
import dimod
import dwave_networkx as dnx
import networkx as nx
import numpy as np

from dwave import embedding
from dwave.system.samplers import DWaveSampler
from minorminer import subgraph as glasgow

from utils.game_graph_properties import GraphProperties


# todo go through this and see if I can trim it down, document it, and make it work
# input should be game graph, output should be dictionary of embeddings


def get_zephyr_subgrid(A, rows, cols, gridsize=4):
    """Make a subgraph of a Zephyr (Advantage2) graph on a set of rows and columns of unit cells.

    Args:
        A (nx.Graph): Qubit connectivity graph
        rows (Iterable): Iterable of rows of unit cells to include
        cols (Iterable): Iterable of columns of unit cells to include

    Returns:
        nx.Graph: The subgraph of A induced on the nodes in "rows" and "cols"
    """

    coords = [dnx.zephyr_coordinates(gridsize).linear_to_zephyr(v) for v in A.nodes]
    c = np.asarray(coords)
    used_coords = [c for c in coords if
                   (c[0] == 0 and c[4] in cols and c[1] >= 2*min(rows) and c[1] <= 2*max(rows)+2) or
                   (c[0] == 1 and c[4] in rows and c[1] >= 2*min(cols) and c[1] <= 2*max(cols)+2)]
    # (u, w, k, z) -> (u, w, k / 2, k % 2, z)

    subgraph = A.subgraph([dnx.zephyr_coordinates(gridsize).zephyr_to_linear(c)
                          for c in used_coords]).copy()

    return subgraph


def get_independent_embeddings(embs):
    """Finds a list of non-overlapping embeddings in `embs`.

    Args:
        embs (list[dict]): a list of embeddings (dict)

    Returns:
        List[dict]: a list of embeddings (dict)
    """
    start = time.process_time()

    Gemb = nx.Graph()
    Gemb.add_nodes_from(range(len(embs)))
    for i, emb1 in enumerate(embs):
        V1 = set(emb1.values())
        for j in range(i + 1, len(embs)):
            emb2 = embs[j]
            V2 = set(emb2.values())
            if not V1.isdisjoint(V2):
                Gemb.add_edge(i, j)
    print(f'Built graph.  Took {time.process_time()-start} seconds')
    start = time.process_time()

    Sbest = None
    max_size = 0
    for i in range(100000):
        if len(Gemb) > 0:
            S = nx.maximal_independent_set(Gemb)
        else:
            return []
        if len(S) > max_size:
            Sbest = S
            max_size = len(S)

    print(f'Built 100,000 greedy MIS.  Took {time.process_time()-start} seconds')
    print(f'Found {len(Sbest)} disjoint embeddings.')
    return [embs[x] for x in Sbest]


def search_for_subgraphs_in_subgrid(B, subgraph, timeout=20, max_number_of_embeddings=np.inf, verbose=True):
    """Find a list of subgraph (embeddings) in a subgrid.

    Args:
        B (nx.Graph): a subgrid
        subgraph (nx.Graph): subgraphs in B to search for
        timeout (int, optional): time limit for search. Defaults to 20.
        max_number_of_embeddings (int, optional): maximum number of embeddings to look for. Defaults to np.inf.
        verbose (bool, optional): Flag for verbosity. Defaults to True.

    Returns:
        List[dict]: a list of embeddings
    """
    embs = []
    while True and len(embs) < max_number_of_embeddings:
        temp = glasgow.find_subgraph(subgraph, B, timeout=timeout, triggered_restarts=True)
        if len(temp) == 0:
            break
        else:
            B.remove_nodes_from(temp.values())
            embs.append(temp)
            if verbose:
                print(f'{len(B)} vertices remain...')

    if verbose:
        print(f'Found {len(embs)} embeddings.')
    return embs


def raster_embedding_search(hardware_graph, subgraph, raster_breadth=2, delete_used=True,
                            verbose=True, gridsize=6, verify_embeddings=False, **kwargs):
    """Returns a matrix (n, L) of subgraph embeddings to hardware_graph.

    Args:
        hardware_graph (nx.Graph): target graph to embed to
        subgraph (nx.Graph): A smaller graph to embed into hardware_graph
        raster_breadth (int, optional): Breadth parameter of raster search. Defaults to 5.
        delete_used (bool, optional): Flag whether nodes in hardware_graph can appear in multiple embeddings.
                                      If set to true, nodes cannot be used in multiple embeddings. Defaults to True.
        verbose (bool, optional): Whether to print progress. Defaults to True.
        gridsize (int, optional): Size of grid. Defaults to 16.
        verify_embeddings (bool, optional): Flag whether embeddings should be verified. Defaults to False.

    Raises:
        Exception: Exception raised when embeddings are invalid and when `verify_embeddings` is True.

    Returns:
        numpy.ndarray: a matrix of embeddings
    """

    A = hardware_graph.copy()

    embs = []
    for row_offset in range(gridsize - raster_breadth + 1):

        for col_offset in range(gridsize - raster_breadth + 1):
            B = get_zephyr_subgrid(A, range(row_offset, row_offset + raster_breadth),
                                   range(col_offset, col_offset + raster_breadth), gridsize)

            if verbose:
                print(f'row,col=({row_offset},{col_offset}) starting with {len(B)} vertices')

            sub_embs = search_for_subgraphs_in_subgrid(B, subgraph, verbose=verbose, **kwargs)
            if delete_used:
                for sub_emb in sub_embs:
                    A.remove_nodes_from(sub_emb.values())

            if verify_embeddings:
                for emb in sub_embs:
                    X = list(embedding.diagnose_embedding({p: [emb[p]] for p in emb}, subgraph, _A))
                    if len(X):
                        print(X[0])
                        raise Exception

            embs += sub_embs

    # Get independent set of embeddings
    independent_embs = get_independent_embeddings(embs)

    embmat = np.asarray([[ie[v] for ie in independent_embs] for v in subgraph.nodes]).T

    return embmat


def main():

    # first, choose a hardware graph, or more generally the target graph for the embedding
    target_graph = DWaveSampler(solver='Advantage2_prototype2.4').to_networkx_graph()

    # next, define source graph
    graph = GraphProperties(graph_number=3)
    source_graph = nx.Graph()
    source_graph.add_nodes_from([k for k in range(graph.vertex_count)])
    source_graph.add_edges_from(graph.edge_list)

    # state = [0, 0, 1, 2, 3, 2, 1, 3, 2, 1]
    # edge_state_map = {1: 0, 2: -1, 3: 1}    # maps edge state to J value 1 => J = 0; 2 => J = -1 FM; 3 => J = +1 AFM
    #
    # edge_vals = state[graph.vertex_count:]
    #
    # j_vals = [edge_state_map[k] for k in edge_vals]

    # bqm = dimod.BinaryQuadraticModel(vartype='SPIN')

    # cnt = 0
    # for each in graph.edge_list:
    #     bqm.add_quadratic(each[0], each[1], 1)
    #     cnt += 1
    #
    # source_graph = dimod.to_networkx_graph(bqm)

    embmat = raster_embedding_search(target_graph, source_graph)

    with open(os.path.join(os.getcwd(), '..', 'results', 'embedding_matrix_graph_3.txt'), "wb") as fp:
        pickle.dump(embmat, fp)

    print('')

    # embmat is an ndarray of size (51, 64) where 51 is the number of embeddings found and 64 is the number of vertices
    # in the original graph


if __name__ == "__main__":
    main()
