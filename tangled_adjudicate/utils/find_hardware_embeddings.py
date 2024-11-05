import time
import os
import pickle
import dwave_networkx as dnx
import networkx as nx
import numpy as np

from dwave import embedding
from dwave.system.samplers import DWaveSampler
from minorminer import subgraph as glasgow

from utils.game_graph_properties import GraphProperties


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
    # c = np.asarray(coords)
    used_coords = [c for c in coords if
                   (c[0] == 0 and c[4] in cols and 2 * min(rows) <= c[1] <= 2 * max(rows) + 2) or
                   (c[0] == 1 and c[4] in rows and 2 * min(cols) <= c[1] <= 2 * max(cols) + 2)]
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

    g_emb = nx.Graph()
    g_emb.add_nodes_from(range(len(embs)))
    for i, emb1 in enumerate(embs):
        V1 = set(emb1.values())
        for j in range(i + 1, len(embs)):
            emb2 = embs[j]
            V2 = set(emb2.values())
            if not V1.isdisjoint(V2):
                g_emb.add_edge(i, j)

    start = time.process_time()

    s_best = None
    max_size = 0
    for i in range(100000):
        if len(g_emb) > 0:
            S = nx.maximal_independent_set(g_emb)
        else:
            return []
        if len(S) > max_size:
            s_best = S
            max_size = len(S)

    print(f'Built 100,000 greedy MIS.  Took {time.process_time()-start} seconds')
    print(f'Found {len(s_best)} disjoint embeddings.')
    return [embs[x] for x in s_best]


def search_for_subgraphs_in_subgrid(B, subgraph, timeout=20, max_number_of_embeddings=np.inf, verbose=False):
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
                            verbose=False, gridsize=6, verify_embeddings=False, **kwargs):
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


def load_embeddings_dictionary(file_name):

    with open(os.path.join(os.getcwd(), '..', 'data', file_name), "rb") as fp:
        dictionary_of_embeddings = pickle.load(fp)

    return dictionary_of_embeddings


def main():
    # generates embeddings into hardware for graphs 1 through 10; default settings take about 10 minutes total to run
    #
    # the dictionary_of_embeddings object looks like
    # dictionary_of_embeddings = {1: [[1094, 1100], [609, 1019] , ... ], 2: [[1093, 1098, 136], [558, 725, 731], ... ]}
    # number found for raster_breadth = 2 and grid_size = 6 : 578, 346, 219, 167, 86, 48, 58, 32, 16, 8

    raster_breadth = 2   # these parameters seem to work well to get a lot of embeddings
    grid_size = 6

    generate_embeddings_file = False  # if True generates the embeddings in /data directory, else attempts to load it
    embeddings_file_name = ('embeddings_dictionary_graphs_1_through_10_raster_breadth_'
                            + str(raster_breadth) + '_gridsize_' + str(grid_size) + '.pkl')   # name of data file

    # checks to see if /data exists; if it doesn't it creates it; if it does, it writes the file to disk
    data_dir = os.path.join(os.getcwd(), '..', 'data')

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    initial_start = time.time()

    if generate_embeddings_file:
        dictionary_of_embeddings = {}

        # choose a hardware graph, or more generally the target graph for the embedding
        target_graph = DWaveSampler(solver='Advantage2_prototype2.4').to_networkx_graph()

        for graph_number in range(1, 11):

            # define source graph
            graph = GraphProperties(graph_number=graph_number)
            source_graph = nx.Graph()
            source_graph.add_nodes_from([k for k in range(graph.vertex_count)])
            source_graph.add_edges_from(graph.edge_list)

            print('********************')
            print('Evaluating graph #', graph_number, ', raster_breadth', raster_breadth, 'and grid_size', grid_size)
            print('********************')

            start = time.time()

            embmat = raster_embedding_search(target_graph, source_graph,
                                             raster_breadth=raster_breadth,
                                             gridsize=grid_size)

            dictionary_of_embeddings[graph_number] = [k.tolist() for k in embmat]
            print('graph_number', graph_number, 'took', time.time() - start, 'seconds.')

        with open(os.path.join(data_dir, embeddings_file_name), "wb") as fp:
            pickle.dump(dictionary_of_embeddings, fp)

    else:

        dictionary_of_embeddings = load_embeddings_dictionary(file_name=embeddings_file_name)
        print('this is the loaded dictionary:')
        print(dictionary_of_embeddings)

    print('total time was', time.time() - initial_start, 'seconds.')


if __name__ == "__main__":
    main()
