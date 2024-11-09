""" given a D-Wave processor (target graph) and a source graph, find as many embeddings as possible of the source
graph into the target hardware graph """
import time
import os
import pickle
import dwave_networkx as dnx
import networkx as nx
import numpy as np

from dwave.system.samplers import DWaveSampler
from minorminer import subgraph as glasgow

from tangled_adjudicate.utils.game_graph_properties import GraphProperties


def get_zephyr_subgrid(qubit_connectivity_graph, rows, cols, gridsize=4):
    """Make a subgraph of a Zephyr (Advantage2) graph on a set of rows and columns of unit cells.

    Args:
        qubit_connectivity_graph (nx.Graph): Qubit connectivity graph
        rows (Iterable): Iterable of rows of unit cells to include
        cols (Iterable): Iterable of columns of unit cells to include
        gridsize: Int

    Returns:
        nx.Graph: The subgraph of qubit_connectivity_graph induced on the nodes in "rows" and "cols"

    """

    coords = [dnx.zephyr_coordinates(gridsize).linear_to_zephyr(v) for v in qubit_connectivity_graph.nodes]

    used_coords = [c for c in coords if
                   (c[0] == 0 and c[4] in cols and 2 * min(rows) <= c[1] <= 2 * max(rows) + 2) or
                   (c[0] == 1 and c[4] in rows and 2 * min(cols) <= c[1] <= 2 * max(cols) + 2)]

    subgraph = qubit_connectivity_graph.subgraph([dnx.zephyr_coordinates(gridsize).zephyr_to_linear(c)
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
        v1 = set(emb1.values())
        for j in range(i + 1, len(embs)):
            emb2 = embs[j]
            v2 = set(emb2.values())
            if not v1.isdisjoint(v2):
                g_emb.add_edge(i, j)

    start = time.process_time()

    s_best = None
    max_size = 0
    for i in range(100000):
        if len(g_emb) > 0:
            s = nx.maximal_independent_set(g_emb)
        else:
            return []
        if len(s) > max_size:
            s_best = s
            max_size = len(s)

    print(f'Built 100,000 greedy MIS.  Took {time.process_time()-start} seconds')
    print(f'Found {len(s_best)} disjoint embeddings.')
    return [embs[x] for x in s_best]


def search_for_subgraphs_in_subgrid(subgrid_graph, subgraph, timeout=20, max_embeddings=np.inf, verbose=False):
    """Find a list of subgraph (embeddings) in a subgrid.

    Args:
        subgrid_graph (nx.Graph): a subgrid
        subgraph (nx.Graph): subgraphs in subgrid_graph to search for
        timeout (int, optional): time limit for search. Defaults to 20.
        max_embeddings (int, optional): maximum number of embeddings to look for. Defaults to np.inf.
        verbose (bool, optional): Flag for verbosity. Defaults to True.

    Returns:
        List[dict]: a list of embeddings
    """
    embs = []
    while True and len(embs) < max_embeddings:
        temp = glasgow.find_subgraph(subgraph, subgrid_graph, timeout=timeout, triggered_restarts=True)
        if len(temp) == 0:
            break
        else:
            subgrid_graph.remove_nodes_from(temp.values())
            embs.append(temp)
            if verbose:
                print(f'{len(subgrid_graph)} vertices remain...')

    if verbose:
        print(f'Found {len(embs)} embeddings.')
    return embs


def raster_embedding_search(hardware_graph, subgraph, raster_breadth=2, delete_used=True,
                            verbose=False, gridsize=6, **kwargs):
    """Returns a matrix (n, L) of subgraph embeddings to hardware_graph.

    Args:
        hardware_graph (nx.Graph): target graph to embed to
        subgraph (nx.Graph): A smaller graph to embed into hardware_graph
        raster_breadth (int, optional): Breadth parameter of raster search. Defaults to 5.
        delete_used (bool, optional): Flag whether nodes in hardware_graph can appear in multiple embeddings.
                                      If set to true, nodes cannot be used in multiple embeddings. Defaults to True.
        verbose (bool, optional): Whether to print progress. Defaults to True.
        gridsize (int, optional): Size of grid. Defaults to 16.

    Returns:
        numpy.ndarray: a matrix of embeddings
    """

    hardware_graph_copy = hardware_graph.copy()

    embs = []
    for row_offset in range(gridsize - raster_breadth + 1):

        for col_offset in range(gridsize - raster_breadth + 1):
            zephyr_subgrid = get_zephyr_subgrid(hardware_graph_copy, range(row_offset, row_offset + raster_breadth),
                                                range(col_offset, col_offset + raster_breadth), gridsize)

            if verbose:
                print(f'row, col=({row_offset}, {col_offset}) starting with {len(zephyr_subgrid)} vertices')

            sub_embs = search_for_subgraphs_in_subgrid(zephyr_subgrid, subgraph, verbose=verbose, **kwargs)
            if delete_used:
                for sub_emb in sub_embs:
                    hardware_graph_copy.remove_nodes_from(sub_emb.values())

            embs += sub_embs

    # Get independent set of embeddings
    independent_embs = get_independent_embeddings(embs)

    embmat = np.asarray([[ie[v] for ie in independent_embs] for v in subgraph.nodes]).T

    return embmat


def get_embeddings(source_graph_number, qc_solver_to_use):
    # generates multiple parallel embeddings into hardware for your graph
    # the smaller the graph, the longer this takes -- e.g. source_graph_number == 1 takes about 4 minutes
    #
    # the list_of_embeddings object looks like
    # list_of_embeddings = [[1093, 1098, 136], [558, 725, 731], ... ]
    # number found for 'Advantage2_prototype2.5', raster_breadth == 2 and grid_size == 6
    # 577, 343, 215, 169, 87, 48, 58, 33, 15, 9

    # these parameters seem to work well to get a lot of embeddings, you can try to change them if you want
    raster_breadth = 2
    grid_size = 6

    file_name = ('embeddings_graph_number_' + str(source_graph_number) + '_raster_breadth_' + str(raster_breadth) +
                 '_gridsize_' + str(grid_size) + '_qc_' + qc_solver_to_use + '.pkl')

    data_dir = os.path.join(os.getcwd(), '..', 'data')      # checks to see if /data exists; if not, creates it

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    file_path = os.path.join(data_dir, file_name)

    # checks to see if the file is there already; if it is, load it; if not, create it
    if os.path.isfile(file_path):
        with open(file_path, "rb") as fp:
            list_of_embeddings = pickle.load(fp)
    else:
        print('********************')
        print('embeddings file not found, creating it, this will only happen once for this choice of parameters ...')
        print('finding embeddings for source graph #', source_graph_number, 'into target graph', qc_solver_to_use)
        print('using raster_breadth', raster_breadth, 'and grid_size', grid_size)
        print('********************')

        start = time.time()

        # target graph for the embedding
        target_graph = DWaveSampler(solver=qc_solver_to_use).to_networkx_graph()

        # source graph
        graph = GraphProperties(graph_number=source_graph_number)
        source_graph = nx.Graph()
        source_graph.add_nodes_from([k for k in range(graph.vertex_count)])
        source_graph.add_edges_from(graph.edge_list)

        # find embeddings
        embmat = raster_embedding_search(target_graph, source_graph,
                                         raster_breadth=raster_breadth,
                                         gridsize=grid_size)

        list_of_embeddings = [k.tolist() for k in embmat]
        print('for graph_number', source_graph_number, 'computing embeddings took', time.time() - start, 'seconds.')

        with open(os.path.join(data_dir, file_name), "wb") as fp:
            pickle.dump(list_of_embeddings, fp)

    return list_of_embeddings
