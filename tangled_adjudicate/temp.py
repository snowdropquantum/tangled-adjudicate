import ast
import sys
import os
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx

from tangled_adjudicate.utils.game_graph_properties import GraphProperties
from tangled_adjudicate.utils.find_graph_automorphisms import get_automorphisms
from tensorflow.python.keras.legacy_tf_layers.core import dropout


class K4(object):
    def __init__(self, x_offset, y_offset, d, vertices, edges, draw_box=False, box_idx=None):
        # vertices is a list of 4 numbers like [0,0,1,2]
        # edges is a list of six numbers in dict order like [1,2,2,2,3,3]
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.d = d
        self.draw_box = draw_box
        self.G = nx.complete_graph(4)
        self.node_colors_map = {0: "black", 1: "red", 2: "blue"}
        self.edge_colors_map = {1: "black", 2: "green", 3: "purple"}

        self.box_colors_map = {
            0: "blue",
            1: "red",
            2: "green",
            3: "purple",
            4: "orange",
            5: "brown",
            6: "pink",
            7: "gray",
            8: "cyan",
            9: "olive",
            10: "lime",
            11: "gold"
        }

        self.vertex_colors = [self.node_colors_map[vertices[v]] for v in self.G.nodes()]
        self.edge_colors = [self.edge_colors_map[edges[edge_idx]] for edge_idx in range(len(self.G.edges()))]

        self.node_labels = {i: str(i) for i in range(4)}

        # Define positions for a square layout
        self.pos = {
            0: (x_offset, y_offset),  # Bottom-left
            1: (x_offset, y_offset+d),  # Top-left
            2: (x_offset+d, y_offset+d),  # Top-right
            3: (x_offset+d, y_offset),  # Bottom-right
        }

        if draw_box:
            self.square = patches.Rectangle(
                (self.x_offset-0.02, self.y_offset-0.02),  # Bottom-left corner (x, y)
                self.d+0.04,  # Width
                self.d+0.04,  # Height
                facecolor=self.box_colors_map[box_idx],  # Fill color
                edgecolor="black",  # Border color
                alpha=0.2  # Transparency (0 is fully transparent, 1 is opaque)
            )

    def draw(self):
        nx.draw_networkx_nodes(self.G, self.pos, node_color=self.vertex_colors, node_size=120)
        nx.draw_networkx_edges(self.G, self.pos, edge_color=self.edge_colors, width=2)
        # nx.draw_networkx_labels(self.G, self.pos, labels=self.node_labels, font_size=6, font_weight="bold", font_color="white")

def main():

    graph_number = 3

    graph = GraphProperties(graph_number)

    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
    data_dir = os.path.join(script_dir, 'data')
    list_of_automorphisms = get_automorphisms(graph_number=graph_number, data_dir=data_dir)

    possible_vertex_states = []
    for positions in itertools.permutations(range(graph.vertex_count), 2):
        lst = [0] * graph.vertex_count
        lst[positions[0]] = 1
        lst[positions[1]] = 2
        possible_vertex_states.append(lst)

    possible_vertex_states.sort()

    elements = [1, 2, 3]
    possible_edge_states = list(itertools.product(elements, repeat=graph.edge_count))

    # all_states is a list of lists enumerating ALL of the game states
    all_states = [j + list(k) for j in possible_vertex_states for k in possible_edge_states]

    # this next part creates a dictionary where the keys are each of the elements of all_states and the values are
    # lists of all the states connected to the key by an automorphism. Note that different automorphisms can lead
    # to the same state, so at some point the list is converted to a set and then back to a list

    all_states_with_symmetries = {}
    all_states_no_symmetries = {}

    # iterate over all enumerated states
    for state in all_states:

        # # create a key for state
        # all_states_with_symmetries[str(state)] = []

        # create a list for all the symmetric states
        list_of_states_connected_by_symmetry = []

        # get indices of the red and blue vertices
        only_vertices = state[:graph.vertex_count]
        red_vertex_index = only_vertices.index(1)
        blue_vertex_index = only_vertices.index(2)

        # iterate over all automorphisms
        for automorph in list_of_automorphisms:

            # initialize the state we want to compute (transforming state under automorph)
            state_transformed_under_automorph = [0] * graph.vertex_count

            # write transformed vertices into the transformed state -- this finishes the vertex part
            state_transformed_under_automorph[automorph[red_vertex_index]] = 1
            state_transformed_under_automorph[automorph[blue_vertex_index]] = 2

            # now we want to transform the edges under the automorphism
            for edge_idx in range(graph.edge_count):
                first_vertex = automorph[graph.edge_list[edge_idx][0]]
                second_vertex = automorph[graph.edge_list[edge_idx][1]]
                if first_vertex < second_vertex:
                    transformed_edge = (first_vertex, second_vertex)
                else:
                    transformed_edge = (second_vertex, first_vertex)

                transformed_edge_idx = graph.edge_list.index(transformed_edge)

                state_transformed_under_automorph.append(state[graph.vertex_count + transformed_edge_idx])

            list_of_states_connected_by_symmetry.append(str(state_transformed_under_automorph))
        # remove duplicates
        all_states_with_symmetries[str(state)] = list(dict.fromkeys(list_of_states_connected_by_symmetry))
        all_states_no_symmetries[str(state)] = list_of_states_connected_by_symmetry

    uniques = []
    duplicates = []

    for k, v in all_states_with_symmetries.items():
        if k not in duplicates:
            uniques.append(k)
        for j in range(1, len(v)):
            duplicates.append(v[j])

    graph_list = []

    states_to_look_at = ['[0, 0, 1, 2, 1, 1, 1, 1, 1, 1]']

    for outer_key, outer_value in all_states_with_symmetries.items():
        if outer_key in states_to_look_at:
            cnt = 0
            print('with duplicates removed there are:', len(outer_value))
            for each in outer_value:
                state = ast.literal_eval(each)
                if cnt < 12:
                    graph_list.append(K4(x_offset=0.05+0.08*cnt, y_offset=0.90-0.48, d=0.04, vertices=state[:graph.vertex_count], edges=state[graph.vertex_count:]))
                else:
                    graph_list.append(
                        K4(x_offset=0.05 + 0.08 * (cnt-12), y_offset=0.90-0.56, d=0.04, vertices=state[:graph.vertex_count],
                           edges=state[graph.vertex_count:]))
                cnt += 1

    box_idx = {}

    for outer_key, outer_value in all_states_no_symmetries.items():
        if outer_key in states_to_look_at:
            cnt = 0
            box_cnt = 0
            print('without duplicates removed there are:', len(outer_value))
            for each in outer_value:
                box_idx[each] = None
            for each in outer_value:
                state = ast.literal_eval(each)
                draw_box = False
                if all_states_no_symmetries[outer_key].count(each) > 1:
                    draw_box = True
                    if box_idx[each] is None:
                        box_idx[each] = box_cnt
                        box_cnt += 1
                if cnt == 0:
                    graph_list.append(K4(x_offset=0.05+0.08*cnt, y_offset=0.90-0.08, d=0.04,
                                         vertices=state[:graph.vertex_count], edges=state[graph.vertex_count:],
                                         draw_box=False, box_idx=box_idx[each]))
                if cnt < 12:
                    graph_list.append(K4(x_offset=0.05+0.08*cnt, y_offset=0.90-0.24, d=0.04,
                                         vertices=state[:graph.vertex_count], edges=state[graph.vertex_count:],
                                         draw_box=draw_box, box_idx=box_idx[each]))
                else:
                    graph_list.append(
                        K4(x_offset=0.05 + 0.08 * (cnt-12), y_offset=0.90-0.32, d=0.04, vertices=state[:graph.vertex_count],
                           edges=state[graph.vertex_count:], draw_box=draw_box, box_idx=box_idx[each]))
                cnt += 1

    # edges_to_use = [1, 2, 2, 1, 3, 3]
    #
    # # graph_list.append(
    # #     K4(x_offset=0.05 + 0, y_offset=0.90 - 0.5, d=0.1, vertices=[0, 0, 0, 0], edges=edges_to_use))
    #
    # graph_list.append(
    #     K4(x_offset=0.05 + 0, y_offset=0.90 - 0.5, d=0.1, vertices=[1, 2, 0, 0], edges=edges_to_use))
    # graph_list.append(
    #     K4(x_offset=0.05 + 0.15, y_offset=0.90 - 0.5, d=0.1, vertices=[1, 0, 2, 0], edges=edges_to_use))
    # graph_list.append(
    #     K4(x_offset=0.05 + 0.3, y_offset=0.90 - 0.5, d=0.1, vertices=[1, 0, 0, 2], edges=edges_to_use))
    # graph_list.append(
    #     K4(x_offset=0.05 + 0.45, y_offset=0.90 - 0.5, d=0.1, vertices=[0, 1, 2, 0], edges=edges_to_use))
    # graph_list.append(
    #     K4(x_offset=0.05 + .6, y_offset=0.90 - 0.5, d=0.1, vertices=[0, 1, 0, 2], edges=edges_to_use))
    # graph_list.append(
    #     K4(x_offset=0.05 + .75, y_offset=0.90 - 0.5, d=0.1, vertices=[0, 0, 1, 2], edges=edges_to_use))
    #
    # graph_list.append(
    #     K4(x_offset=0.05 + 0, y_offset=0.90 - 0.65, d=0.1, vertices=[2, 1, 0, 0], edges=edges_to_use))
    # graph_list.append(
    #     K4(x_offset=0.05 + 0.15, y_offset=0.90 - 0.65, d=0.1, vertices=[2, 0, 1, 0], edges=edges_to_use))
    # graph_list.append(
    #     K4(x_offset=0.05 + 0.3, y_offset=0.90 - 0.65, d=0.1, vertices=[2, 0, 0, 1], edges=edges_to_use))
    # graph_list.append(
    #     K4(x_offset=0.05 + 0.45, y_offset=0.90 - 0.65, d=0.1, vertices=[0, 2, 1, 0], edges=edges_to_use))
    # graph_list.append(
    #     K4(x_offset=0.05 + 0.6, y_offset=0.90 - 0.65, d=0.1, vertices=[0, 2, 0, 1], edges=edges_to_use))
    # graph_list.append(
    #     K4(x_offset=0.05 + 0.75, y_offset=0.90 - 0.65, d=0.1, vertices=[0, 0, 2, 1], edges=edges_to_use))

    # Draw the graph
    fig, ax = plt.subplots(figsize=(8, 8))

    for each in graph_list:
        each.draw()
        if each.draw_box:
            ax.add_patch(each.square)

    # Set x-axis and y-axis limits
    plt.xlim(0, 1)  # Set x-axis range
    plt.ylim(0, 1)  # Set y-axis range

    # Show the plot
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    sys.exit(main())
