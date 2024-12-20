""" stores properties of game graphs """
import sys

# A Tangled game graph is specified by a graph number, which label specific graphs included here. In this module there
# are 11 included graphs numbered 1 through 11. Each graph requires specification of vertex count (how many vertices
# the graph has) and an explicit edge list, which are included for these 11 graphs. If you'd like to add a new graph,
# it's simple -- just add it to the GraphProperties class.


class GraphProperties(object):
    def __init__(self, graph_number):
        # graph_number is an integer, currently in the range 1 to 10, that labels which graph we are using.
        # to add a new graph, simply define a new graph_number (say 11) and provide its vertex_count and edge_list
        # following the pattern here.

        if graph_number == 1:
            # K_2, complete graph on 2 vertices, 1 edge
            self.vertex_count = 2
            self.edge_list = [(0, 1)]

        elif graph_number == 2:
            # K_3, complete graph on 3 vertices, 3 edges
            self.vertex_count = 3
            self.edge_list = [(0, 1), (0, 2), (1, 2)]

        elif graph_number == 3:
            # K_4, complete graph on 4 vertices, 6 edges
            self.vertex_count = 4
            self.edge_list = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

        elif graph_number == 4:
            # hexagon, 6 vertices, 6 edges
            self.vertex_count = 6
            self.edge_list = [(0, 1), (0, 5), (1, 2), (2, 3), (3, 4), (4, 5)]

        elif graph_number == 5:
            # petersen graph, smallest snark, 10 vertices, 15 edges
            self.vertex_count = 10
            self.edge_list = [(0, 2), (0, 3), (0, 6),
                              (1, 3), (1, 4), (1, 7),
                              (2, 4), (2, 8),
                              (3, 9),
                              (4, 5),
                              (5, 6), (5, 9),
                              (6, 7),
                              (7, 8),
                              (8, 9)]

        elif graph_number == 6:
            # 4D tesseract graph, 16 vertices, 32 edges
            self.vertex_count = 16

            self.edge_list = [(0, 3), (0, 5), (0, 9), (0, 15),
                              (1, 4), (1, 6), (1, 8), (1, 10),
                              (2, 5), (2, 7), (2, 9), (2, 11),
                              (3, 6), (3, 10), (3, 12),
                              (4, 7), (4, 11), (4, 13),
                              (5, 12), (5, 14),
                              (6, 13), (6, 15),
                              (7, 8), (7, 14),
                              (8, 9), (8, 15),
                              (9, 10),
                              (10, 11),
                              (11, 12),
                              (12, 13),
                              (13, 14),
                              (14, 15)]

        elif graph_number == 7:
            # D-Wave Zephyr Z(1,1) graph, 12 vertices, 22 edges
            self.vertex_count = 12

            self.edge_list = [(0, 1), (0, 8), (0, 6),
                              (1, 8), (1, 10),
                              (2, 3), (2, 8), (2, 6), (2, 9), (2, 7),
                              (3, 8), (3, 10), (3, 9), (3, 11),
                              (4, 5), (4, 9), (4, 7),
                              (5, 9), (5, 11),
                              (6, 7),
                              (8, 9),
                              (10, 11)]

        elif graph_number == 8:
            # D-Wave Zephyr Z(1,2) graph, 24 vertices, 76 edges
            self.vertex_count = 24

            self.edge_list = [(0, 1), (0, 16), (0, 12), (0, 18), (0, 14),
                              (1, 16), (1, 20), (1, 18), (1, 22),
                              (2, 3), (2, 16), (2, 12), (2, 18), (2, 14),
                              (3, 16), (3, 20), (3, 18), (3, 22),
                              (4, 5), (4, 16), (4, 12), (4, 17), (4, 13), (4, 18), (4, 14), (4, 19), (4, 15),
                              (5, 16), (5, 20), (5, 17), (5, 21), (5, 18), (5, 22), (5, 19), (5, 23),
                              (6, 7), (6, 16), (6, 12), (6, 17), (6, 13), (6, 18), (6, 14), (6, 19), (6, 15),
                              (7, 16), (7, 20), (7, 17), (7, 21), (7, 18), (7, 22), (7, 19), (7, 23),
                              (8, 9), (8, 17), (8, 13), (8, 19), (8, 15),
                              (9, 17), (9, 21), (9, 19), (9, 23),
                              (10, 11), (10, 17), (10, 13), (10, 19), (10, 15),
                              (11, 17), (11, 21), (11, 19), (11, 23),
                              (12, 13),
                              (14, 15),
                              (16, 17),
                              (18, 19),
                              (20, 21),
                              (22, 23)]

        elif graph_number == 9:
            # D-Wave Zephyr Z(1,3) graph, 36 vertices, 162 edges
            self.vertex_count = 36

            self.edge_list = [(0, 1), (0, 24), (0, 18), (0, 26), (0, 20), (0, 28), (0, 22),
                              (1, 24), (1, 30), (1, 26), (1, 32), (1, 28), (1, 34),
                              (2, 3), (2, 24), (2, 18), (2, 26), (2, 20), (2, 28), (2, 22),
                              (3, 24), (3, 30), (3, 26), (3, 32), (3, 28), (3, 34),
                              (4, 5), (4, 24), (4, 18), (4, 26), (4, 20), (4, 28), (4, 22),
                              (5, 24), (5, 30), (5, 26), (5, 32), (5, 28), (5, 34),
                              (6, 7), (6, 24), (6, 18), (6, 25), (6, 19), (6, 26), (6, 20), (6, 27), (6, 21), (6, 28), (6, 22), (6, 29), (6, 23),
                              (7, 24), (7, 30), (7, 25), (7, 31), (7, 26), (7, 32), (7, 27), (7, 33), (7, 28), (7, 34), (7, 29), (7, 35),
                              (8, 9), (8, 24), (8, 18), (8, 25), (8, 19), (8, 26), (8, 20), (8, 27), (8, 21), (8, 28), (8, 22), (8, 29), (8, 23),
                              (9, 24), (9, 30), (9, 25), (9, 31), (9, 26), (9, 32), (9, 27), (9, 33), (9, 28), (9, 34), (9, 29), (9, 35),
                              (10, 11), (10, 24), (10, 18), (10, 25), (10, 19), (10, 26), (10, 20), (10, 27), (10, 21), (10, 28), (10, 22), (10, 29), (10, 23),
                              (11, 24), (11, 30), (11, 25), (11, 31), (11, 26), (11, 32), (11, 27), (11, 33), (11, 28), (11, 34), (11, 29), (11, 35),
                              (12, 13), (12, 25), (12, 19), (12, 27), (12, 21), (12, 29), (12, 23),
                              (13, 25), (13, 31), (13, 27), (13, 33), (13, 29), (13, 35),
                              (14, 15), (14, 25), (14, 19), (14, 27), (14, 21), (14, 29), (14, 23),
                              (15, 25), (15, 31), (15, 27), (15, 33), (15, 29), (15, 35),
                              (16, 17), (16, 25), (16, 19), (16, 27), (16, 21), (16, 29), (16, 23),
                              (17, 25), (17, 31), (17, 27), (17, 33), (17, 29), (17, 35),
                              (18, 19), (20, 21), (22, 23), (24, 25), (26, 27), (28, 29), (30, 31), (32, 33), (34, 35)]

        elif graph_number == 10:
            # D-Wave Zephyr Z(1,4) graph, 48 vertices, 280 edges
            self.vertex_count = 48

            self.edge_list = [(0, 1), (0, 32), (0, 24), (0, 34), (0, 26), (0, 36), (0, 28), (0, 38), (0, 30),
                              (1, 32), (1, 40), (1, 34), (1, 42), (1, 36), (1, 44), (1, 38), (1, 46),
                              (2, 3), (2, 32), (2, 24), (2, 34), (2, 26), (2, 36), (2, 28), (2, 38), (2, 30),
                              (3, 32), (3, 40), (3, 34), (3, 42), (3, 36), (3, 44), (3, 38), (3, 46),
                              (4, 5), (4, 32), (4, 24), (4, 34), (4, 26), (4, 36), (4, 28), (4, 38), (4, 30),
                              (5, 32), (5, 40), (5, 34), (5, 42), (5, 36), (5, 44), (5, 38), (5, 46),
                              (6, 7), (6, 32), (6, 24), (6, 34), (6, 26), (6, 36), (6, 28), (6, 38), (6, 30),
                              (7, 32), (7, 40), (7, 34), (7, 42), (7, 36), (7, 44), (7, 38), (7, 46),
                              (8, 9), (8, 32), (8, 24), (8, 33), (8, 25), (8, 34), (8, 26), (8, 35), (8, 27), (8, 36), (8, 28), (8, 37), (8, 29), (8, 38), (8, 30), (8, 39), (8, 31),
                              (9, 32), (9, 40), (9, 33), (9, 41), (9, 34), (9, 42), (9, 35), (9, 43), (9, 36), (9, 44), (9, 37), (9, 45), (9, 38), (9, 46), (9, 39), (9, 47),
                              (10, 11), (10, 32), (10, 24), (10, 33), (10, 25), (10, 34), (10, 26), (10, 35), (10, 27), (10, 36), (10, 28), (10, 37), (10, 29), (10, 38), (10, 30), (10, 39), (10, 31),
                              (11, 32), (11, 40), (11, 33), (11, 41), (11, 34), (11, 42), (11, 35), (11, 43), (11, 36), (11, 44), (11, 37), (11, 45), (11, 38), (11, 46), (11, 39), (11, 47),
                              (12, 13), (12, 32), (12, 24), (12, 33), (12, 25), (12, 34), (12, 26), (12, 35), (12, 27), (12, 36), (12, 28), (12, 37), (12, 29), (12, 38), (12, 30), (12, 39), (12, 31),
                              (13, 32), (13, 40), (13, 33), (13, 41), (13, 34), (13, 42), (13, 35), (13, 43), (13, 36), (13, 44), (13, 37), (13, 45), (13, 38), (13, 46), (13, 39), (13, 47),
                              (14, 15), (14, 32), (14, 24), (14, 33), (14, 25), (14, 34), (14, 26), (14, 35), (14, 27), (14, 36), (14, 28), (14, 37), (14, 29), (14, 38), (14, 30), (14, 39), (14, 31),
                              (15, 32), (15, 40), (15, 33), (15, 41), (15, 34), (15, 42), (15, 35), (15, 43), (15, 36), (15, 44), (15, 37), (15, 45), (15, 38), (15, 46), (15, 39), (15, 47),
                              (16, 17), (16, 33), (16, 25), (16, 35), (16, 27), (16, 37), (16, 29), (16, 39), (16, 31),
                              (17, 33), (17, 41), (17, 35), (17, 43), (17, 37), (17, 45), (17, 39), (17, 47),
                              (18, 19), (18, 33), (18, 25), (18, 35), (18, 27), (18, 37), (18, 29), (18, 39), (18, 31),
                              (19, 33), (19, 41), (19, 35), (19, 43), (19, 37), (19, 45), (19, 39), (19, 47),
                              (20, 21), (20, 33), (20, 25), (20, 35), (20, 27), (20, 37), (20, 29), (20, 39), (20, 31),
                              (21, 33), (21, 41), (21, 35), (21, 43), (21, 37), (21, 45), (21, 39), (21, 47),
                              (22, 23), (22, 33), (22, 25), (22, 35), (22, 27), (22, 37), (22, 29), (22, 39), (22, 31),
                              (23, 33), (23, 41), (23, 35), (23, 43), (23, 37), (23, 45), (23, 39), (23, 47),
                              (24, 25), (26, 27), (28, 29), (30, 31), (32, 33), (34, 35), (36, 37), (38, 39), (40, 41), (42, 43), (44, 45), (46, 47)]

        elif graph_number == 11:
            # minimal graph for testing; 3 vertices 2 edges
            self.vertex_count = 3

            self.edge_list = [(0, 1), (1, 2)]

        else:

            print('Bad graph_number in GraphProperties initialization -- no graph corresponding to your choice exists.')

        self.edge_count = len(self.edge_list)


def main():
    # this is a debugging tool to make sure everything looks right!
    for graph_number in range(1, 12):
        g = GraphProperties(graph_number=graph_number)
        print('****')
        print('graph', graph_number, 'has', g.vertex_count, 'vertices and', g.edge_count, 'edges.')
        print('edge_list is', g.edge_list)


if __name__ == "__main__":
    sys.exit(main())
