""" adjudication and support parameters """


class Params(object):
    def __init__(self):
        self.GRAPH_NUMBER = 3
        # 1 = 2 vertices
        # 2 = 3 vertices in triangle
        # 3 = 4 vertices, 6 edges
        # 4 = 6 vertices, 1 hexagon, 6 edges
        # 5 = 10 vertices, 15 edges, petersen graph
        # 6 = 16 vertices, 32 edges, non-planar, tesseract
        self.USE_QC = True
        self.USE_PRECOMPUTED_RESULTS = True
        self.GET_PRECOMPUTED_RESULTS = True

        # self.NUM_READS_SA = 10380000   # 2076000 for graph 2
        self.NUM_READS_SA = 10000   # 5256000 for graph 3
        self.NUM_READS_QC = 10
        self.EPSILON = 0.5
        self.ANNEAL_TIME_IN_NS = 5
        self.SPIN_REVERSAL_TRANSFORMS = 2

        self.USE_MOCK_DWAVE_SAMPLER = False