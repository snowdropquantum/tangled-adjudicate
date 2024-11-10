""" adjudication and support parameters """


class Params(object):
    def __init__(self):
        self.GRAPH_NUMBER = 2   # this is the index of the graph to use; defined in /utils/game_graph_properties.py
        # just a reminder which are which:
        # 1 = 2 vertices
        # 2 = 3 vertices in triangle
        # 3 = 4 vertices, 6 edges
        # 4 = 6 vertices, 1 hexagon, 6 edges
        # 5 = 10 vertices, 15 edges, petersen graph
        # 6 = 16 vertices, 32 edges, non-planar, tesseract

        # this is the boundary between a draw and a win; don't touch this unless you are a wizard doing wizard shit
        self.EPSILON = 0.5

        # this is for simulated annealing
        self.NUM_READS_SA = 1000000

        # These are parameters related to the use of QC hardware, if you're not using QC you can just leave these
        self.USE_QC = True
        self.USE_MOCK_DWAVE_SAMPLER = True    # set to True if you want a software version of the hardware (doesn't sample like the HW tho so don't trust it, just for debugging
        self.QC_SOLVER_TO_USE = 'Advantage2_prototype2.5'   # modify if you want to use a different QC
        self.NUM_READS_QC = 10
        self.ANNEAL_TIME_IN_NS = 5
        self.SPIN_REVERSAL_TRANSFORMS = 2
