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

        self.EPSILON = 0.5              # this is the boundary between a draw and a win

        self.NUM_READS_SA = 1000        # this is for simulated annealing

        # These are parameters related to the use of QC hardware, if you're not using QC you can just leave these
        # The defaults here are no shimming, no gauge transforms, only use M=1 automorphism, and collect a lot of
        # samples (N=1000)

        self.USE_QC = False             # set to False if you just want to use e.g. simulated annealer
        self.USE_MOCK_DWAVE_SAMPLER = False   # set to True if you want a software version of the hardware (doesn't sample like the HW tho so don't trust it, just for debugging)
        self.QC_SOLVER_TO_USE = 'Advantage2_prototype2.6'   # modify if you want to use a different QC

        self.NUMBER_OF_CHIP_RUNS = 1    # this is M
        self.NUM_READS_QC = 1000        # this is N
        self.ANNEAL_TIME_IN_NS = 5      # this is the fastest the QC can sweep

        self.USE_GAUGE_TRANSFORM = False
        self.USE_SHIM = False

        self.ALPHA_PHI = 0.00001
        self.SHIM_ITERATIONS = 10


class MinimalAdjudicationParameters(object):
    def __init__(self):
        self.EPSILON = 0.5              # this is the boundary between a draw and a win
        self.USE_QC = False
        self.NUM_READS_SA = 1000        # this is for simulated annealing
