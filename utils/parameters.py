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
        self.USE_PRECOMPUTED_RESULTS = True
        self.GET_PRECOMPUTED_RESULTS = True
