class GraphEvent:
    """
    Represent one temporal interaction in the graph.

    src : source node id
    dst : destination node id
    t   : timestamp
    x   : feature vector (list or tensor)
    y   : label (0 normal, 1 fraud)
    """

    def __init__(self, src, dst, t, x, y):
        self.src = src
        self.dst = dst
        self.t = t
        self.x = x
        self.y = y

    def __repr__(self):
        return f"GraphEvent(src={self.src}, dst={self.dst}, t={self.t}, y={self.y})"

