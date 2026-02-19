class GraphEvent:
    """
    src : node id
    dst : node id
    t   : timestep
    x   : feature vector (list of float)
    y   : label
    """

    def __init__(self, src, dst, t, x, y):
        self.src = src
        self.dst = dst
        self.t = t
        self.x = x
        self.y = y

    def __repr__(self):
        return f"GraphEvent(src={self.src}, dst={self.dst}, t={self.t}, y={self.y})"
