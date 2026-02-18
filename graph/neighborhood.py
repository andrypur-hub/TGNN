from collections import defaultdict

class NeighborFinder:
    def __init__(self):
        self.neighbors = defaultdict(set)

    def add_edge(self, u, v):
        self.neighbors[u].add(v)
        self.neighbors[v].add(u)

    def get_neighbors(self, node):
        return list(self.neighbors[node])
