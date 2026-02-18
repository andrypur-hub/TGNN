import torch

class NodeMemory:
    def __init__(self, n_nodes, dim):
        self.mem = torch.zeros(n_nodes, dim)

    def get(self, ids):
        return self.mem[ids]

    def update(self, ids, new):
        self.mem[ids] = new.detach()
