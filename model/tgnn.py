import torch
import torch.nn as nn

class TGNN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.msg = nn.Linear(dim*2 + 1, dim)
        self.gru = nn.GRUCell(dim, dim)
        self.pred = nn.Linear(dim*2, 1)

    def forward(self, hu, hv, x):
        m = torch.cat([hu, hv, x], dim=-1)
        m = self.msg(m)

        hu_new = self.gru(m, hu)
        hv_new = self.gru(m, hv)

        score = self.pred(torch.cat([hu_new, hv_new], dim=-1))
        return hu_new, hv_new, score
