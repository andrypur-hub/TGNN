import torch
import torch.nn as nn
import torch.nn.functional as F

class TGNN(nn.Module):

    def __init__(self, dim, feature_dim=165):
        super().__init__()

        self.dim = dim

        # message update
        self.mlp = nn.Sequential(
            nn.Linear(dim*2 + feature_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        # attention scoring antar node
        self.attn = nn.Linear(dim*2, 1)

        self.norm = nn.LayerNorm(dim)

    # ===== NEW: neighbor attention =====
    def neighbor_attention(self, h, neigh_list):

        if len(neigh_list) == 0:
            return torch.zeros_like(h)

        neigh = torch.stack(neigh_list)              # (N, D)
        h_repeat = h.repeat(neigh.size(0),1)         # (N, D)

        score = self.attn(torch.cat([h_repeat, neigh], dim=-1))  # (N,1)
        weight = torch.softmax(score.squeeze(), dim=0)           # (N)

        agg = torch.sum(weight.unsqueeze(-1) * neigh, dim=0)     # (D)

        return agg.unsqueeze(0)

    def forward(self, hu, hv, neigh_u, neigh_v, x):

        zu = torch.cat([hu, neigh_u, x], dim=-1)
        zv = torch.cat([hv, neigh_v, x], dim=-1)

        hu_new = self.norm(hu + self.mlp(zu))
        hv_new = self.norm(hv + self.mlp(zv))

        score = torch.sum(hu_new * hv_new, dim=-1)

        return hu_new, hv_new, score
