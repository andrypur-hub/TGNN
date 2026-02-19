import torch
import torch.nn as nn

class TGNN(nn.Module):

    def __init__(self, dim, feature_dim=165):
        super().__init__()

        self.dim = dim

        self.mlp = nn.Sequential(
            nn.Linear(dim*2 + feature_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, hu, hv, neigh_u, neigh_v, x):

        zu = torch.cat([hu, neigh_u, x], dim=-1)
        zv = torch.cat([hv, neigh_v, x], dim=-1)

        hu_new = self.norm(hu + self.mlp(zu))
        hv_new = self.norm(hv + self.mlp(zv))

        score = torch.sum(hu_new * hv_new, dim=-1)

        return hu_new, hv_new, score
