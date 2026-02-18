import torch
import torch.nn as nn


class TGNN(nn.Module):
    """
    Temporal Graph Neural Network (neighbor-aware version)

    Update rule:
    h_u(t+1) = f(h_u(t), h_v(t), mean(neigh_u), x_uv)
    h_v(t+1) = f(h_v(t), h_u(t), mean(neigh_v), x_uv)
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # input: hu, hv, neighbor, edge_feature
        self.mlp = nn.Sequential(
            nn.Linear(dim * 3 + 1, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        # optional normalization (stabilize memory)
        self.norm = nn.LayerNorm(dim)

    def forward(self, hu, hv, neigh_u, neigh_v, x):
        """
        hu : embedding node u  [1, dim]
        hv : embedding node v  [1, dim]
        neigh_u : mean neighbor embedding u [1, dim]
        neigh_v : mean neighbor embedding v [1, dim]
        x : edge feature [1, 1]
        """

        # update u
        zu = torch.cat([hu, hv, neigh_u, x], dim=1)
        hu_new = self.mlp(zu)
        hu_new = self.norm(hu + hu_new)   # residual connection

        # update v
        zv = torch.cat([hv, hu, neigh_v, x], dim=1)
        hv_new = self.mlp(zv)
        hv_new = self.norm(hv + hv_new)

        return hu_new, hv_new, None
