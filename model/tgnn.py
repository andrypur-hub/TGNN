import torch
import torch.nn as nn

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

        # neighbor attention
        self.attn = nn.Linear(dim*2, 1)

        # fraud classifier head
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.ReLU(),
            nn.Linear(dim//2, 1)
        )

        self.norm = nn.LayerNorm(dim)

    # ===== neighbor attention =====
    def neighbor_attention(self, h, neigh_list):

        if len(neigh_list) == 0:
            return torch.zeros_like(h)

        neigh = torch.stack(neigh_list)
        h_repeat = h.repeat(neigh.size(0),1)

        score = self.attn(torch.cat([h_repeat, neigh], dim=-1))
        weight = torch.softmax(score.squeeze(), dim=0)

        agg = torch.sum(weight.unsqueeze(-1) * neigh, dim=0)
        return agg.unsqueeze(0)

    # ===== message passing =====
    def forward(self, hu, hv, neigh_u, neigh_v, x):

        zu = torch.cat([hu, neigh_u, x], dim=-1)
        zv = torch.cat([hv, neigh_v, x], dim=-1)

        hu_new = self.norm(hu + self.mlp(zu))
        hv_new = self.norm(hv + self.mlp(zv))

        # edge interaction score (structure learning)
        score = torch.sum(hu_new * hv_new, dim=-1)

        return hu_new, hv_new, score

    # ===== fraud prediction =====
    def predict(self, h):
        return self.classifier(h)
