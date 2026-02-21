import torch
import torch.nn as nn
import torch.nn.functional as F

class TGNN(nn.Module):

    def __init__(self, dim, feature_dim=165):
        super().__init__()

        self.dim = dim
        self.feature_dim = feature_dim

        # ================= MEMORY UPDATE =================
        self.mlp = nn.Sequential(
            nn.Linear(dim*2 + feature_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.norm = nn.LayerNorm(dim)

        # ================= FRAUD CLASSIFIER =================
        # node embedding + raw transaction feature
        self.classifier = nn.Sequential(
            nn.Linear(dim + feature_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

    # ===== GRAPH TEMPORAL UPDATE =====
    def forward(self, hu, hv, neigh_u, neigh_v, x):

        zu = torch.cat([hu, neigh_u, x], dim=-1)
        zv = torch.cat([hv, neigh_v, x], dim=-1)

        hu_new = self.norm(hu + self.mlp(zu))
        hv_new = self.norm(hv + self.mlp(zv))

        return hu_new, hv_new

    # ===== FRAUD PREDICTION HEAD (PENTING) =====
    def predict(self, h, x):
        z = torch.cat([h, x], dim=-1)
        return self.classifier(z)
