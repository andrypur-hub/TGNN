import torch
import torch.nn as nn
import torch.nn.functional as F

class TGNN(nn.Module):

    def __init__(self, dim, feature_dim=165):
        super().__init__()

        self.dim = dim

        # ===== MESSAGE UPDATE (temporal interaction) =====
        self.mlp = nn.Sequential(
            nn.Linear(dim*2 + feature_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        # ===== NODE NORMALIZATION =====
        self.norm = nn.LayerNorm(dim)

        # ===== FRAUD CLASSIFIER =====
        self.classifier = nn.Sequential(
            nn.Linear(dim + feature_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

        # ===== TEMPORAL NEIGHBOR ATTENTION =====
        self.attn_q = nn.Linear(dim, dim)
        self.attn_k = nn.Linear(dim, dim)
        self.attn_v = nn.Linear(dim, dim)


    # =====================================================
    # Neighbor aggregation with attention
    # =====================================================
    def neighbor_attention(self, h_self, neigh_list):

        if len(neigh_list) == 0:
            return torch.zeros_like(h_self)

        neigh = torch.stack(neigh_list).unsqueeze(0)   # (1, N, D)

        q = self.attn_q(h_self).unsqueeze(1)            # (1,1,D)
        k = self.attn_k(neigh)                          # (1,N,D)
        v = self.attn_v(neigh)                          # (1,N,D)

        score = torch.matmul(q, k.transpose(1,2)) / (self.dim ** 0.5)
        attn = torch.softmax(score, dim=-1)

        out = torch.matmul(attn, v).squeeze(1)          # (1,D)
        return out


    # =====================================================
    # Temporal message passing
    # =====================================================
    def forward(self, hu, hv, neigh_u, neigh_v, x):

        # combine node state + neighbor state + edge feature
        zu = torch.cat([hu, neigh_u, x], dim=-1)
        zv = torch.cat([hv, neigh_v, x], dim=-1)

        # message update
        hu_new = self.norm(hu + self.mlp(zu))
        hv_new = self.norm(hv + self.mlp(zv))

        return hu_new, hv_new


    # =====================================================
    # Fraud prediction head
    # =====================================================
    def predict(self, h, x):

        z = torch.cat([h, x], dim=-1)
        return self.classifier(z)
