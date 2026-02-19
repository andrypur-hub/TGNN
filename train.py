import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

from data.loader.elliptic_loader import load_elliptic_events
from model.tgnn import TGNN

# ================= LOAD DATA =================
DATA_PATH = "/content/drive/MyDrive/ProjectPython/TGNN/Dataset/elliptic_bitcoin_dataset"

events_by_time, labels_by_time, n_nodes = load_elliptic_events(DATA_PATH)

DIM = 64
EPOCHS = 5
TRAIN_TIME = 34   # train pakai masa lalu (1..34), test masa depan (35..49)

device = "cuda" if torch.cuda.is_available() else "cpu"

# waktu asli graph (PENTING!!)
all_times = sorted(events_by_time.keys())
train_times = [t for t in all_times if t <= TRAIN_TIME]
test_times  = [t for t in all_times if t >  TRAIN_TIME]

print("Train times:", train_times[:3], "...", train_times[-3:])
print("Test times :", test_times[:3], "...", test_times[-3:])

# ================= MODEL =================
memory = torch.zeros(n_nodes, DIM, device=device)

tgnn = TGNN(DIM).to(device)
optimizer = torch.optim.Adam(tgnn.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

neighbors = defaultdict(list)

def aggregate(node_id):
    if len(neighbors[node_id]) == 0:
        return torch.zeros(1, DIM, device=device)
    return torch.mean(torch.stack(neighbors[node_id]), dim=0, keepdim=True)

# ===== NEGATIVE SAMPLING =====
import random

def sample_negative(dst_node, n_nodes):
    neg = random.randint(0, n_nodes-1)
    while neg == dst_node:
        neg = random.randint(0, n_nodes-1)
    return neg

# ================= TRAIN =================
for epoch in range(EPOCHS):

    total_loss = 0

    for t in train_times:

        for e in events_by_time[t]:

            hu = memory[e.src].unsqueeze(0)
            hv = memory[e.dst].unsqueeze(0)

            neigh_u = aggregate(e.src)
            neigh_v = aggregate(e.dst)

            x = torch.from_numpy(e.x).float().unsqueeze(0).to(device)
            y = torch.tensor([[e.y]], dtype=torch.float, device=device)

            # ===== POSITIVE EDGE =====
            hu_new, hv_new, pos_logits = tgnn(hu, hv, neigh_u, neigh_v, x)
            pos_label = torch.ones((1,1), device=device)

            # ===== NEGATIVE EDGE =====
            neg_dst = sample_negative(e.dst, n_nodes)

            hv_neg = memory[neg_dst].unsqueeze(0)
            neigh_neg = aggregate(neg_dst)

            _, _, neg_logits = tgnn(hu, hv_neg, neigh_u, neigh_neg, x)
            neg_label = torch.zeros((1,1), device=device)

            # ===== LOSS =====
            loss_pos = criterion(pos_logits.unsqueeze(-1), pos_label)
            loss_neg = criterion(neg_logits.unsqueeze(-1), neg_label)

            loss = loss_pos + loss_neg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update memory hanya saat training
            memory[e.src] = hu_new.detach().squeeze(0)
            memory[e.dst] = hv_new.detach().squeeze(0)

            neighbors[e.src].append(hv_new.detach().squeeze(0))
            neighbors[e.dst].append(hu_new.detach().squeeze(0))

            total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} Train Loss: {total_loss:.4f}")

# ================= TEST =================
print("\n===== EVALUATION ON FUTURE GRAPH =====")

y_true, y_pred = [], []

for t in test_times:

    for e in events_by_time[t]:

        hu = memory[e.src].unsqueeze(0)
        hv = memory[e.dst].unsqueeze(0)

        neigh_u = aggregate(e.src)
        neigh_v = aggregate(e.dst)

        x = torch.from_numpy(e.x).float().unsqueeze(0).to(device)

        _, _, logits = tgnn(hu, hv, neigh_u, neigh_v, x)

        prob = torch.sigmoid(logits).item()

        y_true.append(e.y)
        y_pred.append(prob)

# ================= METRICS =================
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

y_hat = [1 if p > 0.5 else 0 for p in y_pred]

print("\nTEST RESULT (Future Fraud Detection)")
print("Precision :", precision_score(y_true, y_hat, zero_division=0))
print("Recall    :", recall_score(y_true, y_hat, zero_division=0))
print("F1-score  :", f1_score(y_true, y_hat, zero_division=0))
print("ROC-AUC   :", roc_auc_score(y_true, y_pred))
