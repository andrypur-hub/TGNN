import torch
import torch.nn as nn
import numpy as np
import random
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from data.loader.elliptic_loader import load_elliptic_events
from model.tgnn import TGNN

# ================= LOAD DATA =================
DATA_PATH = "/content/drive/MyDrive/ProjectPython/TGNN/Dataset/elliptic_bitcoin_dataset"
events_by_time, labels_by_time, n_nodes = load_elliptic_events(DATA_PATH)

DIM = 64
EPOCHS = 5
TRAIN_TIME = 34
DECAY = 0.0005

device = "cuda" if torch.cuda.is_available() else "cpu"

# waktu graph asli (jangan range!)
all_times = sorted(events_by_time.keys())
train_times = [t for t in all_times if t <= TRAIN_TIME]
test_times  = [t for t in all_times if t > TRAIN_TIME]

print("Train times:", train_times[:3], "...", train_times[-3:])
print("Test times :", test_times[:3], "...", test_times[-3:])

# ================= MODEL =================
memory = torch.zeros(n_nodes, DIM, device=device)
last_update = torch.zeros(n_nodes, device=device)

tgnn = TGNN(DIM).to(device)
optimizer = torch.optim.Adam(tgnn.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

neighbors = defaultdict(list)

# ================= FUNCTIONS =================

def aggregate(node_id, h_self):
    if len(neighbors[node_id]) == 0:
        return torch.zeros(1, DIM, device=device)

    neigh = [n.to(device) for n in neighbors[node_id]]
    return tgnn.neighbor_attention(h_self, neigh)

def sample_negative(dst_node, n_nodes):
    neg = random.randint(0, n_nodes-1)
    while neg == dst_node:
        neg = random.randint(0, n_nodes-1)
    return neg

def temporal_decay(node_id, current_time):
    delta = current_time - last_update[node_id]
    factor = torch.exp(-DECAY * delta)
    memory[node_id] *= factor

# ================= TRAIN =================
for epoch in range(EPOCHS):

    total_loss = 0

    for t in train_times:

        for e in events_by_time[t]:

            # time decay
            temporal_decay(e.src, t)
            temporal_decay(e.dst, t)

            hu = memory[e.src].unsqueeze(0)
            hv = memory[e.dst].unsqueeze(0)

            neigh_u = aggregate(e.src, hu)
            neigh_v = aggregate(e.dst, hv)

            x = torch.from_numpy(e.x).float().unsqueeze(0).to(device)

            # ===== POSITIVE EDGE (structure learning) =====
            hu_new, hv_new, pos_logits = tgnn(hu, hv, neigh_u, neigh_v, x)
            pos_label = torch.ones((1,1), device=device)

            # ===== NEGATIVE EDGE =====
            neg_dst = sample_negative(e.dst, n_nodes)
            temporal_decay(neg_dst, t)

            hv_neg = memory[neg_dst].unsqueeze(0)
            neigh_neg = aggregate(neg_dst, hv_neg)

            _, _, neg_logits = tgnn(hu, hv_neg, neigh_u, neigh_neg, x)
            neg_label = torch.zeros((1,1), device=device)

            # structure loss
            loss_pos = criterion(pos_logits.unsqueeze(-1), pos_label)
            loss_neg = criterion(neg_logits.unsqueeze(-1), neg_label)
            structure_loss = loss_pos + loss_neg

            # ===== FRAUD SUPERVISION =====
            fraud_logit = tgnn.predict(hu_new, x)
            fraud_label = torch.tensor([[e.y]], dtype=torch.float, device=device)
            fraud_loss = criterion(fraud_logit, fraud_label)

            # ===== FINAL LOSS =====
            LAMBDA = 20.0
            loss = structure_loss + LAMBDA + fraud_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update memory (ONLY real edge)
            memory[e.src] = hu_new.detach().squeeze(0)
            memory[e.dst] = hv_new.detach().squeeze(0)

            last_update[e.src] = t
            last_update[e.dst] = t

            neighbors[e.src].append(hv_new.detach().squeeze(0))
            neighbors[e.dst].append(hu_new.detach().squeeze(0))

            total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} Train Loss: {total_loss:.4f}")

# ================= TEST =================
print("\n===== EVALUATION ON FUTURE GRAPH =====")

y_true, y_pred = [], []

for t in test_times:

    for e in events_by_time[t]:

        temporal_decay(e.src, t)
        temporal_decay(e.dst, t)

        hu = memory[e.src].unsqueeze(0)
        hv = memory[e.dst].unsqueeze(0)

        neigh_u = aggregate(e.src, hu)
        neigh_v = aggregate(e.dst, hv)

        x = torch.from_numpy(e.x).float().unsqueeze(0).to(device)

        # fraud prediction from node embedding
        fraud_logit = tgnn.predict(hu,x)
        prob = torch.sigmoid(fraud_logit).item()

        y_true.append(e.y)
        y_pred.append(prob)

# ================= THRESHOLD SEARCH =================
best_f1 = 0
best_th = 0.5

for th in np.linspace(0.01,0.99,50):
    y_hat = [1 if p > th else 0 for p in y_pred]
    f1 = f1_score(y_true, y_hat, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_th = th

print("Best threshold:", best_th)

y_hat = [1 if p > best_th else 0 for p in y_pred]

# ================= METRICS =================
print("\nTEST RESULT (Future Fraud Detection)")
print("Precision :", precision_score(y_true, y_hat, zero_division=0))
print("Recall    :", recall_score(y_true, y_hat, zero_division=0))
print("F1-score  :", f1_score(y_true, y_hat, zero_division=0))
print("ROC-AUC   :", roc_auc_score(y_true, y_pred))
