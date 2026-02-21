import torch
import torch.nn as nn
import numpy as np
import random
import os
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from data.loader.elliptic_loader import load_elliptic_events
from model.tgnn import TGNN

# ================= CONFIG =================
DATA_PATH = "/content/drive/MyDrive/ProjectPython/TGNN/Dataset/elliptic_bitcoin_dataset"
CKPT_PATH = "/content/drive/MyDrive/ProjectPython/TGNN/tgnn_ckpt.pt"

DIM = 64
EPOCHS = 5
TRAIN_TIME = 34
DECAY = 0.0005
LAMBDA = 20.0
BATCH_SIZE = 200        # <<< SPEED KEY
MAX_NEIGH = 20          # <<< LIMIT HISTORY

device = "cuda" if torch.cuda.is_available() else "cpu"

# ================= LOAD DATA =================
events_by_time, labels_by_time, n_nodes = load_elliptic_events(DATA_PATH)

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

# ================= RESUME =================
start_epoch = 0
if os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location=device)

    tgnn.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["opt"])
    memory = ckpt["memory"]
    last_update = ckpt["last_update"]
    neighbors = ckpt["neighbors"]

    start_epoch = ckpt["epoch"] + 1
    print("Resumed from epoch", start_epoch)

# ================= FUNCTIONS =================
def aggregate(node_id, h_self):
    if len(neighbors[node_id]) == 0:
        return torch.zeros(1, DIM, device=device)

    neigh = neighbors[node_id][-MAX_NEIGH:]
    neigh = [n.detach() for n in neigh]
    return tgnn.neighbor_attention(h_self, neigh)

def sample_negative(dst_node):
    neg = random.randint(0, n_nodes-1)
    while neg == dst_node:
        neg = random.randint(0, n_nodes-1)
    return neg

def temporal_decay(node_id, current_time):
    delta = current_time - last_update[node_id]
    memory[node_id] *= torch.exp(-DECAY * delta)

# ================= TRAIN =================
for epoch in range(start_epoch, EPOCHS):

    total_loss = 0
    batch_loss = 0
    batch_count = 0

    for t in train_times:

        print(f"Epoch {epoch+1} timestep {t}")

        for e in events_by_time[t]:

            # decay
            temporal_decay(e.src, t)
            temporal_decay(e.dst, t)

            hu = memory[e.src].detach().clone().unsqueeze(0)
            hv = memory[e.dst].detach().clone().unsqueeze(0)

            neigh_u = aggregate(e.src, hu)
            neigh_v = aggregate(e.dst, hv)

            x = torch.from_numpy(e.x).float().unsqueeze(0).to(device)

            # ===== POS EDGE =====
            hu_new, hv_new = tgnn(hu, hv, neigh_u, neigh_v, x)
            pos_logits = tgnn.predict(hu_new, x)
            pos_label = torch.ones((1,1), device=device)

            # ===== NEG EDGE =====
            neg_dst = sample_negative(e.dst)
            temporal_decay(neg_dst, t)

            hv_neg = memory[neg_dst].detach().clone().unsqueeze(0)
            neigh_neg = aggregate(neg_dst, hv_neg)

            hu_neg, _ = tgnn(hu, hv_neg, neigh_u, neigh_neg, x)
            neg_logits = tgnn.predict(hu_neg, x)
            neg_label = torch.zeros((1,1), device=device)

            structure_loss = criterion(pos_logits, pos_label) + \
                             criterion(neg_logits, neg_label)

            fraud_label = torch.tensor([[e.y]], dtype=torch.float, device=device)
            fraud_loss = criterion(tgnn.predict(hu_new, x), fraud_label)

            loss = structure_loss + LAMBDA * fraud_loss

            # ===== MINI BATCH BACKPROP =====
            batch_loss += loss
            batch_count += 1

            if batch_count >= BATCH_SIZE:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                total_loss += batch_loss.item()
                batch_loss = 0
                batch_count = 0

            # ===== UPDATE MEMORY =====
            with torch.no_grad():
                memory[e.src] = hu_new.squeeze(0)
                memory[e.dst] = hv_new.squeeze(0)
                last_update[e.src] = t
                last_update[e.dst] = t

                neighbors[e.src].append(hv_new.squeeze(0))
                neighbors[e.dst].append(hu_new.squeeze(0))

    print(f"Epoch {epoch+1} loss: {total_loss:.4f}")

    # ===== SAVE CHECKPOINT =====
    torch.save({
        "epoch": epoch,
        "model": tgnn.state_dict(),
        "opt": optimizer.state_dict(),
        "memory": memory,
        "last_update": last_update,
        "neighbors": neighbors
    }, CKPT_PATH)

    print("Checkpoint saved")
