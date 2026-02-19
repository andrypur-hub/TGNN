import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch

from data.loader.elliptic_loader import load_elliptic_events
from model.memory import NodeMemory
from model.tgnn import TGNN
from model.node_classifier import NodeClassifier
from model.evaluator import Evaluator
from model.loss import FocalLoss
from graph.neighborhood import NeighborFinder


DATA_PATH = "/content/drive/MyDrive/ProjectPython/TGNN/Dataset/elliptic_bitcoin_dataset"

print("Loading Elliptic dataset...")
events_by_time, labels_by_time, n_nodes = load_elliptic_events(DATA_PATH)


DIM = 64
tgnn = TGNN(DIM)
memory = NodeMemory(n_nodes, DIM)
classifier = NodeClassifier(DIM)
neighbors = NeighborFinder()

optimizer = torch.optim.Adam(
    list(tgnn.parameters()) + list(classifier.parameters()),
    lr=1e-3
)

criterion = FocalLoss(alpha=0.75, gamma=2.0)

TRAIN_TIME = 34
EPOCHS = 5


# ================= TRAIN =================
for epoch in range(EPOCHS):

    total_loss = 0
    steps = 0

    for t in sorted(events_by_time.keys()):

        events = events_by_time[t]

        # ingest
        for e in events:

            hu = memory.get([e.src])
            hv = memory.get([e.dst])

            nu = neighbors.get_neighbors(e.src)
            nv = neighbors.get_neighbors(e.dst)

            neigh_u = memory.get(nu).mean(dim=0, keepdim=True) if len(nu) else torch.zeros_like(hu)
            neigh_v = memory.get(nv).mean(dim=0, keepdim=True) if len(nv) else torch.zeros_like(hv)

            x = torch.tensor([e.x], dtype=torch.float)

            hu_new, hv_new, _ = tgnn(hu, hv, neigh_u, neigh_v, x)

            memory.update([e.src], hu_new)
            memory.update([e.dst], hv_new)
            neighbors.add_edge(e.src, e.dst)

        if t > TRAIN_TIME or t not in labels_by_time:
            continue

        nodes, labels = zip(*labels_by_time[t])

        h = memory.get(list(nodes))
        logits = classifier(h).squeeze()
        y = torch.tensor(labels, dtype=torch.float)

        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    print(f"Epoch {epoch+1}/{EPOCHS} Train Loss: {total_loss/steps:.4f}")


# ================= TEST =================
print("\n===== EVALUATION ON FUTURE GRAPH =====")

evaluator = Evaluator()

for t in sorted(events_by_time.keys()):

    events = events_by_time[t]

    for e in events:

        hu = memory.get([e.src])
        hv = memory.get([e.dst])

        nu = neighbors.get_neighbors(e.src)
        nv = neighbors.get_neighbors(e.dst)

        neigh_u = memory.get(nu).mean(dim=0, keepdim=True) if len(nu) else torch.zeros_like(hu)
        neigh_v = memory.get(nv).mean(dim=0, keepdim=True) if len(nv) else torch.zeros_like(hv)

        x = torch.tensor([e.x], dtype=torch.float)

        with torch.no_grad():
            hu_new, hv_new, _ = tgnn(hu, hv, neigh_u, neigh_v, x)

        memory.update([e.src], hu_new)
        memory.update([e.dst], hv_new)
        neighbors.add_edge(e.src, e.dst)

    if t <= TRAIN_TIME or t not in labels_by_time:
        continue

    nodes, labels = zip(*labels_by_time[t])

    with torch.no_grad():
        h = memory.get(list(nodes))
        logits = classifier(h).squeeze()
        y = torch.tensor(labels, dtype=torch.float)

    evaluator.add_batch(logits, y)


precision, recall, f1, auc = evaluator.compute()

print(f"""
TEST RESULT (Future Fraud Detection)
Precision : {precision:.4f}
Recall    : {recall:.4f}
F1-score  : {f1:.4f}
ROC-AUC   : {auc:.4f}
""")
