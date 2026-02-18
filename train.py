import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

from data.loader.elliptic_loader import load_elliptic_events
from model.memory import NodeMemory
from model.tgnn import TGNN
from model.node_classifier import NodeClassifier
from model.evaluator import Evaluator


DATA_PATH = "/content/drive/MyDrive/Datasets/elliptic_bitcoin_dataset"

print("Loading Elliptic dataset...")
events_by_time, labels_by_time, n_nodes = load_elliptic_events(DATA_PATH)

DIM = 64
tgnn = TGNN(DIM)
memory = NodeMemory(n_nodes, DIM)
classifier = NodeClassifier(DIM)

optimizer = torch.optim.Adam(
    list(tgnn.parameters()) + list(classifier.parameters()),
    lr=1e-3
)

TRAIN_TIME = 34
EPOCHS = 5


# =========================
# TRAINING (PAST)
# =========================
for epoch in range(EPOCHS):

    total_loss = 0
    steps = 0

    for t in sorted(events_by_time.keys()):

        events = events_by_time[t]

        # ingest always
        for e in events:
            hu = memory.get([e.src])
            hv = memory.get([e.dst])
            x = torch.tensor([[e.x[0]]], dtype=torch.float)

            hu_new, hv_new, _ = tgnn(hu, hv, x)

            memory.update([e.src], hu_new)
            memory.update([e.dst], hv_new)

        # TRAIN only past
        if t > TRAIN_TIME:
            continue

        if t not in labels_by_time:
            continue

        nodes, labels = zip(*labels_by_time[t])

        h = memory.get(list(nodes))
        logits = classifier(h).squeeze()
        y = torch.tensor(labels, dtype=torch.float)

        loss = F.binary_cross_entropy_with_logits(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    print(f"Epoch {epoch+1}/{EPOCHS} Train Loss: {total_loss/steps:.4f}")


# =========================
# TEST (FUTURE PREDICTION)
# =========================
print("\n===== EVALUATION ON FUTURE GRAPH =====")

evaluator = Evaluator()

for t in sorted(events_by_time.keys()):

    events = events_by_time[t]

    # memory still evolves
    for e in events:
        hu = memory.get([e.src])
        hv = memory.get([e.dst])
        x = torch.tensor([[e.x[0]]], dtype=torch.float)

        hu_new, hv_new, _ = tgnn(hu, hv, x)
        memory.update([e.src], hu_new)
        memory.update([e.dst], hv_new)

    # evaluate only future
    if t <= TRAIN_TIME:
        continue

    if t not in labels_by_time:
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
