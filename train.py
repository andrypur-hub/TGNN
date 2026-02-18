import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

from data.loader.elliptic_loader import load_elliptic_events
from model.memory import NodeMemory
from model.tgnn import TGNN
from model.node_classifier import NodeClassifier
from model.evaluator import Evaluator


# ====== PATH DATASET ======
DATA_PATH = "/content/drive/MyDrive/ProjectPython/TGNN/Dataset/elliptic_bitcoin_dataset"

print("Loading Elliptic dataset...")
events_by_time, labels_by_time, n_nodes = load_elliptic_events(DATA_PATH)


# ====== MODEL INIT ======
DIM = 64
tgnn = TGNN(DIM)
memory = NodeMemory(n_nodes, DIM)
classifier = NodeClassifier(DIM)

optimizer = torch.optim.Adam(
    list(tgnn.parameters()) + list(classifier.parameters()),
    lr=1e-3
)

EPOCHS = 5


# ====== TRAINING ======
for epoch in range(EPOCHS):

    total_loss = 0
    steps = 0
    evaluator = Evaluator()

    for t in sorted(events_by_time.keys()):

        events = events_by_time[t]

        # ======================
        # 1) INGEST TRANSACTIONS
        # ======================
        for e in events:
            hu = memory.get([e.src])
            hv = memory.get([e.dst])
            x = torch.tensor([[e.x[0]]], dtype=torch.float)

            hu_new, hv_new, _ = tgnn(hu, hv, x)

            memory.update([e.src], hu_new)
            memory.update([e.dst], hv_new)

        # ======================
        # 2) NODE CLASSIFICATION
        # ======================
        if t not in labels_by_time:
            continue

        nodes = []
        labels = []

        for node, y in labels_by_time[t]:
            nodes.append(node)
            labels.append(y)

        h = memory.get(nodes)
        logits = classifier(h).squeeze()

        y = torch.tensor(labels, dtype=torch.float)

        loss = F.binary_cross_entropy_with_logits(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1

        # collect metrics
        evaluator.add_batch(logits, y)


    precision, recall, f1, auc = evaluator.compute()

    print(f"""
Epoch {epoch+1}/{EPOCHS}
Loss      : {total_loss/steps:.4f}
Precision : {precision:.4f}
Recall    : {recall:.4f}
F1-score  : {f1:.4f}
ROC-AUC   : {auc:.4f}
""")


print("Training selesai")
