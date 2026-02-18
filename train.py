

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from collections import defaultdict

from graph.event import GraphEvent
from graph.indexer import NodeIndexer


import torch
import torch.nn.functional as F

from data.loader.elliptic_loader import load_elliptic_events
from model.memory import NodeMemory
from model.tgnn import TGNN
from model.node_classifier import NodeClassifier

DATA_PATH = "/content/drive/MyDrive/Dataset/elliptic_bitcoin_dataset"

print("Loading Elliptic dataset...")
events_by_time, n_nodes = load_elliptic_events(DATA_PATH)

DIM = 64
tgnn = TGNN(DIM)
memory = NodeMemory(n_nodes, DIM)
classifier = NodeClassifier(DIM)

optimizer = torch.optim.Adam(
    list(tgnn.parameters()) + list(classifier.parameters()),
    lr=1e-3
)

EPOCHS = 3

for epoch in range(EPOCHS):

    total_loss = 0

    for t, events in events_by_time.items():

        # 1️⃣ ingest transactions
        for e in events:
            hu = memory.get([e.src])
            hv = memory.get([e.dst])
            x = torch.tensor([[e.x[0]]], dtype=torch.float)

            hu_new, hv_new, _ = tgnn(hu, hv, x)
            memory.update([e.src], hu_new)
            memory.update([e.dst], hv_new)

        # 2️⃣ node prediction
        nodes = []
        labels = []

        for e in events:
            nodes.append(e.src)
            labels.append(e.y)

        nodes = list(set(nodes))

        h = memory.get(nodes)
        logits = classifier(h).squeeze()

        y = torch.tensor(labels[:len(logits)], dtype=torch.float)

        loss = F.binary_cross_entropy_with_logits(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {total_loss:.4f}")

print("Training selesai")
