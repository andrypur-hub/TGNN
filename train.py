import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import torch
import torch.nn.functional as F

from data.loader.elliptic_loader import load_elliptic_events
from model.memory import NodeMemory
from model.tgnn import TGNN

print("Loading Elliptic dataset...")

events, n_nodes = load_elliptic_events("/content/elliptic_bitcoin_dataset")

DIM = 64
model = TGNN(DIM)
memory = NodeMemory(n_nodes, DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 2

for epoch in range(EPOCHS):

    total_loss = 0

    for e in events:

        hu = memory.get([e.src])
        hv = memory.get([e.dst])
        x = torch.tensor([[e.x[0]]], dtype=torch.float)

        hu_new, hv_new, score = model(hu, hv, x)

        y = torch.tensor([[e.y]], dtype=torch.float)
        loss = F.binary_cross_entropy_with_logits(score, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        memory.update([e.src], hu_new)
        memory.update([e.dst], hv_new)

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {total_loss:.4f}")

print("Training selesai")
