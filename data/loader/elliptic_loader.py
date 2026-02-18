import pandas as pd
from graph.event import GraphEvent
from graph.indexer import NodeIndexer

def load_elliptic_events(path):

    feat = pd.read_csv(f"{path}/elliptic_txs_features.csv", header=None)
    cls  = pd.read_csv(f"{path}/elliptic_txs_classes.csv")
    edge = pd.read_csv(f"{path}/elliptic_txs_edgelist.csv")

    feat = feat.rename(columns={0:"txId",1:"time"})
    nodes = feat.merge(cls, on="txId", how="left")

    nodes["class"] = nodes["class"].replace("unknown","2")
    nodes["class"] = nodes["class"].astype(int)

    idx = NodeIndexer()
    node_time = dict(zip(nodes.txId, nodes.time))
    node_label = dict(zip(nodes.txId, nodes["class"]))

    events = []

    for _, r in edge.iterrows():
        src = idx.get(f"tx_{r['txId1']}")
        dst = idx.get(f"tx_{r['txId2']}")

        t = float(node_time[r['txId1']])
        y = 1 if node_label[r['txId1']] == 1 else 0

        events.append(GraphEvent(src, dst, t, [0.0], y))

    events.sort(key=lambda e: e.t)

    print("Elliptic events:", len(events))
    print("Nodes:", idx.counter)

    return events, idx.counter
