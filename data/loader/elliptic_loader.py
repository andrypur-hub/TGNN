import pandas as pd
from collections import defaultdict

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

    # map tx → node id
    for tx in nodes.txId:
        idx.get(f"tx_{tx}")

    node_time = dict(zip(nodes.txId, nodes.time))
    node_label = dict(zip(nodes.txId, nodes["class"]))

    # ---------- build temporal events ----------
    events_by_time = defaultdict(list)

    for _, r in edge.iterrows():
        src = idx.get(f"tx_{r['txId1']}")
        dst = idx.get(f"tx_{r['txId2']}")
        t = int(node_time[r['txId1']])

        events_by_time[t].append(GraphEvent(src, dst, t, [0.0], 0))

    # ---------- build node labels ----------
    labels_by_time = defaultdict(list)

    for _, r in nodes.iterrows():
        node = idx.get(f"tx_{r['txId']}")
        t = int(r["time"])

        if r["class"] != 2:  # ignore unknown
            y = 1 if r["class"] == 1 else 0
            labels_by_time[t].append((node, y))

    print("Total nodes:", idx.counter)
    print("Timesteps:", len(events_by_time))

    return events_by_time, labels_by_time, idx.counter
