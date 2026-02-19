import pandas as pd
import numpy as np
from collections import defaultdict
from graph.event import GraphEvent
from graph.indexer import NodeIndexer

def load_elliptic_events(path):

    feat = pd.read_csv(f"{path}/elliptic_txs_features.csv", header=None)
    cls  = pd.read_csv(f"{path}/elliptic_txs_classes.csv")
    edge = pd.read_csv(f"{path}/elliptic_txs_edgelist.csv")

    feat = feat.rename(columns={0:"txId",1:"time"})
    nodes = feat.merge(cls, on="txId", how="left")

    nodes["class"] = nodes["class"].replace("unknown",2)
    nodes["class"] = nodes["class"].astype(int)

    feature_cols = list(nodes.columns[2:-1])   # 165 features
    print("Feature dim:", len(feature_cols))

    idx = NodeIndexer()

    node_time = dict(zip(nodes.txId, nodes.time))
    node_label = dict(zip(nodes.txId, nodes["class"]))
    node_feat = {row.txId: row[feature_cols].values.astype(np.float32)
                 for _, row in nodes.iterrows()}

    events_by_time = defaultdict(list)
    labels_by_time = defaultdict(list)

    for _, r in edge.iterrows():

        if r.txId1 not in node_time or r.txId2 not in node_time:
            continue

        t = int(node_time[r.txId1])

        src = idx.get(f"tx_{r.txId1}")
        dst = idx.get(f"tx_{r.txId2}")

        y = 1 if node_label[r.txId1] == 1 else 0
        x = node_feat[r.txId1]

        events_by_time[t].append(GraphEvent(src, dst, t, x, y))
        labels_by_time[t].append(y)

    print("Total nodes:", idx.counter)
    print("Timesteps:", len(events_by_time))

    return dict(sorted(events_by_time.items())), dict(sorted(labels_by_time.items())), idx.counter
