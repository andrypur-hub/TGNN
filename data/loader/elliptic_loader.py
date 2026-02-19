import pandas as pd
from collections import defaultdict
from graph.event import GraphEvent
from graph.indexer import NodeIndexer


def load_elliptic_events(path):

    # ===== load files =====
    feat = pd.read_csv(f"{path}/elliptic_txs_features.csv", header=None)
    cls  = pd.read_csv(f"{path}/elliptic_txs_classes.csv")
    edge = pd.read_csv(f"{path}/elliptic_txs_edgelist.csv")

    feat = feat.rename(columns={0: "txId", 1: "time"})

    # merge class
    nodes = feat.merge(cls, on="txId", how="left")

    # unknown → -1
    nodes["class"] = nodes["class"].replace("unknown", -1)
    nodes["class"] = nodes["class"].astype(int)

    # ===== indexing =====
    idx = NodeIndexer()

    for tx in nodes.txId:
        idx.get(f"tx_{tx}")

    node_time = dict(zip(nodes.txId, nodes.time))
    node_label = dict(zip(nodes.txId, nodes["class"]))

    # ===== node features =====
    feature_cols = list(feat.columns[2:])  # 166 features
    node_feature = {
        r.txId: r[feature_cols].values.astype(float)
        for _, r in feat.iterrows()
    }

    # ===== temporal events =====
    events_by_time = defaultdict(list)

    for _, r in edge.iterrows():
        src_id = r["txId1"]
        dst_id = r["txId2"]

        if src_id not in node_time:
            continue

        src = idx.get(f"tx_{src_id}")
        dst = idx.get(f"tx_{dst_id}")
        t = int(node_time[src_id])

        x = node_feature[src_id]

        events_by_time[t].append(GraphEvent(src, dst, t, x, 0))

    # ===== labels =====
    labels_by_time = defaultdict(list)

    for _, r in nodes.iterrows():
        if r["class"] in [1, 2]:  # labeled only
            node = idx.get(f"tx_{r['txId']}")
            t = int(r["time"])
            y = 1 if r["class"] == 1 else 0
            labels_by_time[t].append((node, y))

    print("Total nodes:", idx.counter)
    print("Timesteps:", len(events_by_time))

    return events_by_time, labels_by_time, idx.counter
