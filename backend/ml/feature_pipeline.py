"""
ml/feature_pipeline.py — converts transaction graph into a labelled feature DataFrame.

Usage:
  python ml/feature_pipeline.py
  # Reads ml/artifacts/accounts.csv + ml/artifacts/transactions.csv
  # Outputs ml/artifacts/features.csv (used by train.py)

Features extracted per account node:
  fan_in_count       - distinct senders in 5h window
  fan_in_amount      - total MYR received in window
  fan_out_count      - distinct receivers in 5h window
  fan_out_amount     - total MYR sent in window
  pass_through_ratio - fraction of inflows matched by rapid outflow (>=65%, <=30min)
  avg_pass_minutes   - average time between receive and forward
  in_degree          - total incoming edge count (all time)
  out_degree         - total outgoing edge count (all time)
  degree_ratio       - out_degree / max(in_degree, 1)
  in_out_amount_diff - fan_in_amount - fan_out_amount
  proximity_to_mule  - BFS hops to nearest known mule (-1 = unreachable)
  known_mule         - 1 if account is a seeded known mule
  is_mule            - LABEL: 1 if account type is mule, 0 otherwise
"""
import csv
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path


WINDOW_MINUTES = 300        # 5h — match prototype
PASS_WINDOW_MINUTES = 30    # max time between receive and forward
PASS_MIN_RATIO = 0.65       # minimum forwarding ratio to count as pass-through
MAX_PROXIMITY_DEPTH = 3     # BFS max hop depth


def load_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_graph(transactions: list[dict], window_minutes: int):
    """Build adjacency + edge data for the feature window."""
    now = datetime.utcnow()
    cutoff = now - timedelta(minutes=window_minutes)

    in_edges: dict[str, list[dict]] = defaultdict(list)
    out_edges: dict[str, list[dict]] = defaultdict(list)
    adj: dict[str, set] = defaultdict(set)

    # All-time degree tracking
    in_degree: dict[str, int] = defaultdict(int)
    out_degree: dict[str, int] = defaultdict(int)

    for t in transactions:
        ts_str = t.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str)
        except ValueError:
            continue

        sender = t["sender_id"]
        receiver = t["receiver_id"]
        amount = float(t["amount"])

        in_degree[receiver] += 1
        out_degree[sender] += 1
        adj[sender].add(receiver)
        adj[receiver].add(sender)

        if ts >= cutoff:
            edge = {"sender_id": sender, "receiver_id": receiver, "amount": amount, "ts": ts}
            in_edges[receiver].append(edge)
            out_edges[sender].append(edge)

    return in_edges, out_edges, adj, in_degree, out_degree


def compute_proximity(accounts: list[dict], adj: dict[str, set], max_depth: int) -> dict[str, int]:
    """BFS from all known mules, returns hop distance for every account."""
    known_mules = {a["id"] for a in accounts if a.get("known_mule") in (True, "True", "true", "1")}
    proximity = {a["id"]: -1 for a in accounts}
    for mid in known_mules:
        proximity[mid] = 0

    frontier = list(known_mules)
    visited = set(known_mules)
    depth = 0

    while frontier and depth < max_depth:
        depth += 1
        next_frontier = []
        for node in frontier:
            for neighbor in adj.get(node, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    proximity[neighbor] = depth
                    next_frontier.append(neighbor)
        frontier = next_frontier

    return proximity


def extract_features(accounts: list[dict], transactions: list[dict]) -> list[dict]:
    in_edges, out_edges, adj, in_degree, out_degree = build_graph(transactions, WINDOW_MINUTES)
    proximity = compute_proximity(accounts, adj, MAX_PROXIMITY_DEPTH)

    # Normalisation denominators
    all_fan_in = [len(set(e["sender_id"] for e in in_edges[a["id"]])) for a in accounts]
    all_fan_out = [len(set(e["receiver_id"] for e in out_edges[a["id"]])) for a in accounts]
    max_fan_in = max(all_fan_in or [1], default=1) or 1
    max_fan_out = max(all_fan_out or [1], default=1) or 1

    rows = []
    for acc in accounts:
        acc_id = acc["id"]
        ins = in_edges[acc_id]
        outs = out_edges[acc_id]

        fan_in_count = len(set(e["sender_id"] for e in ins))
        fan_out_count = len(set(e["receiver_id"] for e in outs))
        fan_in_amount = sum(e["amount"] for e in ins)
        fan_out_amount = sum(e["amount"] for e in outs)

        # Pass-through detection
        velocity_hits = 0
        velocity_minutes_sum = 0.0
        for in_e in ins:
            match = next(
                (
                    o for o in outs
                    if o["ts"] >= in_e["ts"]
                    and (o["ts"] - in_e["ts"]).total_seconds() / 60 <= PASS_WINDOW_MINUTES
                    and o["amount"] >= PASS_MIN_RATIO * in_e["amount"]
                ),
                None,
            )
            if match:
                velocity_hits += 1
                velocity_minutes_sum += (match["ts"] - in_e["ts"]).total_seconds() / 60

        pass_through_ratio = velocity_hits / len(ins) if ins else 0.0
        avg_pass_minutes = velocity_minutes_sum / velocity_hits if velocity_hits else 0.0

        ind = in_degree[acc_id]
        outd = out_degree[acc_id]
        prox = proximity.get(acc_id, -1)
        is_known_mule = int(str(acc.get("known_mule", False)).lower() in ("true", "1"))

        rows.append({
            "account_id": acc_id,
            "label": acc.get("label", ""),
            "account_type": acc.get("account_type", "unknown"),
            # Features
            "fan_in_count": fan_in_count,
            "fan_in_amount": round(fan_in_amount, 2),
            "fan_out_count": fan_out_count,
            "fan_out_amount": round(fan_out_amount, 2),
            "pass_through_ratio": round(pass_through_ratio, 4),
            "avg_pass_minutes": round(avg_pass_minutes, 2),
            "in_degree": ind,
            "out_degree": outd,
            "degree_ratio": round(outd / max(ind, 1), 4),
            "in_out_amount_diff": round(fan_in_amount - fan_out_amount, 2),
            "proximity_to_mule": prox,
            "known_mule": is_known_mule,
            # Label
            "is_mule": acc.get("is_mule", 1 if acc.get("account_type") == "mule" else 0),
        })

    return rows


def export_features(rows: list[dict], out_path: str = "ml/artifacts/features.csv"):
    if not rows:
        print("No rows to export.")
        return

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    mule_count = sum(1 for r in rows if int(r["is_mule"]) == 1)
    print(f"Features written: {out_path}")
    print(f"  Total rows: {len(rows)}")
    print(f"  Mule (is_mule=1): {mule_count}")
    print(f"  Clean (is_mule=0): {len(rows) - mule_count}")
    print(f"  Class ratio: 1:{round((len(rows)-mule_count)/max(mule_count,1), 1)}")


if __name__ == "__main__":
    acc_path = "ml/artifacts/accounts.csv"
    txn_path = "ml/artifacts/transactions.csv"

    if not Path(acc_path).exists() or not Path(txn_path).exists():
        print(f"CSV files not found. Run python ml/data_gen.py first.")
        sys.exit(1)

    print("Loading CSVs...")
    accounts = load_csv(acc_path)
    transactions = load_csv(txn_path)
    print(f"  Accounts: {len(accounts)}, Transactions: {len(transactions)}")

    print("Extracting features...")
    rows = extract_features(accounts, transactions)

    export_features(rows)