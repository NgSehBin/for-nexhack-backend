"""
ml/data_gen.py — Synthetic mule network data generator.

Produces 3 clusters that EXACTLY match the prototype's hardcoded data:
  Cluster 1: Collector 7711, 9 victim sources, 3 mule dispersal accounts (Mule 2290 known)
  Cluster 2: Victim Account A, 4-hop chain (Mule A4 known)
  Cluster 3: Collector 4402, 6 victim sources, 2 mule dispersal accounts
  Clean:     8 benign customers + 6 random pairs

Run directly to generate CSV files and optionally seed the API:
  python ml/data_gen.py             # generates CSVs only
  python ml/data_gen.py --seed-api  # POSTs to http://localhost:8000

CSVs written to:
  ml/artifacts/accounts.csv
  ml/artifacts/transactions.csv
"""
import argparse
import csv
import json
import random
import uuid
from datetime import datetime, timedelta

NOW = datetime.utcnow()
MIN = timedelta(minutes=1)


def ri(a: int, b: int) -> int:
    return random.randint(a, b)


def rf(a: float, b: float) -> float:
    return round(random.uniform(a, b), 2)


def new_id() -> str:
    return str(uuid.uuid4())


def ts_ago(minutes: int) -> datetime:
    return NOW - timedelta(minutes=minutes)


# ── Build synthetic dataset ────────────────────────────────────────────────

def generate_dataset(seed: int = 42) -> tuple[list[dict], list[dict]]:
    random.seed(seed)
    accounts: list[dict] = []
    transactions: list[dict] = []

    def account(label: str, acc_type: str, known_mule: bool = False) -> dict:
        a = {
            "id": new_id(),
            "label": label,
            "account_type": acc_type,
            "known_mule": known_mule,
            "is_mule": 1 if acc_type == "mule" else 0,
        }
        accounts.append(a)
        return a

    def txn(from_acc: dict, to_acc: dict, amount: float, minutes_ago: int) -> dict:
        t = {
            "id": new_id(),
            "sender_id": from_acc["id"],
            "receiver_id": to_acc["id"],
            "amount": amount,
            "currency": "MYR",
            "timestamp": ts_ago(minutes_ago).isoformat(),
            "channel": random.choice(["DuitNow", "IBG", "FPX"]),
            "reference": None,
        }
        transactions.append(t)
        return t

    # ── Cluster 1: Fan-in collector + 3 dispersal mules ─────────────────
    # Matches prototype: Collector 7711 ← 9 victims, then → Mule 2290/5510/8123
    collector1 = account("Collector 7711", "mule")
    sources1 = []
    for i in range(9):
        s = account(f"Victim-src {i+1}", "source")
        sources1.append(s)
        txn(s, collector1, rf(800, 4500), ri(140, 200))

    mule2290 = account("Mule 2290", "mule", known_mule=True)
    mule5510 = account("Mule 5510", "mule")
    mule8123 = account("Mule 8123", "mule")
    for m in [mule2290, mule5510, mule8123]:
        txn(collector1, m, rf(2000, 9000), ri(110, 135))

    # ── Cluster 2: 4-hop chain (pass-through) ───────────────────────────
    # Matches prototype: Victim Account A → Mule A1 → A2 → A3 → A4 (known)
    victim2 = account("Victim Account A", "victim")
    chain = [
        account("Mule A1", "mule"),
        account("Mule A2", "mule"),
        account("Mule A3", "mule"),
        account("Mule A4", "mule", known_mule=True),
    ]
    last_t = ri(80, 95)
    amt = 18500.0
    txn(victim2, chain[0], amt, last_t)
    for i in range(len(chain) - 1):
        last_t -= ri(12, 19)
        amt = round(amt * 0.94, 2)
        txn(chain[i], chain[i + 1], amt, max(last_t, 2))

    # ── Cluster 3: Fan-in collector + 2 dispersal mules ─────────────────
    # Matches prototype: Collector 4402 ← 6 victims → Mule 9921/3387
    collector3 = account("Collector 4402", "mule")
    sources3 = []
    for i in range(6):
        s = account(f"Victim-src {i+10}", "source")
        sources3.append(s)
        txn(s, collector3, rf(600, 3200), ri(200, 260))

    mule9921 = account("Mule 9921", "mule")
    mule3387 = account("Mule 3387", "mule")
    for m in [mule9921, mule3387]:
        txn(collector3, m, rf(3000, 7000), ri(175, 195))

    # ── Clean traffic ────────────────────────────────────────────────────
    # 8 benign customers + 14 random P2P transactions
    benign = [account(f"Customer {i+1}", "benign") for i in range(8)]
    for _ in range(14):
        a, b = random.sample(benign, 2)
        txn(a, b, rf(50, 1800), ri(300, 4000))

    # 6 isolated benign pairs (not connected to any mule cluster)
    for i in range(6):
        a = account(f"Account {i+1000}", "benign")
        b = account(f"Account {i+2000}", "benign")
        txn(a, b, rf(100, 900), ri(500, 6000))

    return accounts, transactions


# ── CSV export ─────────────────────────────────────────────────────────────

def export_csv(accounts: list[dict], transactions: list[dict], out_dir: str = "ml/artifacts"):
    import os
    os.makedirs(out_dir, exist_ok=True)

    acc_path = f"{out_dir}/accounts.csv"
    with open(acc_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "account_type", "known_mule", "is_mule"])
        writer.writeheader()
        writer.writerows(accounts)
    print(f"Accounts written: {acc_path} ({len(accounts)} rows)")

    txn_path = f"{out_dir}/transactions.csv"
    with open(txn_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "sender_id", "receiver_id", "amount", "currency", "timestamp", "channel", "reference"])
        writer.writeheader()
        writer.writerows(transactions)
    print(f"Transactions written: {txn_path} ({len(transactions)} rows)")


# ── API seed ───────────────────────────────────────────────────────────────

def seed_api(accounts: list[dict], transactions: list[dict], base_url: str = "http://localhost:8000"):
    """POST synthetic data to the running API for live demo."""
    import httpx

    print(f"\nSeeding API at {base_url}...")

    # Login with default admin
    r = httpx.post(f"{base_url}/api/auth/token", data={
        "username": "admin@muledetect.local",
        "password": "hackathon2026",
    })
    if r.status_code != 200:
        print(f"Login failed: {r.status_code} {r.text}")
        return
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # POST accounts in batches of 100
    acc_payload = [
        {
            "id": a["id"],
            "label": a["label"],
            "account_type": a["account_type"] if a["account_type"] in ("mule", "source", "victim", "benign") else "unknown",
            "known_mule": a["known_mule"],
        }
        for a in accounts
    ]
    for i in range(0, len(acc_payload), 100):
        batch = acc_payload[i:i+100]
        r = httpx.post(f"{base_url}/api/accounts", json={"accounts": batch}, headers=headers, timeout=30)
        print(f"  Accounts batch {i//100 + 1}: {r.status_code}")

    # POST transactions in batches of 100
    txn_payload = [
        {
            "id": t["id"],
            "sender_id": t["sender_id"],
            "receiver_id": t["receiver_id"],
            "amount": t["amount"],
            "currency": t["currency"],
            "timestamp": t["timestamp"],
            "channel": t["channel"],
            "reference": t["reference"],
        }
        for t in transactions
    ]
    for i in range(0, len(txn_payload), 100):
        batch = txn_payload[i:i+100]
        r = httpx.post(f"{base_url}/api/transactions", json={"transactions": batch}, headers=headers, timeout=30)
        print(f"  Transactions batch {i//100 + 1}: {r.status_code}")

    print("\nSeed complete. Wait ~5 seconds for graph rebuild, then visit /api/clusters")


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic mule detection data")
    parser.add_argument("--seed-api", action="store_true", help="POST data to running API")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    print("Generating synthetic dataset...")
    accounts, transactions = generate_dataset(seed=args.seed)
    print(f"Generated: {len(accounts)} accounts, {len(transactions)} transactions")

    mule_count = sum(1 for a in accounts if a["account_type"] == "mule")
    print(f"  Mule accounts: {mule_count}")
    print(f"  Clean accounts: {len(accounts) - mule_count}")

    export_csv(accounts, transactions)

    if args.seed_api:
        seed_api(accounts, transactions, base_url=args.api_url)