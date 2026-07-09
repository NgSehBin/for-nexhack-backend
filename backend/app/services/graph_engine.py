"""
Graph engine — builds and queries the transaction graph.
Scoring weights match the prototype exactly:
  fanIn 0.30  |  fanOut 0.20  |  velocity 0.30  |  proximity 0.20
"""
import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timedelta

import networkx as nx

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Graph singleton ────────────────────────────────────────────────────────

_engine: "GraphEngine | None" = None


def get_engine() -> "GraphEngine":
    global _engine
    if _engine is None:
        _engine = GraphEngine()
    return _engine


# ── Engine class ───────────────────────────────────────────────────────────

class GraphEngine:
    def __init__(self):
        self.G: nx.DiGraph = nx.DiGraph()
        self._lock = asyncio.Lock()
        self._account_meta: dict[str, dict] = {}   # id → {label, known_mule, type}
        self._transactions: list[dict] = []         # raw list for graph response

    async def add_transactions(self, txns: list[dict], accounts: list[dict] | None = None):
        async with self._lock:
            # Update account metadata (label, known_mule)
            if accounts:
                for acc in accounts:
                    self._account_meta[acc["id"]] = {
                        "label": acc.get("label", acc["id"][:8]),
                        "known_mule": acc.get("known_mule", False),
                        "account_type": acc.get("account_type", "unknown"),
                    }

            for t in txns:
                if self.G.number_of_nodes() > settings.GRAPH_MAX_NODES:
                    logger.warning("Graph node cap reached — pruning old edges")
                    self._prune_old_edges()
                    if self.G.number_of_nodes() > settings.GRAPH_MAX_NODES:
                        logger.error("Still over cap after prune — skipping remaining txns")
                        break

                s, r = t["sender_id"], t["receiver_id"]
                ts = t["timestamp"] if isinstance(t["timestamp"], datetime) else datetime.fromisoformat(str(t["timestamp"]))

                # Ensure nodes exist with metadata
                for node_id in (s, r):
                    if not self.G.has_node(node_id):
                        meta = self._account_meta.get(node_id, {})
                        self.G.add_node(
                            node_id,
                            label=meta.get("label", node_id[:8]),
                            known_mule=meta.get("known_mule", False),
                            account_type=meta.get("account_type", "unknown"),
                        )

                # Merge edges
                if self.G.has_edge(s, r):
                    self.G[s][r]["amount"] += t["amount"]
                    self.G[s][r]["count"] += 1
                    self.G[s][r]["timestamps"].append(ts)
                else:
                    self.G.add_edge(s, r, amount=t["amount"], count=1, timestamps=[ts])

                self._transactions.append(t)

    def _prune_old_edges(self):
        cutoff = datetime.utcnow() - timedelta(hours=48)
        stale = [
            (u, v) for u, v, d in self.G.edges(data=True)
            if all(ts < cutoff for ts in d.get("timestamps", [datetime.utcnow()]))
        ]
        self.G.remove_edges_from(stale)
        self.G.remove_nodes_from(list(nx.isolates(self.G)))
        # Prune raw transaction list too
        self._transactions = [
            t for t in self._transactions
            if (t.get("timestamp") or datetime.utcnow()) > cutoff
        ]
        logger.info(f"Pruned {len(stale)} stale edges")

    # ── Feature extraction ────────────────────────────────────────────────

    def _window_cutoff(self, minutes: int) -> datetime:
        return datetime.utcnow() - timedelta(minutes=minutes)

    def detect_fan_in(self, node: str, window_minutes: int = 30) -> dict:
        """Count distinct senders to node within window. Matches prototype."""
        cutoff = self._window_cutoff(window_minutes)
        recent_senders = {
            u for u, _, d in self.G.in_edges(node, data=True)
            if any(ts > cutoff for ts in d.get("timestamps", []))
        }
        in_amount = sum(
            self.G[u][node]["amount"] for u in recent_senders
            if self.G.has_edge(u, node)
        )
        return {"fan_in_count": len(recent_senders), "fan_in_amount": in_amount}

    def detect_fan_out(self, node: str, window_minutes: int = 60) -> dict:
        """Count distinct receivers from node within window."""
        cutoff = self._window_cutoff(window_minutes)
        recent_receivers = {
            v for _, v, d in self.G.out_edges(node, data=True)
            if any(ts > cutoff for ts in d.get("timestamps", []))
        }
        out_amount = sum(
            self.G[node][v]["amount"] for v in recent_receivers
            if self.G.has_edge(node, v)
        )
        return {"fan_out_count": len(recent_receivers), "fan_out_amount": out_amount}

    def detect_pass_through(self, node: str, window_minutes: int = 30,
                             min_ratio: float = 0.65) -> dict:
        """
        For each in-edge, look for an out-edge within window_minutes at >= min_ratio of in-amount.
        Matches prototype: passThroughRatio = velocityHits / ins.length
        """
        cutoff = self._window_cutoff(window_minutes * 10)  # broader fetch window
        in_edges = [
            (u, d) for u, _, d in self.G.in_edges(node, data=True)
            if any(ts > cutoff for ts in d.get("timestamps", []))
        ]
        out_edges = [
            (v, d) for _, v, d in self.G.out_edges(node, data=True)
            if any(ts > cutoff for ts in d.get("timestamps", []))
        ]

        if not in_edges:
            return {"pass_through_ratio": 0.0, "avg_pass_minutes": None}

        hits = 0
        pass_minutes = []
        for _, in_d in in_edges:
            in_ts_list = sorted(in_d.get("timestamps", []))
            in_amt = in_d["amount"] / max(len(in_ts_list), 1)
            for _, out_d in out_edges:
                out_ts_list = sorted(out_d.get("timestamps", []))
                for in_ts in in_ts_list:
                    for out_ts in out_ts_list:
                        delta_min = (out_ts - in_ts).total_seconds() / 60
                        if 0 <= delta_min <= window_minutes:
                            out_amt = out_d["amount"] / max(len(out_ts_list), 1)
                            if out_amt >= min_ratio * in_amt:
                                hits += 1
                                pass_minutes.append(delta_min)

        ratio = min(hits / len(in_edges), 1.0)
        avg_min = sum(pass_minutes) / len(pass_minutes) if pass_minutes else None
        return {"pass_through_ratio": round(ratio, 4), "avg_pass_minutes": avg_min}

    def proximity_to_known_mules(self, node: str) -> int:
        """BFS hop distance to nearest confirmed mule. -1 if unreachable. Max depth 3."""
        known = {n for n, d in self.G.nodes(data=True) if d.get("known_mule")}
        if node in known:
            return 0
        if not known:
            return -1

        undirected = self.G.to_undirected()
        min_dist = float("inf")
        for mule in known:
            try:
                d = nx.shortest_path_length(undirected, source=node, target=mule)
                if d < min_dist:
                    min_dist = d
                if min_dist == 1:
                    break
            except nx.NetworkXNoPath:
                pass
        return int(min_dist) if min_dist <= 3 else -1

    def compute_node_score(self, node: str) -> dict:
        """
        Compute mule risk score 0-100. Weights match prototype exactly:
          fanIn 0.30 | fanOut 0.20 | velocity 0.30 | proximity 0.20
        """
        fi = self.detect_fan_in(node, window_minutes=settings.GRAPH_WINDOW_MINUTES)
        fo = self.detect_fan_out(node, window_minutes=settings.GRAPH_WINDOW_MINUTES)
        pt = self.detect_pass_through(node)

        max_fan_in = max((self.G.in_degree(n) for n in self.G.nodes), default=1) or 1
        max_fan_out = max((self.G.out_degree(n) for n in self.G.nodes), default=1) or 1

        fan_in_score = fi["fan_in_count"] / max_fan_in
        fan_out_score = fo["fan_out_count"] / max_fan_out

        avg_pass = pt["avg_pass_minutes"]
        if pt["pass_through_ratio"] > 0 and avg_pass is not None:
            velocity_score = pt["pass_through_ratio"] * max(0, 1 - avg_pass / 30)
        else:
            velocity_score = 0.0

        prox = self.proximity_to_known_mules(node)
        proximity_score = max(0, 1 - prox / 3) if prox >= 0 else 0.0

        raw = 100 * (
            0.30 * fan_in_score +
            0.20 * fan_out_score +
            0.30 * velocity_score +
            0.20 * proximity_score
        )
        score = int(min(100, round(raw)))

        return {
            "score": score,
            "fan_in_count": fi["fan_in_count"],
            "fan_in_amount": fi["fan_in_amount"],
            "fan_out_count": fo["fan_out_count"],
            "fan_out_amount": fo["fan_out_amount"],
            "pass_through_ratio": pt["pass_through_ratio"],
            "avg_pass_minutes": pt["avg_pass_minutes"],
            "in_degree": self.G.in_degree(node),
            "out_degree": self.G.out_degree(node),
            "proximity_to_mule": prox,
        }

    def compute_all_scores(self) -> dict[str, dict]:
        """Compute scores for every node. Called after graph rebuild."""
        scores = {}
        for node in self.G.nodes:
            feat = self.compute_node_score(node)
            self.G.nodes[node]["_score"] = feat["score"]
            scores[node] = feat
        return scores

    # ── Cluster detection ─────────────────────────────────────────────────

    def detect_clusters(self, seed_threshold: int = 28) -> list[dict]:
        """
        Louvain community detection, seeded by high-scoring nodes.
        Returns clusters with clusterScore matching prototype formula:
          clusterScore = avg*0.5 + max*0.5 + min(8, memberCount)
        """
        import community as louvain

        if self.G.number_of_nodes() < 3:
            return []

        node_scores = {n: self.G.nodes[n].get("_score", 0) for n in self.G.nodes}
        seeds = {n for n, s in node_scores.items() if s >= seed_threshold}

        undirected = self.G.to_undirected()
        try:
            partition = louvain.best_partition(undirected)
        except Exception as e:
            logger.warning(f"Louvain failed: {e}")
            return []

        # Group into communities
        communities: dict[int, list] = defaultdict(list)
        for node, cid in partition.items():
            communities[cid].append(node)

        clusters = []
        for cid, members in communities.items():
            # Only form a cluster if ≥ 1 seed present and size ≥ 3
            if len(members) < 3:
                continue
            if not any(m in seeds for m in members):
                continue

            scores = [node_scores.get(m, 0) for m in members]
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            # Prototype formula
            cluster_score = int(min(100, avg_score * 0.5 + max_score * 0.5 + min(8, len(members))))

            # Total flow = sum of intra-cluster edges
            total_flow = sum(
                d["amount"]
                for u, v, d in self.G.edges(data=True)
                if u in members and v in members
            )

            known_mule_count = sum(
                1 for m in members if self.G.nodes[m].get("known_mule", False)
            )

            clusters.append({
                "account_ids": members,
                "risk_score": cluster_score,
                "total_flow": round(total_flow, 2),
                "known_mule_count": known_mule_count,
                "pattern_flags": self._detect_patterns(members),
                "node_scores": {m: node_scores.get(m, 0) for m in members},
            })

        clusters.sort(key=lambda c: c["risk_score"], reverse=True)
        return clusters

    def _detect_patterns(self, members: list[str]) -> dict:
        """Detect which patterns are present in this cluster."""
        fan_in = any(
            self.G.in_degree(m) >= 3 for m in members
        )
        fan_out = any(
            self.G.out_degree(m) >= 3 for m in members
        )
        pass_through = any(
            self.G.nodes[m].get("_score", 0) >= 40 and
            self.G.in_degree(m) > 0 and self.G.out_degree(m) > 0
            for m in members
        )
        proximity = any(
            self.G.nodes[m].get("known_mule", False) for m in members
        )
        return {
            "fan_in": fan_in,
            "fan_out": fan_out,
            "pass_through": pass_through,
            "proximity_to_known_mule": proximity,
        }

    def get_graph_data(self, window_minutes: int | None = None) -> dict:
        """Returns nodes and edges for the frontend graph canvas."""
        wm = window_minutes or settings.GRAPH_WINDOW_MINUTES
        cutoff = self._window_cutoff(wm)

        nodes = []
        for node_id, data in self.G.nodes(data=True):
            score = data.get("_score", 0)
            tier = "high" if score >= settings.SCORE_HIGH else \
                   "elevated" if score >= settings.SCORE_ELEVATED else "clean"
            nodes.append({
                "id": node_id,
                "label": data.get("label", node_id[:8]),
                "score": score,
                "tier": tier,
                "known_mule": data.get("known_mule", False),
            })

        edges = []
        for u, v, d in self.G.edges(data=True):
            if any(ts > cutoff for ts in d.get("timestamps", [])):
                edges.append({"from": u, "to": v, "amount": round(d["amount"], 2)})

        return {"nodes": nodes, "edges": edges}
