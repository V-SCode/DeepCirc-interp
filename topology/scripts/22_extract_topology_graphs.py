"""Phase 22 — Extract slim topology-graph manifest for local rendering.

Produces a small JSON file with per-topology graph structure (nodes +
edges + types) for the 215 trained topologies, plus the 44 markers
(separately flagged). The full population.pkl lives on cluster scratch
and isn't versioned; this slim manifest IS committed so local figures
can render topology DAGs without needing population.pkl.

Output schema:

    {
      "scope": {"include_baselines": false, "size_classes": [4,5,6,7]},
      "group": "G3",
      "n_trained": 215,
      "n_markers": 44,
      "topologies": [
        {
          "topology_id": "deab6c1f",
          "regulator_count": 4,
          "source": ["0x2B", "0x4D", "0x71"],
          "primary_target": "0x2B",
          "qc_tier": "A1",
          "is_marker": false,
          "nodes": [
            {"id": 0, "type": "IN"},
            {"id": 1, "type": "IN"},
            {"id": 2, "type": "IN"},
            {"id": 3, "type": "NOR"},
            ...
          ],
          "edges": [[0, 3], [1, 3], ...]
        },
        ...
      ]
    }

Node `type` is one of {IN, NOT, NOR, OUT}, derived from the NetworkX
graph's node attributes (P3 emits this during population assembly).

Compute: seconds. Reads population.pkl + qc_tiers.pkl, walks the graphs.

Usage:

    python topology/scripts/22_extract_topology_graphs.py \\
        --population_manifest $DEEPCIRC_SCRATCH/population/G3/population.pkl \\
        --qc_tiers            $DEEPCIRC_SCRATCH/population/G3/qc_tiers.pkl \\
        --output              data/topology_g3/topology_graphs.json
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import networkx as nx


def node_type(graph: nx.DiGraph, node) -> str:
    """Classify a node by its in/out degree (mirrors upstream NIG typing).

    in=0       -> IN  (sensor input)
    out=0      -> OUT (reporter output)
    in=1, out>0 -> NOT
    in=2, out>0 -> NOR
    else        -> ?  (shouldn't happen in well-formed NIGs)
    """
    in_deg = graph.in_degree(node)
    out_deg = graph.out_degree(node)
    if in_deg == 0:
        return "IN"
    if out_deg == 0:
        return "OUT"
    if in_deg == 1 and out_deg > 0:
        return "NOT"
    if in_deg == 2 and out_deg > 0:
        return "NOR"
    return "?"


def graph_to_record(t: dict, *, is_marker: bool, qc_tier: str) -> dict:
    """Build the slim per-topology record."""
    G = t.get("graph")
    if G is None:
        return None
    nodes_sorted = sorted(G.nodes())
    # Re-index from 0 for the manifest (preserves DAG structure)
    node_index = {n: i for i, n in enumerate(nodes_sorted)}
    nodes = []
    for n in nodes_sorted:
        nodes.append({"id": node_index[n], "type": node_type(G, n)})
    edges = []
    for u, v in G.edges():
        edges.append([node_index[u], node_index[v]])
    raw_source = t.get("source")
    if isinstance(raw_source, list):
        primary = raw_source[0] if raw_source else "?"
    else:
        primary = raw_source or "?"
    return {
        "topology_id":     t["topology_id"],
        "regulator_count": int(t.get("regulator_count", -1)),
        "source":          raw_source,
        "primary_target":  primary,
        "qc_tier":         qc_tier,
        "is_marker":       is_marker,
        "nodes":           nodes,
        "edges":           edges,
        "num_nodes":       len(nodes),
        "num_edges":       len(edges),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--population_manifest", type=Path, required=True)
    ap.add_argument("--qc_tiers", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    print(f"Phase 22 — extract topology graphs")
    print(f"  population: {args.population_manifest}")
    print(f"  qc_tiers:   {args.qc_tiers}")
    print(f"  output:     {args.output}")
    print()

    with open(args.population_manifest, "rb") as f:
        pop = pickle.load(f)
    with open(args.qc_tiers, "rb") as f:
        qc = pickle.load(f)
    qc_by_tid = {r["topology_id"]: r for r in qc["records"]}

    records = []
    n_trained = n_markers = n_skipped = 0
    for t in pop["topologies"]:
        if t.get("is_baseline", False):
            continue
        if t.get("regulator_count") not in (4, 5, 6, 7):
            continue
        tid = t["topology_id"]
        qc_rec = qc_by_tid.get(tid, {})
        tier = qc_rec.get("tier", "?")
        is_marker = tier in ("A0", "BROKEN")
        rec = graph_to_record(t, is_marker=is_marker, qc_tier=tier)
        if rec is None:
            n_skipped += 1
            continue
        records.append(rec)
        if is_marker:
            n_markers += 1
        else:
            n_trained += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "scope": {"include_baselines": False, "size_classes": [4, 5, 6, 7]},
        "group": pop.get("group", "G?"),
        "n_trained": n_trained,
        "n_markers": n_markers,
        "n_skipped": n_skipped,
        "topologies": records,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    sz_kb = args.output.stat().st_size / 1024
    print(f"[emit] {args.output} ({sz_kb:.1f} KB)")
    print(f"  n_trained: {n_trained}")
    print(f"  n_markers: {n_markers}")
    print(f"  n_skipped: {n_skipped}")
    print(f"\nPhase 22 done.")


if __name__ == "__main__":
    main()
