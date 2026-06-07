"""Phase 3 — assemble the topology population manifest for a group.

**v2.0 protocol (locked 2026-05-06): per-(target × size class) up-to-5 sampling.**

For each target in the group's cumulative target set, this script:
  1. Loads the target's raw PPO Stage-1 registry (paper-scale per
     `pipeline.yaml`).
  2. Partitions the target's WL-distinct topologies by regulator count
     into the four size classes {4, 5, 6, 7}-reg.
  3. From each (target × size) cell, selects up to N topologies
     (default N = 5 per the v2.0 protocol). Parser-picks (P1.5 tagged)
     are preferred when capping; remaining slots filled in deterministic
     WL-hash order.
  4. Records structural floors per target: cells with 0 candidates
     are documented (e.g., 0x6D's 4/5/6-reg are structurally
     impossible because the function's minimum is 7-reg).
  5. WL-deduplicates the union of selections across all (target × size)
     cells. A topology that appears in multiple targets' top-N gets one
     entry with a multi-source label.

**Why this differs from the DeepCirc paper's selection** (Methods lines
447-452): the paper takes **minimum-regulator-only** topologies after
pruning + interference filters — appropriate for picking a single
yellow-dot per target. Our cross-topology study deliberately expands
this to "up to 5 per (target × size class) at sizes 4-7-reg" so that
size-dependent and shape-dependent design rules can be extracted across
a meaningful population. The methodological choice is documented in
PROJECT_STATE_TOPOLOGY.md §2 decision #7 and §4.3.

**Baselines (P2) are deferred under v2.0** — the `--baselines_dir` flag
is still accepted but the active pipeline does not pass it. See
PROJECT_STATE_TOPOLOGY.md §1.4 for the baselines-as-post-paper-add-on
rationale.

Pipeline:

    PPO registry (P1, paper-scale)   upstream parser (P1.5, optional, tag only)
        │                                  │
        └────────────────┬─────────────────┘
                         ▼
    P3 partitions per target by energy, takes up to N per (target ×
    size) cell, WL-dedupes the union, computes graph-role fingerprints,
    writes population.pkl. P2 baselines optionally appended only if
    `--baselines_dir` is passed (deferred under v2.0).

Output schema (pickled dict):
    {
      "group": "G1",
      "per_target_per_size": 5,
      "size_classes": [4, 5, 6, 7],
      "n_total": <int>, "n_agent": <int>, "n_baseline": <int>,
      "structural_coverage": {              # NEW v2.0
        "0xEE": {4: 5, 5: 5, 6: 5, 7: 5},  # achieved per-cell counts
        "0x17": {4: 0, 5: 5, 6: 3, 7: 5},  # 4-reg structurally absent
        "0x2B": {4: 2, 5: 1, 6: 5, 7: 5},  # ...
        "0x6D": {4: 0, 5: 0, 6: 0, 7: 4},  # 7-reg only
      },
      "topologies": [
        {
          "topology_id": "<wl-8>",
          "wl_hash": "<wl-full>",
          "source": "0x2B" | ["0x2B", "0xEE"] | "baseline:<parent>",
          "selected_for": [("0x2B", 5), ("0xEE", 4)],   # NEW v2.0: which (target, size) cells this entry fills
          "is_baseline": False,
          "is_parser_pick": True,
          "energy": 5, "regulator_count": 5,
          "num_nodes": 9, "num_edges": 11,
          "depth": 3,
          "graph": <networkx.DiGraph>,
          "fingerprints": [...],
          "nngga_pkl": "<resolved_nngga_root>/<topology_id>/optimal_topologies.pkl",
          "nngga_circuit_name": "<topology_id>",
        },
        ...
      ],
    }

`nngga_pkl` resolves under the group-agnostic per-topology pool (Option C, R5):
`$DEEPCIRC_SCRATCH/topology_data/nngga/<topology_id>/optimal_topologies.pkl`.
Idempotent: skips topology-pkl write if file already exists, so G2's P3
reuses G1's already-generated pkls. Override with `--nngga_root` to write
elsewhere (e.g., scratch test pool).

Usage:
    python 03_assemble_population.py \\
        --group G2 \\
        --per_target_per_size 5 \\
        --output_dir $DEEPCIRC_SCRATCH/population/G2 \\
        --topology_runs_dir $DEEPCIRC_SCRATCH/runs/stage1
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Iterator

import networkx as nx
from networkx.algorithms import weisfeiler_lehman_graph_hash


GROUP_TARGETS = {
    "G1": ["0x2B", "0x17", "0x6D", "0xEE"],
    "G2": ["0x2B", "0x17", "0x6D", "0xEE",
           "0x4D", "0xCC", "0xE8", "0x33", "0x66", "0x3C"],
    "G3": ["0x2B", "0x17", "0x6D", "0xEE",
           "0x4D", "0xCC", "0xE8", "0x33", "0x66", "0x3C",
           "0x2C", "0x03", "0x0F", "0x06", "0x07", "0x71",
           "0x8E", "0x77", "0x47", "0x1B"],
}

SIZE_CLASSES = (4, 5, 6, 7)


# -----------------------------------------------------------------------------
# Graph helpers
# -----------------------------------------------------------------------------
def _to_graph(x) -> nx.DiGraph:
    if isinstance(x, (nx.Graph, nx.DiGraph)):
        return x
    if isinstance(x, dict):
        return nx.node_link_graph(x)
    raise TypeError(f"unexpected type: {type(x)}")


def wl_hash(G: nx.DiGraph) -> str:
    return weisfeiler_lehman_graph_hash(G, node_attr=None, edge_attr=None,
                                        iterations=30, digest_size=16)


def compute_energy(G: nx.DiGraph) -> int:
    n_inputs = sum(1 for n in G if G.in_degree(n) == 0)
    n_outputs = sum(1 for n in G if G.out_degree(n) == 0)
    return G.number_of_nodes() - n_inputs - n_outputs


def compute_depth(G: nx.DiGraph) -> int:
    if G.number_of_nodes() == 0:
        return 0
    inputs = [n for n in G if G.in_degree(n) == 0]
    if not inputs:
        return 0
    dist: dict = {n: 0 for n in inputs}
    for n in nx.topological_sort(G):
        for s in G.successors(n):
            dist[s] = max(dist.get(s, 0), dist.get(n, 0) + 1)
    return max(dist.values()) if dist else 0


def compute_fingerprints(G: nx.DiGraph) -> list[dict]:
    """Per PROJECT_STATE_TOPOLOGY.md §7.1."""
    inputs = {n for n in G if G.in_degree(n) == 0}
    outputs = {n for n in G if G.out_degree(n) == 0}
    internals = [n for n in G.nodes() if n not in inputs and n not in outputs]

    depth_from_input: dict = {n: 0 for n in inputs}
    for n in nx.topological_sort(G):
        for s in G.successors(n):
            depth_from_input[s] = max(
                depth_from_input.get(s, 0), depth_from_input.get(n, 0) + 1
            )
    depth_to_output: dict = {n: 0 for n in outputs}
    for n in reversed(list(nx.topological_sort(G))):
        for p in G.predecessors(n):
            depth_to_output[p] = max(
                depth_to_output.get(p, 0), depth_to_output.get(n, 0) + 1
            )

    fingerprints: list[dict] = []
    for n in internals:
        in_deg = G.in_degree(n)
        out_deg = G.out_degree(n)
        node_type = "NOR" if in_deg == 2 else ("NOT" if in_deg == 1 else f"unusual({in_deg})")
        pred_depths = sorted([depth_from_input.get(p, 0) for p in G.predecessors(n)])
        # Successor signature mirrors predecessor: for each immediate child,
        # report depth_to_output (how close to output the child is). Combined
        # with predecessor_depth_signature this gives full position context —
        # "my parents are X depth from input, my children are Y depth from
        # output" — and lets L2 cells split finer than predecessor-only.
        # See PROJECT_STATE_TOPOLOGY.md §7.1; v1.2 caveat resolved (Path B.1).
        succ_depths = sorted([depth_to_output.get(s, 0) for s in G.successors(n)])
        fingerprints.append({
            "node": n,
            "depth_to_output": depth_to_output.get(n, 0),
            "depth_from_input": depth_from_input.get(n, 0),
            "in_degree": in_deg,
            "out_degree": out_deg,
            "predecessor_depth_signature": ",".join(str(d) for d in pred_depths),
            "successor_depth_signature":   ",".join(str(d) for d in succ_depths),
            "node_type": node_type,
        })
    fingerprints.sort(key=lambda fp: (fp["depth_to_output"], fp["node"]))
    return fingerprints


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------
def load_raw_registry(reg_pkl: Path) -> Iterator[tuple[nx.DiGraph, float]]:
    """Yield (canonical_graph, energy) from a raw P1 registry pickle.

    Registry format (from upstream's `save_registry_pickle`):
        dict[wl_hash, list[(canonical_node_link, original_node_link, energy)]]
    """
    with open(reg_pkl, "rb") as f:
        reg = pickle.load(f)
    for _wl, bucket in reg.items():
        for canon_data, _orig, energy in bucket:
            yield _to_graph(canon_data), float(energy)


def load_parser_pick_hashes(parsed_pkl: Path) -> set[str]:
    """Return the set of WL hashes the upstream parser identified as Pareto picks.

    The parser writes a list[networkx.DiGraph] to optimal_topologies.pkl. We
    re-hash here so that "is_parser_pick" can be looked up by hash.

    If parsed_pkl doesn't exist (P1.5 not run), returns an empty set — the
    population is still built from the raw registry, just without parser
    tagging.
    """
    if not parsed_pkl.exists():
        return set()
    with open(parsed_pkl, "rb") as f:
        graphs = pickle.load(f)
    if not isinstance(graphs, list):
        return set()
    return {wl_hash(_to_graph(G)) for G in graphs}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group", choices=["G1", "G2", "G3"], required=True)
    p.add_argument("--per_target_per_size", type=int, default=5,
                   help="v2.0 protocol: up to N WL-distinct topologies per "
                        "(target × size class) cell. Default 5.")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--topology_runs_dir", type=Path, required=True,
                   help="Stage-1 output root; expects "
                        "<HEX>/trained_masked/trained_final_shared_registry.pkl "
                        "and (optionally) <HEX>/trained_masked/optimal_topologies/optimal_topologies.pkl")
    p.add_argument("--baselines_dir", type=Path,
                   help="P2 output dir containing baselines.pkl + nngga_format/. "
                        "DEFERRED under v2.0; pass only to revisit baselines as a "
                        "post-paper add-on (see PROJECT_STATE_TOPOLOGY.md §1.4).")
    p.add_argument("--nngga_root", type=Path, default=None,
                   help="Where to write per-topology NNGGA input pkls. "
                        "Default: $DEEPCIRC_SCRATCH/topology_data/nngga/ "
                        "(group-agnostic per-topology pool, Option C / R5). "
                        "Override only for test pools. Idempotent: skips "
                        "topology-pkl write if the file already exists "
                        "(reuses earlier groups' pkls).")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_hexes = GROUP_TARGETS[args.group]

    print(f"Phase 3 — population assembly for {args.group}")
    print(f"  targets              : {target_hexes}")
    print(f"  per_target_per_size  : {args.per_target_per_size}  (v2.0)")
    print(f"  size_classes         : {list(SIZE_CLASSES)}")
    print()

    # ------------------------------------------------------------------------
    # Step 1: per-target → per-(target × size) up-to-N selection
    # ------------------------------------------------------------------------
    # Each target's PPO registry is partitioned by regulator count into the
    # SIZE_CLASSES buckets. From each (target × size) cell we take up to
    # `per_target_per_size` WL-distinct topologies, parser-picks first.
    # Cells with 0 candidates are recorded as structural floors (e.g., 0x6D's
    # 4/5/6-reg cells are structurally impossible; only 7-reg exists).
    print(f"[1] per-(target × size) selection (up to {args.per_target_per_size} "
          f"per cell)")

    # Per-WL-hash union of selections (for cross-target dedup at end)
    by_hash: dict[str, dict] = {}
    # Achieved count per (target, size) cell — used for structural-coverage report
    structural_coverage: dict[str, dict[int, int]] = {
        hex_str: {sz: 0 for sz in SIZE_CLASSES} for hex_str in target_hexes
    }

    for hex_str in target_hexes:
        raw_pkl = (args.topology_runs_dir / hex_str / "trained_masked"
                   / "trained_final_shared_registry.pkl")
        if not raw_pkl.exists():
            print(f"  ⚠️  {hex_str}: raw registry not found at {raw_pkl}; skipping")
            continue

        # Optional parser pkl for tagging (P1.5 output)
        parsed_pkl = (args.topology_runs_dir / hex_str / "trained_masked"
                      / "optimal_topologies" / "optimal_topologies.pkl")
        parser_picks = load_parser_pick_hashes(parsed_pkl)
        parser_status = (f"{len(parser_picks)} parser picks"
                         if parser_picks else "no P1.5 parser pkl")

        # Bucket this target's WL-distinct candidates by size
        target_buckets: dict[int, list[dict]] = {sz: [] for sz in SIZE_CLASSES}
        seen_hashes: set[str] = set()
        n_loaded = 0
        for G, energy in load_raw_registry(raw_pkl):
            n_loaded += 1
            energy_int = int(round(energy))
            if energy_int not in target_buckets:
                continue  # outside SIZE_CLASSES (e.g., 8+ reg) — skip
            h = wl_hash(G)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            target_buckets[energy_int].append({
                "topology_id":    h[:8],
                "wl_hash":        h,
                "graph":          G,
                "energy":         energy_int,
                "is_parser_pick": h in parser_picks,
            })

        # For each size cell, take up to N parser-picks-first then by WL hash
        n_target_cells_full = 0  # cells that hit the per_target_per_size cap
        n_target_kept = 0
        cell_summary: list[str] = []
        for sz in SIZE_CLASSES:
            pool = sorted(
                target_buckets[sz],
                key=lambda e: (not e["is_parser_pick"], e["wl_hash"]),
            )
            take = pool[: args.per_target_per_size]
            structural_coverage[hex_str][sz] = len(take)
            n_target_kept += len(take)
            if len(take) == args.per_target_per_size:
                n_target_cells_full += 1
            for entry in take:
                h = entry["wl_hash"]
                if h not in by_hash:
                    by_hash[h] = {
                        "topology_id":    entry["topology_id"],
                        "wl_hash":        h,
                        "source":         hex_str,
                        "selected_for":   [(hex_str, sz)],
                        "is_baseline":    False,
                        "is_parser_pick": entry["is_parser_pick"],
                        "energy":         entry["energy"],
                        "graph":          entry["graph"],
                    }
                else:
                    # WL-collision across targets: same topology appears in
                    # multiple targets' top-N. Merge sources + selected_for.
                    existing = by_hash[h]
                    if isinstance(existing["source"], list):
                        if hex_str not in existing["source"]:
                            existing["source"].append(hex_str)
                    else:
                        if existing["source"] != hex_str:
                            existing["source"] = [existing["source"], hex_str]
                    existing["selected_for"].append((hex_str, sz))
                    if entry["is_parser_pick"]:
                        existing["is_parser_pick"] = True
            cell_summary.append(
                f"{sz}-reg={len(take)}/{len(target_buckets[sz])}"
            )
        print(f"  {hex_str}: {n_loaded} raw → " + ", ".join(cell_summary)
              + f"  ({parser_status})")

    selected: list[dict] = list(by_hash.values())
    n_agent = len(selected)

    # ------------------------------------------------------------------------
    # Structural-coverage report: cells where the agent produced 0 candidates
    # are flagged. These are either (a) structurally impossible (e.g., 0x6D
    # cannot be implemented at 4/5/6-reg) or (b) under-explored at current
    # paper-scale episode count. The distinction is biological vs. compute.
    # ------------------------------------------------------------------------
    print(f"\n[1b] Structural coverage table (achieved per-cell counts):")
    print(f"  {'target':>8s} | " + " ".join(f"{sz}-reg" for sz in SIZE_CLASSES)
          + " | total")
    print(f"  {'-' * 8} | " + " ".join("-" * 5 for _ in SIZE_CLASSES)
          + " | -----")
    for hex_str in target_hexes:
        row = structural_coverage[hex_str]
        total = sum(row.values())
        cells = " ".join(f"{row[sz]:>5d}" for sz in SIZE_CLASSES)
        empty = [sz for sz in SIZE_CLASSES if row[sz] == 0]
        flag = f"  ⚠️  empty: {empty}" if empty else ""
        print(f"  {hex_str:>8s} | {cells} | {total:>5d}{flag}")

    n_parser_picks_total = sum(1 for c in selected if c.get("is_parser_pick"))
    print(f"\n  agent topologies (post-WL-dedup union): {n_agent}")
    print(f"  parser-picks among selected            : {n_parser_picks_total}")
    print(f"  cross-target collisions (multi-source) : "
          f"{sum(1 for c in selected if isinstance(c['source'], list))}")

    if n_agent == 0:
        raise SystemExit(
            "ERROR: no agent topologies selected. Either:\n"
            "  (a) Phase 1 has not been run for any of this group's targets, OR\n"
            "  (b) Stage-1 registry pkls are at a different path — check --topology_runs_dir.\n"
            f"  Expected raw registry: <topology_runs_dir>/<HEX>/trained_masked/trained_final_shared_registry.pkl"
        )

    # ------------------------------------------------------------------------
    # Step 3: append baselines (DEFERRED under v2.0 — opt-in only)
    # ------------------------------------------------------------------------
    # Hamming-1 baselines are no longer part of the active pipeline (state
    # doc §1.4 long-term / post-paper add-on). The script still accepts
    # `--baselines_dir` for backward compatibility and for revisiting the
    # comparison after the main paper ships.
    n_baseline = 0
    if args.baselines_dir:
        bp = args.baselines_dir / "baselines.pkl"
        if bp.exists():
            print(f"\n[3] (DEFERRED in v2.0) adding baselines from {bp}")
            with open(bp, "rb") as f:
                baseline_bundle = pickle.load(f)
            for b in baseline_bundle["baselines"]:
                G = b["graph"]
                h = wl_hash(G)
                if any(s["wl_hash"] == h for s in selected):
                    continue
                energy = compute_energy(G)
                selected.append({
                    "topology_id": h[:8],
                    "wl_hash": h,
                    "source": f"baseline:{b['parent_target']}/{b['parent_hash'][:8]}",
                    "selected_for": [],  # baselines aren't in any (target, size) cell
                    "is_baseline": True,
                    "is_parser_pick": False,
                    "energy": energy,
                    "graph": G,
                    "parent_size": b["parent_size"],
                    "nngga_pkl_baseline": b.get("nngga_pkl"),
                })
                n_baseline += 1
            print(f"  added {n_baseline} baselines (some may have been deduped against agent set)")
        else:
            print(f"\n[3] (DEFERRED in v2.0) no baselines.pkl at {bp} — skipping")
    else:
        print(f"\n[3] (DEFERRED in v2.0) --baselines_dir not passed — agent-only manifest")

    # ------------------------------------------------------------------------
    # Step 4: per-entry NNGGA pkl + fingerprints + topology features
    # ------------------------------------------------------------------------
    # nngga_root default (post-R5, Option C): group-agnostic per-topology pool
    # at $DEEPCIRC_SCRATCH/topology_data/nngga/. G2/G3 P3 fires reuse G1's
    # pkls via the per-entry exists-skip below. Override --nngga_root only
    # for test pools.
    if args.nngga_root:
        nngga_root = args.nngga_root
    else:
        scratch = os.environ.get("DEEPCIRC_SCRATCH")
        if not scratch:
            raise SystemExit(
                "ERROR: --nngga_root not provided and $DEEPCIRC_SCRATCH is unset.\n"
                "       Either activate the conda env (which sets DEEPCIRC_SCRATCH),\n"
                "       or pass --nngga_root explicitly (e.g., a test pool)."
            )
        nngga_root = Path(scratch) / "topology_data" / "nngga"
    nngga_root.mkdir(parents=True, exist_ok=True)
    print(f"\n[4] writing per-entry NNGGA pkls + fingerprints for "
          f"{len(selected)} topologies → {nngga_root}/")

    n_pkl_written = 0
    n_pkl_reused  = 0
    for entry in selected:
        G = entry["graph"]
        entry["regulator_count"] = entry["energy"]
        entry["num_nodes"] = G.number_of_nodes()
        entry["num_edges"] = G.number_of_edges()
        entry["depth"] = compute_depth(G)
        entry["fingerprints"] = compute_fingerprints(G)
        entry["group"] = args.group

        per_dir = nngga_root / entry["topology_id"]
        per_dir.mkdir(parents=True, exist_ok=True)
        per_pkl = per_dir / "optimal_topologies.pkl"

        # Idempotent: skip pkl write if it already exists (reuses earlier
        # groups' pkls when nngga_root is the group-agnostic pool, or
        # rewrites are skipped within a single group's re-run).
        if per_pkl.exists() and per_pkl.stat().st_size > 0:
            n_pkl_reused += 1
        else:
            with open(per_pkl, "wb") as f:
                pickle.dump([G], f)
            n_pkl_written += 1
        entry["nngga_pkl"] = str(per_pkl)
        entry["nngga_circuit_name"] = entry["topology_id"]

    print(f"    pkls written: {n_pkl_written}; reused (already existed): "
          f"{n_pkl_reused}")

    # ------------------------------------------------------------------------
    # Step 5: save manifest + JSON summary
    # ------------------------------------------------------------------------
    print(f"\n[5] writing population manifest")
    manifest = {
        "group": args.group,
        "protocol_version": "v2.0",
        "per_target_per_size": args.per_target_per_size,
        "size_classes": list(SIZE_CLASSES),
        "structural_coverage": structural_coverage,
        "n_total": len(selected),
        "n_agent": n_agent,
        "n_baseline": n_baseline,
        "n_parser_picks": sum(1 for e in selected if e.get("is_parser_pick")),
        "topologies": selected,
    }
    out_pkl = args.output_dir / "population.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(manifest, f)

    summary = {
        "group": args.group,
        "protocol_version": "v2.0",
        "per_target_per_size": args.per_target_per_size,
        "size_classes": list(SIZE_CLASSES),
        "structural_coverage": structural_coverage,
        "n_total": len(selected),
        "n_agent": n_agent,
        "n_baseline": n_baseline,
        "n_parser_picks": sum(1 for e in selected if e.get("is_parser_pick")),
        "by_size": {
            sz: sum(1 for e in selected if e["energy"] == sz) for sz in SIZE_CLASSES
        },
        "by_size_agent_only": {
            sz: sum(1 for e in selected if e["energy"] == sz and not e["is_baseline"])
            for sz in SIZE_CLASSES
        },
        "by_size_baseline_only": {
            sz: sum(1 for e in selected if e["energy"] == sz and e["is_baseline"])
            for sz in SIZE_CLASSES
        },
        "by_size_parser_picks": {
            sz: sum(1 for e in selected if e["energy"] == sz and e.get("is_parser_pick"))
            for sz in SIZE_CLASSES
        },
        "topologies": [
            {
                "topology_id":     e["topology_id"],
                "source":          e["source"],
                "selected_for":    e.get("selected_for", []),
                "energy":          e["energy"],
                "num_nodes":       e["num_nodes"],
                "depth":           e["depth"],
                "is_baseline":     e["is_baseline"],
                "is_parser_pick":  e.get("is_parser_pick", False),
                "nngga_pkl":       e["nngga_pkl"],
                "nngga_circuit_name": e["nngga_circuit_name"],
            }
            for e in selected
        ],
    }
    out_json = args.output_dir / "population_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 64)
    print(f"Phase 3 done.  Group {args.group}, {len(selected)} topologies in population.")
    print(f"  protocol     : v2.0 (per-target × per-size up-to-{args.per_target_per_size})")
    print(f"  manifest     : {out_pkl}")
    print(f"  summary      : {out_json}")
    print(f"  by size      : {summary['by_size']}")
    print(f"     agent     : {summary['by_size_agent_only']}")
    if n_baseline:
        print(f"     baseline  : {summary['by_size_baseline_only']}")
    print(f"     parser-pk : {summary['by_size_parser_picks']}")
    print(f"  per-entry NNGGA pkls under: {nngga_root}/")
    print("=" * 64)


if __name__ == "__main__":
    main()
