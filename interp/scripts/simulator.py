"""Local simulator wrapper around `utils5.py` from the upstream DeepCirc repo.

Provides two top-level functions:

    simulate_perm(name, perm) -> {
        circuit_score, growth_score, interference_flag,
        output_signals_per_state, intermediate_inputs_per_state,
        per_gate_growth_per_state,
    }

    simulate_perms_batch(name, perms) -> DataFrame
        Vectorized for N permutations. Same per-row columns as `simulate_perm`
        plus the scalar scores. Keeps per-state detail compact.

Requires the upstream repo cloned at
`/Users/venkatvege/deepcircMI/DeepCirc_upstream` (for `dgd.utils.utils5` and
`dgd/data/*_DeepCirc.json`).
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

UPSTREAM_ROOT = Path("/Users/venkatvege/deepcircMI/DeepCirc_upstream")
UPSTREAM_DATA = UPSTREAM_ROOT / "dgd" / "data"

if str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

from dgd.utils import utils5 as U  # noqa: E402

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from loaders import EXEMPLARS, load_yellow_dot_graph  # noqa: E402


# ---------- static data (loaded once) ----------

@lru_cache(maxsize=1)
def library_df() -> pd.DataFrame:
    """20-row DataFrame indexed 0..19 with columns Repressor, RBS, ymaxa, ymina, Ka, n.

    Matches `assign_representations_with_io_nodes_3`'s expected `df` argument.
    """
    with open(UPSTREAM_DATA / "response_data_3_inputs_DeepCirc.json") as f:
        d = json.load(f)
    df = pd.DataFrame({
        "Repressor": d["Repressor"],
        "RBS": d["RBS"],
        "ymaxa": d["ymaxa"],
        "ymina": d["ymina"],
        "Ka": d["Ka"],
        "n": d["n"],
    })
    return df


@lru_cache(maxsize=1)
def input_states() -> list[dict]:
    """The 8 physical input-signal assignments (dicts mapping input-node-id -> RPU)."""
    table, _names, _idxs, _namelist = U.load_input_data(
        str(UPSTREAM_DATA / "input_data_3_inputs_DeepCirc.json")
    )
    return table


@lru_cache(maxsize=1)
def logical_states() -> list[dict]:
    """The 8 logical input-signal assignments matched to `input_states`."""
    return U.binary_truth_table(3)


@lru_cache(maxsize=1)
def growth_df() -> pd.DataFrame:
    """Gate-level growth (aka toxicity) interpolation table.

    Each row: gate_name (e.g. "F1_AmeR"), input list, growth list.
    """
    with open(UPSTREAM_DATA / "growth_data_3_inputs_DeepCirc.json") as f:
        data = json.load(f)
    return pd.DataFrame(data)


# ---------- topology extraction ----------

def _topology_nodes_edges(name: str):
    """Return (adj_matrix, node_order) for the Fig. 3B exemplar.

    adj_matrix: (N, N) int array; node_order: list of node IDs in row/col order.

    The yellow-dot pickle ships a `nx.DiGraph` with nodes 0..N-1 (inputs at
    low indices, regulators at high indices, single output in between).
    """
    G = load_yellow_dot_graph(name)
    nodes = sorted(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes, dtype=np.int64)
    return A, nodes


# ---------- per-permutation simulation ----------

def build_assigned_graph(name: str, perm) -> nx.DiGraph:
    """Return a fully-assigned DiGraph for `name`'s topology with `perm`.

    Uses the upstream `assign_representations_with_io_nodes_3` so the slot
    ordering matches the training pipeline exactly (confirmed by Sebastian).
    """
    A, _nodes = _topology_nodes_edges(name)
    df = library_df()
    perm = [int(x) for x in perm]
    return U.assign_representations_with_io_nodes_3(A, df, perm)


def simulate_perm(name: str, perm, include_detail: bool = False) -> dict:
    """Full simulator: circuit score + growth score + interference for one perm."""
    G = build_assigned_graph(name, perm)
    states = input_states()
    logical = logical_states()

    out, c_score = _circuit_score(G, states, logical)
    g_score, g_detail = U.calculate_toxicity_score(states, G, growth_df())
    interference, _ = U.is_interference(G)

    result = {
        "circuit_score": float(c_score) if c_score is not None else None,
        "growth_score": float(g_score),
        "interference_flag": bool(interference),
    }
    if include_detail:
        result.update({
            "outputs_per_state": g_detail["outputs"],
            "intermediates_per_state": g_detail["intermediates"],
            "per_gate_growth_per_state": g_detail["growth_scores"],
            "multiplied_growth_per_state": g_detail["multiplied_growth_scores"],
        })
    return result


def _circuit_score(G, input_states_list, logical_states_list):
    """Reproduces the paper pipeline: propagate, build logical/physical per-state dicts,
    then compute min(ON) / max(OFF). Returns (physical_outputs_per_state, score)."""
    physical = U.simulate_signal_propagation(G, input_states_list)
    # Each entry is a dict {output_node: signal_sum}. Flatten to one (output_node, val)
    # per state. For multi-output graphs the upstream takes the first pair; all our
    # exemplars have a single output node (node 3) post-pickle.
    logical_list = []
    physical_list = []
    for log_dict, phys_dict in zip(logical_states_list, physical):
        # logical is per-input dict; we convert to per-output by OR-ing all inputs that
        # are labeled ON. The paper's truth-table is hex-encoded per the 8 states,
        # but the simulator's `calculate_circuit_score` expects a logical-output
        # flag, not raw inputs. For the training pipeline this was computed via
        # `calculate_truth_table_v2` from the topology -- we match by using the
        # exemplar's hex ID.
        # Instead of reconstructing logic from the DAG (which is expensive),
        # use hex_id_bit.
        logical_list.append({list(phys_dict.keys())[0]: None})  # placeholder, overwritten
        physical_list.append({k: phys_dict[k] for k in list(phys_dict.keys())[:1]})
    return physical, None  # overwritten below


# The placeholder above is replaced by the exemplar-specific implementation below.
def _circuit_score(G, input_states_list, logical_states_list):  # noqa: F811
    """Proper version: derive per-state logical outputs from the topology itself."""
    logical_outputs = U.simulate_signal_propagation_binary(G, logical_states_list)
    physical = U.simulate_signal_propagation(G, input_states_list)

    logical_list = []
    physical_list = []
    for log_dict, phys_dict in zip(logical_outputs, physical):
        k_log = list(log_dict.keys())[0]
        k_phy = list(phys_dict.keys())[0]
        logical_list.append({k_log: bool(log_dict[k_log])})
        physical_list.append({k_phy: float(phys_dict[k_phy])})

    score = U.calculate_circuit_score(logical_list, physical_list)
    return physical, score


def simulate_perms_batch(name: str, perms, progress: bool = False) -> pd.DataFrame:
    """Vectorized simulation of a batch of perms. Returns per-row DataFrame."""
    rows = []
    N = len(perms)
    for i, p in enumerate(perms):
        r = simulate_perm(name, p)
        r["permutation"] = tuple(int(x) for x in p)
        rows.append(r)
        if progress and (i + 1) % 100 == 0:
            print(f"  {i+1}/{N}", flush=True)
    return pd.DataFrame(rows)


__all__ = [
    "library_df", "input_states", "logical_states", "growth_df",
    "build_assigned_graph", "simulate_perm", "simulate_perms_batch",
]
