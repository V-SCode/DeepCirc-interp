"""Single-source-of-truth loaders for the three Fig. 3B exemplar folders.

Run `01_loaders.py` for a smoke test.

All functions return plain Python / pandas / torch objects — no DeepCirc-specific
wrapper types. `toxicity_score` on disk is the paper's growth score; loaders here
expose both names where useful (we keep the disk name for keys into raw files to
avoid silent mismatches, and surface `growth_score` as an alias).
"""
from __future__ import annotations

import ast
import json
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

LIBRARY_SIZE = 20  # 20-part TetR-family library per paper
SENSOR_ORDER = ["IPTG", "aTc", "Ara"]  # increasing significance per paper (Methods)


# Exemplar folder root.
#
# Default lives at <repo>/data/exemplars/ so the pipeline works out-of-the-box
# once data is downloaded; override with the DEEPCIRC_EXEMPLARS env var to
# point at another location (e.g. a Zenodo cache or a USB drive).
_DEFAULT_EXEMPLARS_DIR = Path(__file__).resolve().parents[2] / "data" / "exemplars"
EXEMPLARS_ROOT = Path(
    os.environ.get("DEEPCIRC_EXEMPLARS", str(_DEFAULT_EXEMPLARS_DIR))
).expanduser()


@dataclass(frozen=True)
class Exemplar:
    name: str          # "0x2B" / "0x17" / "0x6D"
    folder: str        # directory name under EXEMPLARS_ROOT
    suffix: str        # file suffix {graph_idx}_{perm_idx}
    n_regs: int        # regulator count
    graph_idx: int
    perm_idx: int

    @property
    def dir(self) -> Path:
        return EXEMPLARS_ROOT / self.folder


EXEMPLARS: dict[str, Exemplar] = {
    "0x2B": Exemplar("0x2B", "0x2B_design", "0_0", 5, 0, 0),
    "0x17": Exemplar("0x17", "0x17_design", "1_4", 6, 1, 4),
    "0x6D": Exemplar("0x6D", "0x6D_design", "0_1", 7, 0, 1),
}


# Display-name mapping for figure titles. The hex IDs identify specific
# topologies; for design-space-wide analyses (landscape audits, ANOVA
# decompositions, funnel diagrams, motif catalogs, failure censuses) the
# regulator-count tag is more readable for an audience.
TOPOLOGY_DISPLAY = {
    "0x2B": "5-reg",
    "0x17": "6-reg",
    "0x6D": "7-reg",
}


def display_name(name: str) -> str:
    """'0x2B' → '5-reg', etc.  Use in figure titles for design-space analyses."""
    return TOPOLOGY_DISPLAY.get(name, name)


# -------- permutation / encoding utilities --------

def _parse_bracketed_perm(s: str) -> list[int]:
    """Parse strings like '[ 2 19 15 13 12]' as a list of ints."""
    inner = s.strip().strip("[]")
    return [int(x) for x in inner.split()]


def _parse_dash_perm(s: str) -> list[int]:
    return [int(x) for x in s.split("-")]


def one_hot_perm(perm, g: int = LIBRARY_SIZE) -> torch.Tensor:
    """Encode a length-l int-permutation as a flat (l*g,) float32 tensor."""
    perm = np.asarray(perm, dtype=np.int64)
    l = int(perm.shape[0])
    oh = np.zeros((l, g), dtype=np.float32)
    oh[np.arange(l), perm] = 1.0
    return torch.from_numpy(oh.flatten())


def one_hot_batch(perms, g: int = LIBRARY_SIZE) -> torch.Tensor:
    """perms: (N, l) int array -> (N, l*g) float32 tensor."""
    perms = np.asarray(perms, dtype=np.int64)
    N, l = perms.shape
    oh = np.zeros((N, l, g), dtype=np.float32)
    ix = np.arange(l)
    for i in range(N):
        oh[i, ix, perms[i]] = 1.0
    return torch.from_numpy(oh.reshape(N, l * g))


# -------- topology / yellow-dot --------

def load_yellow_dot_graph(name: str):
    """Returns networkx.DiGraph with node attrs (type/Repressor/RBS/Hill params)."""
    import networkx as nx  # local import so pure-score scripts don't need networkx

    ex = EXEMPLARS[name]
    path = ex.dir / f"optimal_topology_with_parts_assigned_{name}_{ex.suffix}.pkl"
    with open(path, "rb") as f:
        g = pickle.load(f)
    assert isinstance(g, nx.DiGraph)
    return g


def get_yellow_dot_info(name: str) -> dict:
    """Returns the sim-rank-1 record: perm, circuit_score, growth_score, interference."""
    ex = EXEMPLARS[name]
    df = load_top_ranked(name)
    top = df.iloc[0].to_dict()
    return {
        "perm": top["permutation"],
        "circuit_score": float(top["circuit_score"]),
        "growth_score": float(top["growth_score"]),
        "interference": bool(top["interference"]),
    }


def regulator_slot_metadata(name: str) -> pd.DataFrame:
    """For the yellow-dot of `name`, returns one row per regulator slot:
    [slot_idx, node_id, repressor, rbs, ymaxa, ymina, Ka, n].

    Slot order = ascending networkx node index among regulator nodes.
    Confirmed by cross-ref against yellow-dot perm (see docs/data_inventory.md §2).
    """
    import networkx as nx
    g = load_yellow_dot_graph(name)
    rows = []
    reg_nodes = sorted([n for n, d in g.nodes(data=True) if "Repressor" in d])
    for slot, n in enumerate(reg_nodes):
        d = g.nodes[n]
        rows.append({
            "slot_idx": slot,
            "node_id": n,
            "repressor": d["Repressor"],
            "rbs": d["RBS"],
            "ymaxa": float(d["ymaxa"]),
            "ymina": float(d["ymina"]),
            "Ka": float(d["Ka"]),
            "n_hill": float(d["n"]),
        })
    return pd.DataFrame(rows)


# -------- tabular data --------

def _pickle_list_to_df(paths) -> pd.DataFrame:
    """Concatenate list-of-dicts pickle shards into one DataFrame.

    Schema: circuit_score (float), toxicity_score (float), interference_flag (bool),
    permutation (tuple[int]).
    """
    frames = []
    for p in paths:
        with open(p, "rb") as f:
            rows = pickle.load(f)
        df = pd.DataFrame(rows)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    # Normalize permutation column to tuple-of-ints
    df["permutation"] = df["permutation"].apply(lambda x: tuple(int(v) for v in x))
    # Add growth_score alias
    df["growth_score"] = df["toxicity_score"]
    return df


def load_training_data(name: str) -> pd.DataFrame:
    ex = EXEMPLARS[name]
    subdir = ex.dir / f"training_data_{ex.folder}_{ex.graph_idx}_permutation_{ex.perm_idx}"
    shards = sorted(subdir.glob("results_large_batch_*.pkl"))
    return _pickle_list_to_df(shards)


def load_scored_high_predicted(name: str) -> pd.DataFrame:
    ex = EXEMPLARS[name]
    subdir = ex.dir / f"scores_of_designs_predicted_to_have_high_scores_{name}_{ex.suffix}"
    shards = sorted(subdir.glob("results_large_batch_*.pkl"))
    return _pickle_list_to_df(shards)


def load_predicted_high_perms(name: str) -> np.ndarray:
    """The (N, l) int64 array of NN-predicted-high permutations (no scores)."""
    ex = EXEMPLARS[name]
    return np.load(ex.dir / f"gate_assignments_predicted_to_have_high_scores_{name}_{ex.suffix}.npy")


def load_top_ranked(name: str) -> pd.DataFrame:
    """The top-100 sim-ranked designs CSV. Columns: Rank, circuit_score, growth_score,
    toxicity_score, interference, permutation (tuple[int])."""
    ex = EXEMPLARS[name]
    df = pd.read_csv(ex.dir / f"scores_top_designs_{name}_{ex.suffix}.csv")
    df = df.rename(columns={"Circuit Score": "circuit_score", "Toxicity Score": "toxicity_score"})
    df["growth_score"] = df["toxicity_score"]
    df["permutation"] = df["Permutation"].apply(_parse_bracketed_perm).apply(tuple)
    df = df.drop(columns=["Permutation"])
    return df


def load_top_in_training(name: str) -> pd.DataFrame:
    ex = EXEMPLARS[name]
    # Note filename typo "trainning"
    df = pd.read_csv(ex.dir / f"scores_top_designs_in_trainning_data_{name}_{ex.suffix}.csv")
    df = df.rename(columns={"Circuit Score": "circuit_score", "Toxicity Score": "toxicity_score"})
    df["growth_score"] = df["toxicity_score"]
    df["permutation"] = df["Permutation"].apply(_parse_bracketed_perm).apply(tuple)
    df = df.drop(columns=["Permutation"])
    return df


def load_nn_prediction_of_best(name: str) -> dict:
    """Reads the NN's prediction for the sim-rank-1 design."""
    ex = EXEMPLARS[name]
    csv = ex.dir / f"nn_prediction_of_highest_performing design__circuit_score_{name}_{ex.suffix}.csv"
    df = pd.read_csv(csv)
    row = df.iloc[0]
    return {
        "perm": tuple(_parse_dash_perm(row["permutation"])),
        "circuit_pred": float(row["circuit_score_pred"]),
        "growth_pred": float(row["toxicity_score_pred"]),
    }


def load_run_metadata(name: str) -> dict:
    ex = EXEMPLARS[name]
    with open(ex.dir / f"run_metadata_{name}_{ex.suffix}.json") as f:
        return json.load(f)


def load_test_metrics(name: str) -> pd.DataFrame:
    ex = EXEMPLARS[name]
    return pd.read_csv(ex.dir / "model_performance_on_testing_set.csv")


# -------- models --------

def _build_mlp_from_state_dict(sd: dict) -> nn.Sequential:
    """Reconstruct the MLP from the state_dict.

    Architecture: Linear -> ReLU -> Linear -> ReLU -> ... -> Linear (last has no ReLU).
    Total 10 Linear layers; 9 hidden @ width 100; output dim 1.
    """
    linear_idxs = sorted({int(k.split(".")[1]) for k in sd if k.endswith(".weight")})
    modules = []
    for i, li in enumerate(linear_idxs):
        w = sd[f"model.{li}.weight"]
        out_dim, in_dim = w.shape
        modules.append(nn.Linear(in_dim, out_dim))
        if i < len(linear_idxs) - 1:
            modules.append(nn.ReLU())
    m = nn.Sequential(*modules)
    remapped = {}
    seq_idx = 0
    for li in linear_idxs:
        remapped[f"{seq_idx}.weight"] = sd[f"model.{li}.weight"]
        remapped[f"{seq_idx}.bias"] = sd[f"model.{li}.bias"]
        seq_idx += 2
    m.load_state_dict(remapped)
    m.eval()
    return m


def load_circuit_mlp(name: str) -> nn.Sequential:
    ex = EXEMPLARS[name]
    sd = torch.load(ex.dir / "circuit_score_model.pt", map_location="cpu", weights_only=False)
    return _build_mlp_from_state_dict(sd)


def load_growth_mlp(name: str) -> nn.Sequential:
    """Alias: the on-disk file is `toxicity_score_model.pt` but encodes growth score."""
    ex = EXEMPLARS[name]
    sd = torch.load(ex.dir / "toxicity_score_model.pt", map_location="cpu", weights_only=False)
    return _build_mlp_from_state_dict(sd)


def load_training_losses(name: str) -> dict:
    ex = EXEMPLARS[name]
    return {
        "circuit": torch.load(ex.dir / "circuit_score_model_losses.pt",
                              map_location="cpu", weights_only=False),
        "growth": torch.load(ex.dir / "toxicity_score_model_losses.pt",
                             map_location="cpu", weights_only=False),
    }


@torch.no_grad()
def predict(model: nn.Sequential, perms, g: int = LIBRARY_SIZE,
            batch_size: int = 8192) -> np.ndarray:
    """Run the MLP on a (N, l) int array of permutations. Returns (N,) float array."""
    perms = np.asarray(perms, dtype=np.int64)
    N, l = perms.shape
    out = np.empty(N, dtype=np.float32)
    for i in range(0, N, batch_size):
        x = one_hot_batch(perms[i:i+batch_size], g=g)
        out[i:i+batch_size] = model(x).squeeze(-1).numpy()
    return out


# -------- metric transforms (Phase 0 / Phase 6 ready) --------

def m_logic(circuit_score: np.ndarray | float) -> np.ndarray | float:
    """Worst-case logical margin: log(c)."""
    return np.log(np.clip(circuit_score, 1e-12, None))


def m_burden(growth_score: np.ndarray | float) -> np.ndarray | float:
    """Additive burden cost in worst input state: -log(g)."""
    return -np.log(np.clip(growth_score, 1e-12, None))


__all__ = [
    "EXEMPLARS", "Exemplar", "LIBRARY_SIZE", "SENSOR_ORDER",
    "one_hot_perm", "one_hot_batch",
    "load_yellow_dot_graph", "get_yellow_dot_info", "regulator_slot_metadata",
    "load_training_data", "load_scored_high_predicted", "load_predicted_high_perms",
    "load_top_ranked", "load_top_in_training", "load_nn_prediction_of_best",
    "load_run_metadata", "load_test_metrics",
    "load_circuit_mlp", "load_growth_mlp", "load_training_losses",
    "predict", "m_logic", "m_burden",
]
