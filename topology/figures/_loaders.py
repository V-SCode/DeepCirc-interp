"""Path-abstracted CSV loaders for the figure pipeline.

Every fig0N_*.py script reads through this module so paths are resolved once
and group-swapping (G1 → G2 → G3) needs no script edits. Loaders return
pandas DataFrames; callers don't see paths directly.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]  # DeepCirc-interp/
DATA = REPO_ROOT / "data"


def _data_dir(group: str) -> Path:
    # New layout: data/topology_g{1,2,3}/...
    p = DATA / f"topology_{group.lower()}"
    if not p.is_dir():
        raise FileNotFoundError(f"Data directory not found: {p}")
    return p


# ---------------------------------------------------------------------------
# L2 single-part enrichment
# ---------------------------------------------------------------------------

def load_l2(group: str = "G1") -> pd.DataFrame:
    """Universal L2 enrichment: one row per (task, fp_key, part_idx)."""
    return pd.read_csv(_data_dir(group) / "l2" / "l2_enrichment.csv")


def load_l2_by_target(group: str = "G1") -> pd.DataFrame:
    """Per-target L2 enrichment: one row per (task, source, fp_key, part_idx)."""
    return pd.read_csv(_data_dir(group) / "l2" / "l2_enrichment_by_target.csv")


# ---------------------------------------------------------------------------
# L2 pairwise compound enrichment
# ---------------------------------------------------------------------------

def load_l2_pairwise(group: str = "G1") -> pd.DataFrame:
    """Pairwise compound enrichment (gzipped CSV — large)."""
    path = _data_dir(group) / "l2" / "l2_pairwise_enrichment.csv.gz"
    with gzip.open(path) as f:
        return pd.read_csv(f)


def load_l2_pairwise_summary(group: str = "G1") -> dict:
    """L2 pairwise compound summary JSON (P12 output).

    Schema:
      - scope: include_baselines / size_classes / scope_note
      - n_topologies_processed, n_topologies_skipped
      - thresholds: min_topologies, min_targets, min_total_designs, top_k_per_task
      - top_compound_rules_per_task: dict[task_name, list[rule_dict]]
        Each rule_dict has: part_a, part_b, fp_a, fp_b, same_role, log_odds,
        direction, n_topologies, n_targets, total_designs, top_count, bot_count.
      - per_topology: dict (per-topology metadata)
    """
    path = _data_dir(group) / "l2" / "l2_pairwise_summary.json"
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# L3 topology features + correlations
# ---------------------------------------------------------------------------

def load_l3_features(group: str = "G1") -> pd.DataFrame:
    """Per-topology features + score envelope (one row per analyzed topology in <group>)."""
    return pd.read_csv(_data_dir(group) / "l3" / "l3_topology_features.csv")


def load_l3_correlations(group: str = "G1") -> pd.DataFrame:
    """Feature × envelope-metric Pearson correlation table."""
    return pd.read_csv(_data_dir(group) / "l3" / "l3_correlations.csv")


# ---------------------------------------------------------------------------
# L3 motif counts + correlations (dual-view: structural / typed)
# ---------------------------------------------------------------------------

def load_motif_counts(group: str = "G1", view: str = "structural", k: int = 3,
                      exclude_in_nodes: bool = False) -> pd.DataFrame:
    """Load topology × motif counts table.

    Pass ``exclude_in_nodes=True`` to load the no-IN variant produced by
    P13 ``--exclude_in_nodes`` (filenames carry a ``_no_in`` suffix).
    """
    if view not in ("structural", "typed"):
        raise ValueError(f"view must be 'structural' or 'typed'; got {view!r}")
    k_suffix = "" if k == 3 else f"_k{k}"
    in_suffix = "_no_in" if exclude_in_nodes else ""
    return pd.read_csv(
        _data_dir(group) / "l3" / f"l3_motif_counts_{view}{k_suffix}{in_suffix}.csv"
    )


def load_motif_correlations(group: str = "G1", view: str = "structural", k: int = 3,
                            exclude_in_nodes: bool = False) -> pd.DataFrame:
    """Load motif × score partial-r correlations.

    Pass ``exclude_in_nodes=True`` to load the no-IN variant produced by
    P13 ``--exclude_in_nodes`` (filenames carry a ``_no_in`` suffix).
    """
    if view not in ("structural", "typed"):
        raise ValueError(f"view must be 'structural' or 'typed'; got {view!r}")
    k_suffix = "" if k == 3 else f"_k{k}"
    in_suffix = "_no_in" if exclude_in_nodes else ""
    return pd.read_csv(
        _data_dir(group) / "l3" / f"l3_motif_correlations_{view}{k_suffix}{in_suffix}.csv"
    )


def load_motif_summary(group: str = "G1", view: str = "structural", k: int = 3,
                       exclude_in_nodes: bool = False) -> dict:
    """Load the per-(view, k) motif summary JSON (top motifs by |partial r|).

    Pass ``exclude_in_nodes=True`` for the no-IN variant.
    """
    if view not in ("structural", "typed"):
        raise ValueError(f"view must be 'structural' or 'typed'; got {view!r}")
    k_suffix = "" if k == 3 else f"_k{k}"
    in_suffix = "_no_in" if exclude_in_nodes else ""
    with open(_data_dir(group) / "l3" / f"l3_motif_summary_{view}{k_suffix}{in_suffix}.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fold_baseline_to_target(df: pd.DataFrame, source_col: str = "source") -> pd.DataFrame:
    """Strip 'baseline:HEX/...' prefix and group with parent target.

    Adds a `target` column (e.g. '0x17' for both '0x17' and 'baseline:0x17/...').
    Original `source` column preserved.
    """
    out = df.copy()
    out["target"] = out[source_col].astype(str).str.replace("baseline:", "", regex=False).str.split("/").str[0]
    return out


# ---------------------------------------------------------------------------
# Substrate summary (L3 summary JSON: tier counts, source counts, markers)
# ---------------------------------------------------------------------------

def load_substrate_summary(group: str = "G1") -> dict:
    """Load L3 substrate summary JSON (P10 output).

    Schema:
      - n_topologies: int — analyzed topologies (excludes markers)
      - n_skipped:    int — A0 marker count
      - by_tier:      dict[tier_name, count] — A1/A2/A3 counts (A0 = n_skipped)
      - by_source:    dict[target_hex, count] — per-target totals (analyzed only)
      - top_features_for_*: per-envelope-metric Pearson rankings
      - scope: include_baselines / size_classes / scope_locked_on / scope_note
    """
    path = _data_dir(group) / "l3" / "l3_summary.json"
    return json.loads(path.read_text())


def population_substrate_table(group: str = "G1",
                               targets_order: list[str] | None = None,
                               sizes: list[int] | None = None) -> pd.DataFrame:
    """Per-(target × size) topology counts for the analyzed substrate.

    Returns a DataFrame indexed by `regulator_count`, columns = target hexes,
    values = count of analyzed (A1+A2+A3) topologies in that (target, size)
    cell. Use `load_substrate_summary(group)["n_skipped"]` for the A0 marker
    count (markers don't have analysis records so don't appear here).

    `targets_order` and `sizes` set the column / row order if provided;
    otherwise inferred from the data.

    Empty cells (e.g. 0x6D 4-reg = structural floor) appear as 0.
    """
    feat = load_l3_features(group)
    if "is_baseline" in feat.columns:
        feat = feat[~feat["is_baseline"]]
    if targets_order is None:
        targets_order = sorted(feat["source"].unique().tolist())
    if sizes is None:
        sizes = sorted(feat["regulator_count"].unique().tolist())

    table = (feat.groupby(["regulator_count", "source"])
                  .size()
                  .unstack(fill_value=0)
                  .reindex(index=sizes, columns=targets_order, fill_value=0))
    return table


# ---------------------------------------------------------------------------
# Pareto frontier (P14 outputs)
# ---------------------------------------------------------------------------

def load_pareto_knees(group: str = "G3") -> pd.DataFrame:
    """Per-topology knee designs (one row per topology).

    Schema: topology_id, source, gate_count, qc_tier, n_perms_trained,
    n_designs_total, n_perm_front_union, n_pareto_front, circuit_log_max,
    toxicity_max, plus knee_{A,B,C}_{perm,design_row_idx,circuit_log,
    toxicity,part_names,gate_assignments}.

    part_names + gate_assignments are JSON strings.
    """
    return pd.read_csv(_data_dir(group) / "pareto" / "knee_designs.csv")


def load_pareto_fronts(group: str = "G3") -> pd.DataFrame:
    """All topologies' Pareto-front rows (5K+ rows for G3).

    Schema: topology_id, perm_idx, circuit_log, toxicity, gate_assignments,
    part_names, source, gate_count. Each row is one Pareto-undominated design.
    """
    path = _data_dir(group) / "pareto" / "all_topology_fronts.csv.gz"
    if path.exists():
        with gzip.open(path) as f:
            return pd.read_csv(f)
    # Fallback to uncompressed
    return pd.read_csv(_data_dir(group) / "pareto" / "all_topology_fronts.csv")


def load_pareto_portfolio(group: str = "G3") -> pd.DataFrame:
    """Buildable cross-target portfolio (filtered + ranked knees)."""
    return pd.read_csv(_data_dir(group) / "pareto" / "cross_target_portfolio.csv")


def load_pareto_summary(group: str = "G3") -> dict:
    """Pareto summary JSON: scope, thresholds, per-knee aggregates."""
    return json.loads((_data_dir(group) / "pareto" / "pareto_summary.json").read_text())


# ---------------------------------------------------------------------------
# Part substitution matrix (P15 outputs)
# ---------------------------------------------------------------------------

def load_part_correlations(group: str = "G3") -> pd.DataFrame:
    """Long-format pairwise part correlations.

    Cols: task ∈ {circuit_log, toxicity, combined}, method ∈ {pearson, spearman},
    part_a, part_b, corr, n_shared_cells, low_confidence.
    """
    return pd.read_csv(_data_dir(group) / "substitution" / "part_correlations.csv")


def load_functional_groups(group: str = "G3") -> pd.DataFrame:
    """Per-(task, method, part) cluster assignments at three |corr| cutoffs.

    Cols: task, method, part_name, family, cluster_corr_0.5,
    cluster_corr_0.7, cluster_corr_0.85, n_self_cells.
    """
    return pd.read_csv(_data_dir(group) / "substitution" / "functional_groups.csv")


def load_substitution_summary(group: str = "G3") -> dict:
    """Substitution summary JSON."""
    return json.loads(
        (_data_dir(group) / "substitution" / "substitution_summary.json").read_text())


def correlation_matrix(group: str, task: str, method: str) -> pd.DataFrame:
    """Reconstruct a square correlation matrix (parts × parts) for a
    given task + method from the long-format correlations CSV.
    """
    df = load_part_correlations(group)
    sub = df[(df["task"] == task) & (df["method"] == method)]
    parts = sorted(set(sub["part_a"]) | set(sub["part_b"]))
    mat = pd.DataFrame(np.eye(len(parts)), index=parts, columns=parts)
    for _, r in sub.iterrows():
        v = r["corr"]
        mat.at[r["part_a"], r["part_b"]] = v
        mat.at[r["part_b"], r["part_a"]] = v
    return mat


# ---------------------------------------------------------------------------
# Marker failure-mode taxonomy (P16 outputs)
# ---------------------------------------------------------------------------

def load_marker_features(group: str = "G3") -> pd.DataFrame:
    """All 259 topologies' graph features + tier."""
    return pd.read_csv(_data_dir(group) / "markers" / "all_features_with_tier.csv")


def load_feature_differential(group: str = "G3") -> pd.DataFrame:
    """Per-feature Cohen's d for markers (A0) vs A1, sorted by |d|."""
    return pd.read_csv(_data_dir(group) / "markers" / "feature_differential.csv")


def load_marker_clusters(group: str = "G3") -> pd.DataFrame:
    """44 markers × cluster_k2/k3/k4 + z-scored features."""
    return pd.read_csv(_data_dir(group) / "markers" / "marker_clusters.csv")


def load_marker_a1_pairs(group: str = "G3") -> pd.DataFrame:
    """Each marker → nearest-A1 with feature deltas."""
    return pd.read_csv(_data_dir(group) / "markers" / "marker_a1_pairs.csv")


def load_design_rule_nots(group: str = "G3") -> dict:
    """Decision-tree-derived design-rule-NOTs."""
    return json.loads(
        (_data_dir(group) / "markers" / "design_rule_nots.json").read_text())


def load_marker_summary(group: str = "G3") -> dict:
    """Marker taxonomy summary JSON."""
    return json.loads(
        (_data_dir(group) / "markers" / "marker_summary.json").read_text())


# ---------------------------------------------------------------------------
# Core part set (P17 outputs)
# ---------------------------------------------------------------------------

def load_part_frequency_tiers(group: str = "G3") -> pd.DataFrame:
    """Per-(tier, weighting, part) frequency + ranks."""
    return pd.read_csv(_data_dir(group) / "core_parts" / "part_frequency_tiers.csv")


def load_family_frequency_tiers(group: str = "G3") -> pd.DataFrame:
    """Per-(tier, weighting, family) frequency + ranks."""
    return pd.read_csv(_data_dir(group) / "core_parts" / "family_frequency_tiers.csv")


def load_cumulative_coverage(group: str = "G3") -> pd.DataFrame:
    """Per-(tier, weighting, K) cumulative-coverage curve."""
    return pd.read_csv(_data_dir(group) / "core_parts" / "cumulative_coverage.csv")


def load_recommended_inventories(group: str = "G3") -> dict:
    """Scenario lists per (tier, weighting)."""
    return json.loads(
        (_data_dir(group) / "core_parts" / "recommended_inventories.json").read_text())


def load_core_parts_summary(group: str = "G3") -> dict:
    """Core-part summary JSON."""
    return json.loads(
        (_data_dir(group) / "core_parts" / "core_parts_summary.json").read_text())


# ---------------------------------------------------------------------------
# PPO+GAT latent embedding (P18 outputs)
# ---------------------------------------------------------------------------

def load_ppo_latents(group: str = "G3") -> np.ndarray:
    """Stack of 128-d topology embeddings from the trained Stage-1 agent."""
    return np.load(_data_dir(group) / "ppo_latent" / "latents.npy")


def load_ppo_latent_metadata(group: str = "G3") -> pd.DataFrame:
    """Per-topology metadata aligned to latents.npy row order."""
    return pd.read_csv(_data_dir(group) / "ppo_latent" / "latent_metadata.csv")


def load_ppo_cluster_quality(group: str = "G3") -> dict:
    """Silhouette + ARI metrics per labeling."""
    return json.loads(
        (_data_dir(group) / "ppo_latent" / "cluster_quality.json").read_text())


def load_ppo_latent_summary(group: str = "G3") -> dict:
    """PPO latent summary JSON."""
    return json.loads(
        (_data_dir(group) / "ppo_latent" / "ppo_latent_summary.json").read_text())


# ---------------------------------------------------------------------------
# PPO+GAT latent ablations (P19 / Plan 5.5 outputs — const4 / const7 / gat_only)
# ---------------------------------------------------------------------------

_VALID_ABLATION_MODES = ("const4", "const7", "gat_only")


def _ablation_dir(group: str, mode: str) -> Path:
    if mode not in _VALID_ABLATION_MODES:
        raise ValueError(f"unknown ablation mode {mode!r}; "
                         f"expected one of {_VALID_ABLATION_MODES}")
    return _data_dir(group) / "ppo_latent_ablation" / mode


def load_ppo_ablation_latents(mode: str, group: str = "G3") -> np.ndarray:
    """128-d (const4/const7) or 256-d (gat_only) topology embeddings."""
    return np.load(_ablation_dir(group, mode) / "latents.npy")


def load_ppo_ablation_metadata(mode: str, group: str = "G3") -> pd.DataFrame:
    """Per-topology metadata for one ablation mode."""
    return pd.read_csv(_ablation_dir(group, mode) / "latent_metadata.csv")


def load_ppo_ablation_cluster_quality(mode: str, group: str = "G3") -> dict:
    """Silhouette + ARI metrics for one ablation mode."""
    return json.loads(
        (_ablation_dir(group, mode) / "cluster_quality.json").read_text())


def load_ppo_ablation_summary(mode: str, group: str = "G3") -> dict:
    """Summary JSON for one ablation mode."""
    return json.loads(
        (_ablation_dir(group, mode) / "ablation_summary.json").read_text())


# ---------------------------------------------------------------------------
# Stage-2 MLP activation fingerprints (P20 / Plan 6 outputs — circuit + toxicity)
# ---------------------------------------------------------------------------

_VALID_MLP_TASKS = ("circuit", "toxicity")


def _mlp_fp_dir(group: str, task: str) -> Path:
    if task not in _VALID_MLP_TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of "
                         f"{_VALID_MLP_TASKS}")
    return _data_dir(group) / "mlp_fingerprint" / task


def load_mlp_fingerprints(task: str, group: str = "G3",
                          *, normalization: str = "l2") -> np.ndarray:
    """Per-(topology, task) fingerprint matrix.

    normalization='l2' returns L2-normalized canonical fingerprints
    (shape comparison; default). normalization='raw' returns the
    pre-normalization mean-activation fingerprints (magnitude included).
    """
    fname = {"l2": "fingerprints.npy",
             "raw": "fingerprints_raw.npy"}[normalization]
    return np.load(_mlp_fp_dir(group, task) / fname)


def load_mlp_fingerprint_metadata(task: str, group: str = "G3") -> pd.DataFrame:
    """Per-topology metadata aligned to fingerprints.npy row order."""
    return pd.read_csv(_mlp_fp_dir(group, task) / "fingerprint_metadata.csv")


def load_mlp_fingerprint_cluster_quality(task: str, group: str = "G3",
                                          *, normalization: str = "l2") -> dict:
    """Silhouette + ARI metrics per labeling.

    normalization='l2' returns L2-normalized canonical metrics
    (cluster_quality.json); normalization='raw' returns the
    magnitude-included view (cluster_quality_raw.json).
    """
    fname = {"l2": "cluster_quality.json",
             "raw": "cluster_quality_raw.json"}[normalization]
    return json.loads((_mlp_fp_dir(group, task) / fname).read_text())


def load_mlp_fingerprint_summary(task: str, group: str = "G3") -> dict:
    """Phase 20 summary JSON (includes both L2 + raw cluster metrics,
    plus fp_norm_stats diagnostic)."""
    return json.loads(
        (_mlp_fp_dir(group, task) / "fingerprint_summary.json").read_text())


# ---------------------------------------------------------------------------
# Multi-tier (slot, part) frequency patterns (P21 / Plan 7 outputs)
# ---------------------------------------------------------------------------

def _recurring_dir(group: str) -> Path:
    return _data_dir(group) / "recurring_patterns"


def load_tier_frequency_table(group: str = "G3") -> pd.DataFrame:
    """Single-element (graph_role, part, tier) frequencies for P21."""
    return pd.read_csv(_recurring_dir(group) / "tier_frequency_table.csv")


def load_pairwise_tier_frequency(group: str = "G3") -> pd.DataFrame:
    """Pairwise (graph_role_pair, part_pair, tier) frequencies. Auto-
    handles gzipped variant (large file usually gzipped before commit)."""
    base = _recurring_dir(group)
    csv = base / "pairwise_tier_frequency.csv"
    gz = base / "pairwise_tier_frequency.csv.gz"
    if csv.exists():
        return pd.read_csv(csv)
    if gz.exists():
        return pd.read_csv(gz)
    raise FileNotFoundError(f"missing pairwise_tier_frequency.csv{{,.gz}} at {base}")


def load_discriminative_cells(group: str = "G3") -> pd.DataFrame:
    """Pivoted single-element cells with sharpness + monotonicity flags."""
    return pd.read_csv(_recurring_dir(group) / "discriminative_cells.csv")


def load_pattern_catalogs(group: str = "G3") -> dict:
    """Top-K success / failure / non-monotonic catalogs."""
    return json.loads(
        (_recurring_dir(group) / "success_failure_pattern_catalogs.json").read_text())


def load_phase21_summary(group: str = "G3") -> dict:
    """Headline counts + scope + per-task metrics."""
    return json.loads(
        (_recurring_dir(group) / "phase21_summary.json").read_text())


def qc_tier_counts(group: str = "G1") -> dict[str, int]:
    """QC tier counts {A1, A2, A3, A0} for the substrate.

    A1/A2/A3 from `l3_summary.json["by_tier"]`; A0 from `n_skipped`. Tiers
    not present in by_tier are reported as 0.
    """
    summary = load_substrate_summary(group)
    by_tier = summary.get("by_tier", {})
    return {
        "A1": int(by_tier.get("A1", 0)),
        "A2": int(by_tier.get("A2", 0)),
        "A3": int(by_tier.get("A3", 0)),
        "A0": int(summary.get("n_skipped", 0)),
    }
