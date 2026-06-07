"""fig30 — Size-conditioned typed expansion of non-linear iso classes
(combined k=3/4/5 pool).

Companion to fig29 (size-conditioned structural ranking) — drills into
the iso classes that fig29 surfaces. Iso classes from k=3, k=4, AND
k=5 are pooled into one ranking per size class; the top-3 iso classes
per size (by within-size log_odds_top01_vs_bot01) get a panel each.

2 figures total: 2 tasks (circuit, growth). Layout per figure:
  4 rows (size 4 / 5 / 6 / 7) × 3 panels (top-3 iso classes per size).

Each panel content:
  - Title strip: k = N · log₂(top1_S / bot1_S) at this size · partial r
  - Parent structural skeleton (no labels) — sized to its own k
  - 3 typed variants below ranked by within-size partial r, with NIG
    type-labeled nodes (IN / NOT / NOR-in/mid/out / OUT-NOT/OR2/...).

Reuses fig18_motif_summary.parse_motif + draw_motif for the typed render
(same NIG color palette as fig28).
"""
from __future__ import annotations

import argparse
import sys
from itertools import permutations
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import _style as S
from fig18_motif_summary import parse_motif, NODE_COLORS, NODE_LABELS
# Reuse fig29's bezier-aware structural-DAG renderer for the parent
# skeleton so fig30's "general motif up top" gets the same paper-blue
# nodes, horizontal-edge fidelity, and natural-direction landing as
# fig29. ``draw_edges_on_dag`` + ``layered_layout`` mean the typed
# sub-graphs use the EXACT same layout as the parent skeleton (just
# with NIG-coloured labelled nodes on top instead of plain blue), so a
# reader can match part assignments to structural positions 1-to-1.
# ``_wrap_footer`` matches fig29's multi-line footer.
from fig29_simplemotif_by_size import (
    draw_structural_dag as draw_struct_skeleton,
    draw_edges_on_dag,
    layered_layout,
    _draw_size_separators,
    _wrap_footer,
)


REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "topology_g3" / "motif_tier_analysis"
SIZE_TIER_DATA = REPO / "data" / "topology_g3" / "size_tier_designs"
LAPLACE_ALPHA = 0.5  # additive smoothing for log_odds (handles zero-presence variants)

K_VALUES = (3, 4, 5)
# Display default: 5/6/7-reg only (4-reg dropped 2026-05-20 per user direction).
SIZE_CLASSES_DEFAULT = (5, 6, 7)
SIZE_CLASSES_ALL     = (4, 5, 6, 7)
SIZE_CLASSES = SIZE_CLASSES_DEFAULT  # mutated by CLI in main()

# Motif-view → on-disk filename suffix (mirrors P13/P24/P26/P27).
VIEW_SUFFIX = {"full": "", "no_in": "_no_in",
               "no_out": "_no_out",
               "internal_only": "_internal_only"}
VIEW_TITLE_TAG = {
    "full":          "",
    "no_in":         "  ·  NO-IN (sensor inputs stripped)",
    "no_out":        "  ·  NO-OUT (OUT-* terminal gates stripped)",
    "internal_only": "  ·  INTERNAL-ONLY (IN + OUT-* stripped)",
}
SIZE_LABEL = {s: f"{s}-REG" for s in SIZE_CLASSES}
SIZE_COLOR = {
    4: "#1A9850",
    5: "#5AAE61",
    6: "#F0B90B",
    7: "#D7263D",
}
TASKS = {"circuit": "circuit", "growth": "toxicity"}

N_ISO_PER_SIZE = 3
N_TYPED_PER_PANEL = 3


def _align_typed_to_parent(types: list[str], A_typed: np.ndarray,
                             parent_iso_key: str,
                             k: int) -> tuple[list[str], np.ndarray]:
    """Find a node permutation P such that P @ A_typed @ P^T == the
    parent's canonical adjacency (from parent_iso_key), and apply P to
    the typed-variant's types list so types[i] is the type at the
    parent-canonical position i.

    Why this matters: P27's typed_motif string encodes each typed
    variant in its OWN canonical node order, which generally does NOT
    match the parent iso_key's node order. Passing the raw A_typed to
    layered_layout gives a DIFFERENT layout than the parent's
    skeleton — same shape, rotated/mirrored — which breaks the
    1-to-1 mapping between parent skeleton and typed variants below.
    Aligning to the parent's canonical order fixes the orientation.

    For k ≤ 5 the brute-force permutation enumeration is cheap
    (≤120 permutations). Returns (aligned_types, parent_A).
    """
    A_parent = np.array(
        [int(c) for c in parent_iso_key], dtype=A_typed.dtype
    ).reshape(k, k)
    for perm in permutations(range(k)):
        idx = list(perm)
        new_A = A_typed[np.ix_(idx, idx)]
        if np.array_equal(new_A, A_parent):
            aligned = [types[idx[i]] for i in range(k)]
            return aligned, A_parent
    # No permutation matched — fall back to the raw typed matrix.
    # (Should be unreachable for typed variants of the same iso class.)
    return types, A_typed


def draw_typed_motif(ax, motif_str: str,
                      parent_iso_key: str | None = None, *,
                      node_size: float = 0.22,
                      label_size: float = 7.5) -> None:
    """Render a single typed-variant motif with NIG part-assignment node
    styling (fig18 NODE_COLORS / NODE_LABELS) ON TOP of fig29's
    bezier-aware edge logic.

    Layout: fig29's layered_layout, applied to the typed variant's
    adjacency AFTER permuting it into the PARENT's canonical node
    order (see _align_typed_to_parent). The result lines up
    node-for-node with the parent skeleton above so the reader can
    match part assignment to structural position 1-to-1.

    Edges: draw_edges_on_dag from fig29.
    """
    types, A, k = parse_motif(motif_str)
    if parent_iso_key is not None:
        types, A = _align_typed_to_parent(types, A, parent_iso_key, k)
    coords = layered_layout(A)

    xs = [coords[i][0] for i in range(k)]
    ys = [coords[i][1] for i in range(k)]
    pad_x, pad_y = 0.6, 0.7
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    ax.set_aspect("equal")
    ax.axis("off")

    draw_edges_on_dag(ax, A, coords, k, node_radius=node_size)

    for n in range(k):
        x, y = coords[n]
        face = NODE_COLORS.get(types[n], "#bbb")
        ax.add_patch(plt.Circle((x, y), node_size, facecolor=face,
                                  edgecolor="#333", linewidth=0.8,
                                  zorder=4))
        label_text = NODE_LABELS.get(types[n], types[n])
        is_out = types[n].startswith("OUT")
        ax.text(x, y, label_text, ha="center", va="center",
                fontsize=label_size,
                color="white" if is_out else "#111",
                fontweight="bold", zorder=5)


def _size_tier_totals(task_disk: str,
                       tier_pair: str = "01") -> dict[int, dict[str, int]]:
    """Total number of designs in (top_<TP>, bot_<TP>) within each size class.
    Sums per-topology tier counts from P23.2's size_tier_design_counts.csv,
    restricted to topologies of the given size. Used for Laplace-smoothed
    log_odds at typed-variant level. ``tier_pair`` ∈ {'01', '05'}.
    """
    df = pd.read_csv(SIZE_TIER_DATA / "size_tier_design_counts.csv")
    col_top = f"n_top_{tier_pair}_{task_disk}_S"
    col_bot = f"n_bot_{tier_pair}_{task_disk}_S"
    totals: dict[int, dict[str, int]] = {}
    for s in SIZE_CLASSES:
        sub = df[df["regulator_count"] == s]
        totals[s] = {
            f"top_{tier_pair}": int(sub[col_top].sum()),
            f"bot_{tier_pair}": int(sub[col_bot].sum()),
        }
    return totals


def _attach_typed_log_odds(typed_df: pd.DataFrame,
                            tier_totals: dict[int, dict[str, int]],
                            tier_pair: str = "01") -> pd.DataFrame:
    """Add a Laplace-smoothed log_odds_top<TP>_vs_bot<TP>_S column to typed_df.
    Mirrors the parent convention but with additive smoothing so variants
    with zero presence in either tier still rank finite.
    """
    out = typed_df.copy()
    log2 = np.log(2.0)
    log_odds_vals = np.full(len(out), np.nan)
    top_key = f"top_{tier_pair}"
    bot_key = f"bot_{tier_pair}"
    for i, row in out.iterrows():
        s = int(row["size_class"])
        tot = tier_totals.get(s)
        if tot is None:
            continue
        top_n = float(row.get(f"presence_designs_{top_key}", 0))
        bot_n = float(row.get(f"presence_designs_{bot_key}", 0))
        p_top = (top_n + LAPLACE_ALPHA) / (tot[top_key] + 1.0)
        p_bot = (bot_n + LAPLACE_ALPHA) / (tot[bot_key] + 1.0)
        if p_top > 0 and p_bot > 0:
            log_odds_vals[i] = float(np.log(p_top / p_bot) / log2)
    out[f"log_odds_top{tier_pair}_vs_bot{tier_pair}_S"] = log_odds_vals
    return out


def select_top_iso_per_size(task_disk: str,
                             motif_view: str = "full",
                             tier_pair: str = "01",
                             min_typed_variants: int = 1,
                             ) -> dict[int, pd.DataFrame]:
    """For each size class, pick top-N non-linear iso classes by within-size
    log_odds_top<TP>_vs_bot<TP> across the combined k=3+4+5 pool.

    ``min_typed_variants`` filters parent iso classes by typed-variant
    breadth — the *point* of fig30 is showing how part assignments vary
    the score. With ``min_typed_variants=3`` (recommended default), each
    selected parent has ≥3 NIG-typed variants in the data; with =1 the
    behaviour matches the pre-2026-05-21 select. If no iso class at a
    given size has enough variants, we fall back to the next-lower
    breadth threshold so the panel is never empty.

    ``motif_view`` picks the on-disk CSV suffix.
    """
    suffix = VIEW_SUFFIX[motif_view]
    frames = []
    for k in K_VALUES:
        path = DATA / f"structural_iso_by_size_{task_disk}_k{k}{suffix}.csv"
        if not path.exists():
            print(f"  [warn] missing {path.name}; skipping k={k} for "
                  f"view={motif_view}", file=sys.stderr)
            continue
        df = pd.read_csv(path, dtype={"iso_key": str})
        df["iso_key"] = df["iso_key"].str.zfill(k * k)
        frames.append(df)
    if not frames:
        raise SystemExit(
            f"no structural_iso_by_size_{task_disk}_k*{suffix}.csv files "
            f"found under {DATA}; run P26 with --motif_view {motif_view} first"
        )
    all_iso = pd.concat(frames, ignore_index=True)

    typed = pd.read_csv(
        DATA / f"typed_expansions_by_size_{task_disk}{suffix}.csv",
        dtype={"parent_iso_key": str},
    )
    typed["parent_iso_key"] = typed.apply(
        lambda r: str(r["parent_iso_key"]).zfill(int(r["k"]) ** 2),
        axis=1)
    # n_typed_variants per (k, size_class, parent_iso_key)
    typed_breadth = (
        typed.groupby(["k", "size_class", "parent_iso_key"])
              .size()
              .to_dict()
    )

    rank_col = f"log_odds_top{tier_pair}_vs_bot{tier_pair}"
    out: dict[int, pd.DataFrame] = {}
    for s in SIZE_CLASSES:
        pool = all_iso[(all_iso["size_class"] == s) & ~all_iso["is_linear"]]
        pool = pool.dropna(subset=[rank_col]).copy()
        pool["n_typed_variants"] = pool.apply(
            lambda r: typed_breadth.get((int(r["k"]), s, r["iso_key"]), 0),
            axis=1,
        )
        # Walk breadth thresholds DOWN from the requested floor to 1 so a
        # sparse size still fills its panels with the best available.
        chosen = pd.DataFrame()
        for floor in range(int(min_typed_variants), 0, -1):
            cand = pool[pool["n_typed_variants"] >= floor]
            if len(cand) >= N_ISO_PER_SIZE or floor == 1:
                chosen = cand
                break
        top = chosen.sort_values(rank_col,
                                   ascending=False).head(N_ISO_PER_SIZE)
        out[s] = top.reset_index(drop=True)
    return out


def panel_for_parent(ax_panel, parent_row: pd.Series,
                       typed_df: pd.DataFrame, *,
                       size_color: str,
                       tier_pair: str = "01") -> None:
    """One panel: parent skeleton on top + top-3 typed variants below."""
    parent_col = f"log_odds_top{tier_pair}_vs_bot{tier_pair}"
    typed_col  = f"log_odds_top{tier_pair}_vs_bot{tier_pair}_S"
    top_design_col = f"presence_designs_top_{tier_pair}"
    bot_design_col = f"presence_designs_bot_{tier_pair}"
    tier_pct = {"01": "1", "05": "5", "10": "10", "25": "25"}[tier_pair]
    ax_panel.set_xlim(0, 100); ax_panel.set_ylim(0, 100); ax_panel.axis("off")
    k = int(parent_row["k"])
    iso_key = parent_row["iso_key"]
    s = int(parent_row["size_class"])
    log_odds = float(parent_row[parent_col])

    # Volume = how many designs in the matched-tail elite + failure pool
    # contain this structural motif. Replaces the partial_r line per user
    # 2026-05-21: design count is more interpretable than the controlled
    # correlation coefficient at the audience level fig30 targets.
    v_top = int(parent_row.get(top_design_col, 0) or 0)
    v_bot = int(parent_row.get(bot_design_col, 0) or 0)
    volume = v_top + v_bot

    sign = "+" if log_odds >= 0 else ""
    title = (f"k = {k}    log₂(top{tier_pct}_S/bot{tier_pct}_S) "
             f"= {sign}{log_odds:.2f}")
    ax_panel.text(50, 96, title, ha="center", va="top",
                  fontsize=10, fontweight="bold", color="#111")
    n_topo_total = int(parent_row.get("n_topo_with_motif_in_size", 0))
    n_topo_size  = int(parent_row.get("n_topo_in_size", 0))
    ax_panel.text(50, 90,
                   f"n_topo = {n_topo_total} / {n_topo_size}  ·  "
                   f"volume = {volume}",
                   ha="center", va="top",
                   fontsize=8.5, color="#111")

    # Parent skeleton (center-top)
    sk = ax_panel.inset_axes([0.30, 0.60, 0.40, 0.28])
    draw_struct_skeleton(sk, iso_key, k, node_radius=0.20)
    ax_panel.text(50, 56,
                   f"TYPED  VARIANTS  (top 3 by within-size "
                   f"log₂(top{tier_pct}_S/bot{tier_pct}_S))",
                   ha="center", va="top",
                   fontsize=8.5, fontweight="bold", color="#111",
                   family="monospace")

    matches = typed_df[
        (typed_df["k"] == k)
        & (typed_df["parent_iso_key"] == iso_key)
        & (typed_df["size_class"] == s)
    ].dropna(subset=[typed_col]).sort_values(typed_col, ascending=False)

    if len(matches) == 0:
        ax_panel.text(50, 30, "(no typed variants at this size)",
                       ha="center", va="center", fontsize=9,
                       color="#888", style="italic")
        return

    top_variants = matches.head(N_TYPED_PER_PANEL)
    n_show = len(top_variants)
    for i, (_, var) in enumerate(top_variants.iterrows()):
        left = 0.02 + i * (0.96 / n_show)
        width = 0.96 / n_show
        sub = ax_panel.inset_axes([left, 0.04, width, 0.50])
        sub.axis("off")
        sub.set_xlim(0, 1); sub.set_ylim(0, 1)
        dag = sub.inset_axes([0.05, 0.30, 0.90, 0.65])
        try:
            draw_typed_motif(dag, var["typed_motif"],
                              parent_iso_key=iso_key,
                              node_size=0.24, label_size=8.0)
        except Exception as e:
            dag.axis("off")
            dag.text(0.5, 0.5, f"parse err: {e!s:.20s}",
                       ha="center", va="center", fontsize=6)
        lo = float(var[typed_col])
        # User direction 2026-05-21: keep typed-variant log₂ in green
        # (positive) / red (negative) — this colour is informative
        # (sign signal), everything else in this figure is black.
        color = "#117733" if lo >= 0 else "#D7263D"
        sign = "+" if lo >= 0 else ""
        # Volume: how many designs (top + bot tier) contain THIS typed
        # variant. Sits under n_topo per user 2026-05-21.
        v_top = int(var.get(top_design_col, 0) or 0)
        v_bot = int(var.get(bot_design_col, 0) or 0)
        v_volume = v_top + v_bot
        sub.text(0.5, 0.22, f"log₂ = {sign}{lo:.2f}",
                   ha="center", va="center",
                   fontsize=9.5, fontweight="bold", color=color)
        sub.text(0.5, 0.10,
                   f"n_topo = {int(var['n_topo_with_motif_in_size'])}",
                   ha="center", va="center", fontsize=7.5, color="#111")
        sub.text(0.5, 0.02,
                   f"volume = {v_volume}",
                   ha="center", va="center", fontsize=7.5, color="#111")


def build_figure(task_disp: str, motif_view: str = "full",
                 sizes: list[int] | None = None,
                 min_n_topo: int = 1,
                 tier_pair: str = "01",
                 min_typed_variants: int = 3) -> None:
    if tier_pair not in ("01", "05", "10", "25"):
        raise ValueError(f"tier_pair must be '01'|'05'|'10'|'25'; got {tier_pair!r}")
    task_disk = TASKS[task_disp]
    suffix = VIEW_SUFFIX[motif_view]
    sizes = list(sizes) if sizes else list(SIZE_CLASSES)
    top_per_size = select_top_iso_per_size(
        task_disk, motif_view=motif_view, tier_pair=tier_pair,
        min_typed_variants=min_typed_variants,
    )
    # Filter parent iso classes by n_topo_with_motif_in_size if requested.
    if min_n_topo > 1:
        top_per_size = {
            s: df[df["n_topo_with_motif_in_size"] >= min_n_topo]
                 .reset_index(drop=True)
            for s, df in top_per_size.items()
        }
    typed_df = pd.read_csv(
        DATA / f"typed_expansions_by_size_{task_disk}{suffix}.csv",
        dtype={"parent_iso_key": str},
    )
    typed_df["parent_iso_key"] = typed_df.apply(
        lambda r: str(r["parent_iso_key"]).zfill(int(r["k"]) ** 2),
        axis=1)
    tier_totals = _size_tier_totals(task_disk, tier_pair=tier_pair)
    typed_df = _attach_typed_log_odds(typed_df, tier_totals,
                                         tier_pair=tier_pair)

    n_total = sum(len(top_per_size.get(s, [])) for s in sizes)
    if n_total == 0:
        print(f"  skip fig30_typed_by_size_{task_disp}: no eligible parents")
        return

    n_size_rows = len(sizes)
    # Canvas height scales with row count so each row keeps its breathing room.
    fig_h = 22.0 * n_size_rows / 4.0
    fig = plt.figure(figsize=(22.0, fig_h))
    # bottom=0.09 (was 0.04) leaves clean headroom below the last typed
    # row for the multi-line wrapped footer (2026-05-21 user direction).
    gs = fig.add_gridspec(n_size_rows, N_ISO_PER_SIZE,
                           hspace=0.28, wspace=0.10,
                           left=0.05, right=0.97, top=0.93, bottom=0.09)

    for r, s in enumerate(sizes):
        size_color = SIZE_COLOR[s]
        # Size label on left edge (text always black per 2026-05-21 user
        # direction; the size-coloured vertical strip on each cell's left
        # margin still carries the size identity visually).
        fig.text(0.015, 0.93 - (r + 0.5) * (0.84 / n_size_rows),
                  SIZE_LABEL[s], ha="left", va="center", rotation=90,
                  fontsize=11, fontweight="bold", color="#111")
        parents_s = top_per_size.get(s, pd.DataFrame())
        for c in range(N_ISO_PER_SIZE):
            ax = fig.add_subplot(gs[r, c])
            ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
            ax.axvspan(0, 1.5, color=size_color, alpha=0.18)
            if c < len(parents_s):
                panel_for_parent(ax, parents_s.iloc[c], typed_df,
                                  size_color=size_color,
                                  tier_pair=tier_pair)
            else:
                ax.text(50, 50, "(no more iso classes\nat this size)",
                          ha="center", va="center", fontsize=9,
                          color="#888", style="italic")

    # Horizontal separator lines between size groups (one row per size,
    # so separators sit in the gap below each row except the last).
    _draw_size_separators(fig, gs, n_size_rows)

    S.figure_title(fig,
        f"Typed expansion by SIZE  ·  "
        f"{'CIRCUIT' if task_disp == 'circuit' else 'GROWTH'}  ·  "
        f"k = 3 + 4 + 5 (combined)  ·  "
        f"non-linear iso classes with strongest within-size log₂ "
        f"discrimination{VIEW_TITLE_TAG[motif_view]}")
    footer = (
        "Each row = one size class; each panel = one non-linear iso class "
        "(top 3 within size by within-size log₂(top1_S/bot1_S), across the "
        "combined k=3+4+5 pool). k value shown per panel. Parent structural "
        "shape on top; top-3 typed variants below ranked by the same metric "
        "(Laplace-smoothed log₂(top1_S/bot1_S), α=0.5). Typed variants share "
        "the skeleton but differ in NIG node typing (IN / NOT / "
        "NOR-in/mid/out / OUT-NOT/OR2)."
    )
    if motif_view == "internal_only":
        footer += ("  Internal-only view: IN sensors and OUT-* terminal gates "
                    "excluded from motif enumeration. Typed variants therefore "
                    "use only NOT and NOR labels.")
    elif motif_view == "no_in":
        footer += "  No-IN view: sensor inputs excluded; OUT-* gates retained."
    elif motif_view == "no_out":
        footer += ("  No-OUT view: OUT-* terminal gates excluded; IN sensors "
                    "retained (symmetric counterpart to no_in).")
    fig.text(0.5, 0.01, _wrap_footer(footer),
              ha="center", va="bottom", fontsize=9, color="#555",
              style="italic")
    tier_tag = "" if tier_pair == "01" else f"_top{tier_pair}_bot{tier_pair}"
    S.save_figure(fig,
                    f"fig30_typed_by_size_{task_disp}{suffix}{tier_tag}",
                    group="simplemotif")
    plt.close(fig)
    print(f"  rendered fig30_typed_by_size_{task_disp}{suffix}{tier_tag}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="all",
                     choices=["all", "circuit", "growth"])
    ap.add_argument("--motif_view",
                     choices=list(VIEW_SUFFIX.keys()), default="full",
                     help="Motif universe to read (mirrors P13/P24/P26/P27). "
                          "'full' (default), 'no_in', 'no_out', or "
                          "'internal_only'. Outputs gain a matching filename "
                          "suffix.")
    ap.add_argument("--sizes", type=int, nargs="+",
                     default=list(SIZE_CLASSES_DEFAULT),
                     help="Size classes to plot (regulator counts). "
                          "Default = 5 6 7 (4-reg dropped 2026-05-20). Pass "
                          "`--sizes 4 5 6 7` to include 4-reg back in.")
    ap.add_argument("--min_n_topo", type=int, default=1,
                     help="Display filter: drop parent iso classes with "
                          "n_topo_with_motif_in_size < this value. Default 1.")
    ap.add_argument("--tier_pair", choices=["01", "05", "10", "25"],
                     default="01",
                     help="Matched-tail tier pair for parent ranking + "
                          "Laplace-smoothed typed log_odds. '01' (default) "
                          "= top-1%%. '05' / '10' / '25' progressively wider "
                          "tails — each surfaces more narrow shapes by "
                          "intersecting more topology footprints.")
    ap.add_argument("--min_typed_variants", type=int, default=3,
                     help="Filter parent iso classes to those with at least "
                          "this many NIG-typed variants in the data. The "
                          "point of fig30 is showing how part assignment "
                          "varies the score, so 1-variant parents undercut "
                          "the figure. Default 3; falls back to lower "
                          "thresholds at sparse sizes (4-/7-reg internal-only) "
                          "if needed so panels never go empty.")
    args = ap.parse_args()
    S.setup_matplotlib()

    global SIZE_CLASSES
    SIZE_CLASSES = tuple(args.sizes)

    tasks = list(TASKS) if args.task == "all" else [args.task]
    for t in tasks:
        build_figure(t, motif_view=args.motif_view,
                     sizes=args.sizes, min_n_topo=args.min_n_topo,
                     tier_pair=args.tier_pair,
                     min_typed_variants=args.min_typed_variants)


if __name__ == "__main__":
    main()
