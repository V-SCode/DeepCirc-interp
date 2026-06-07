"""Figure S05 — Panel D: cross-topology interaction conservation (m_logic).

Faithful paper-pipeline port of `deepcirc_sae/outputs/figures/presentation/
E4_interaction_conservation`. Same data + same logic + same fingerprint
ordering, restyled to the paper PASTEL palette and grey/non-bold labels.

For each pair (slot_i, slot_j) in the Shapley-Taylor 2-body decomposition
of each yellow-dot (0x2B / 0x17 / 0x6D), retag the pair in graph-relative
coords as `(family@d{distance-to-output})`. Join across the three
topologies to find fingerprint pairs whose interaction is conserved in
sign + magnitude. Render as an N×N heatmap with row/col labels coloured
by TF family, plus a side panel listing the shared pairs explicitly.

Only the circuit objective (m_logic) is visualised. Growth Φᵢⱼ are
O(1e-3) and not the controlling signal in any topology (Module D).

Source data: data/interp_processed/
              shapley_taylor_sim_{0x2B,0x17,0x6D}.json
              (falls back to pairwise_*.json if simulator-valued
               coefficients are absent)

Outputs:
  panels/vector/panel_d.pdf | .svg
  panels/raster/panel_d.png

Run:
    python \
        figures/setFinal/figS15/scripts/build_panel_d.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle

SCRIPT_PATH = Path(__file__).resolve()
PKG_ROOT    = SCRIPT_PATH.parents[4]
REPO_ROOT   = PKG_ROOT
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(REPO_ROOT / "interp" / "scripts"))
import loaders as L  # noqa: E402

from figures.figtools import use_style, save_panel, figsize_mm  # noqa: E402
from figures.styles.colors import (  # noqa: E402
    PASTEL, FAMILY_COLORS, DARK_GRAY, MID_GRAY, LIGHT_GRAY,
)


PANELS_VEC = (PKG_ROOT / "figures" / "setFinal" / "figS15"
              / "panels" / "vector")
MAIN_PROC  = REPO_ROOT / "data" / "interp_processed"

TOPOLOGIES        = ["0x2B", "0x17", "0x6D"]
TOPOLOGY_DISPLAY  = {"0x2B": "5-reg", "0x17": "6-reg", "0x6D": "7-reg"}
TOPOLOGY_COLORS   = {
    "0x2B": "#7FA970",            # green-ish (matches PASTEL family-of-greens)
    "0x17": PASTEL["purple"],     # purple
    "0x6D": PASTEL["orange"],     # orange
}
ALL_PRESENT = 3

TEXT_GREY = "#666666"

# Paper-pastel divergent cmap (red = antagonism, green = synergy).
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "paper_synergy_antagonism",
    [(0.0, PASTEL["red"]),
     (0.5, "#FFFFFF"),
     (1.0, PASTEL["green"])],
)

# --- canvas ----------------------------------------------------------------

CANVAS_W_MM = 220
CANVAS_H_MM = 150


# --- data ------------------------------------------------------------------

def _load_pairwise(name: str) -> dict:
    sim_fp = MAIN_PROC / f"shapley_taylor_sim_{name}.json"
    if sim_fp.exists():
        with open(sim_fp) as f:
            sd = json.load(f)
        return {
            "slot_labels": sd["slot_labels"],
            "pairwise_circuit": sd["circuit"]["pair_shap"],
        }
    with open(MAIN_PROC / f"pairwise_{name}.json") as f:
        d = json.load(f)
    return {
        "slot_labels": d["slot_labels"],
        "pairwise_circuit": d["pairwise_circuit"],
    }


def _topology_slot_coords(name: str) -> dict[int, dict]:
    G = L.load_yellow_dot_graph(name)
    nodes = sorted(G.nodes())
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    output_nodes = [n for n in nodes if out_deg[n] == 0]
    input_nodes  = [n for n in nodes if in_deg[n] == 0]
    regulators   = [n for n in sorted(nodes)
                    if n not in output_nodes and n not in input_nodes]
    out_node = output_nodes[0]
    coords = {}
    for slot_idx, node_id in enumerate(regulators):
        try:
            dist = nx.shortest_path_length(G, source=node_id, target=out_node)
        except nx.NetworkXNoPath:
            dist = -1
        coords[slot_idx] = {"dist_to_output": dist}
    return coords


def _family_of_slot_label(label: str) -> str:
    return label.split("-")[0]


def _per_topology_fingerprint_pairs(name: str) -> list[tuple[str, str, float]]:
    d      = _load_pairwise(name)
    coords = _topology_slot_coords(name)
    slot_labels = d["slot_labels"]
    pair = np.array(d["pairwise_circuit"])
    n = pair.shape[0]
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            phi = float(pair[i, j])
            fp_i = (f"{_family_of_slot_label(slot_labels[i])}"
                    f"@d{coords[i]['dist_to_output']}")
            fp_j = (f"{_family_of_slot_label(slot_labels[j])}"
                    f"@d{coords[j]['dist_to_output']}")
            a, b = sorted((fp_i, fp_j))
            out.append((a, b, phi))
    return out


def _aggregate_pairs() -> dict:
    rows = []
    for name in TOPOLOGIES:
        for a, b, phi in _per_topology_fingerprint_pairs(name):
            rows.append({"fp_a": a, "fp_b": b, "topology": name, "phi": phi})
    cross = pd.DataFrame(rows)

    # Average within-topology duplicates first.
    cross_within = (
        cross.groupby(["fp_a", "fp_b", "topology"], as_index=False)["phi"].mean()
    )
    per_pair = {}
    for (a, b), g in cross_within.groupby(["fp_a", "fp_b"]):
        topos = g["topology"].tolist()
        phis  = g["phi"].tolist()
        mean_phi = float(np.mean(phis))
        signs = {np.sign(p) for p in phis if abs(p) > 0}
        per_pair[(a, b)] = {
            "topologies": topos, "phis": phis,
            "mean_phi": mean_phi, "n_present": len(topos),
            "sign_consistent": (len(signs) == 1),
        }
    return per_pair


def _compute_membership(per_pair: dict) -> dict[str, set[str]]:
    out = defaultdict(set)
    for (a, b), d in per_pair.items():
        for t in d["topologies"]:
            out[a].add(t)
            out[b].add(t)
    return dict(out)


def _fingerprint_ordering(per_pair: dict,
                          membership: dict[str, set[str]]
                          ) -> tuple[list[str], dict]:
    weight = defaultdict(float)
    for (a, b), d in per_pair.items():
        weight[a] += abs(d["mean_phi"])
        weight[b] += abs(d["mean_phi"])

    primary = {}
    for fp, topos in membership.items():
        if len(topos) == 3:
            primary[fp] = "all"
        elif len(topos) == 2:
            primary[fp] = "+".join(sorted(topos))
        else:
            primary[fp] = next(iter(topos))

    bucket_order = ["all"]
    for k in sorted({v for v in primary.values() if "+" in v}):
        bucket_order.append(k)
    bucket_order.extend(TOPOLOGIES)

    ordered, blocks = [], {}
    for bucket in bucket_order:
        in_bucket = sorted(
            [fp for fp, b in primary.items() if b == bucket],
            key=lambda fp: (-weight[fp], fp),
        )
        if not in_bucket:
            continue
        start = len(ordered)
        ordered.extend(in_bucket)
        blocks[bucket] = (start, len(ordered))
    return ordered, blocks


# --- draw ------------------------------------------------------------------

def build_panel() -> None:
    use_style()

    per_pair   = _aggregate_pairs()
    membership = _compute_membership(per_pair)
    fp_order, blocks = _fingerprint_ordering(per_pair, membership)

    n = len(fp_order)
    idx = {fp: i for i, fp in enumerate(fp_order)}

    M_value      = np.full((n, n), np.nan)
    M_present    = np.zeros((n, n), dtype=int)
    M_consistent = np.zeros((n, n), dtype=bool)
    shared_pairs = []

    for (a, b), d in per_pair.items():
        ia, ib = idx[a], idx[b]
        M_value[ia, ib] = d["mean_phi"]
        M_value[ib, ia] = d["mean_phi"]
        M_present[ia, ib] = d["n_present"]
        M_present[ib, ia] = d["n_present"]
        if d["n_present"] >= 2:
            shared_pairs.append({
                "fp_a": a, "fp_b": b,
                "mean_phi": d["mean_phi"],
                "n_present": d["n_present"],
                "sign_consistent": d["sign_consistent"],
                "topologies": d["topologies"],
                "phis": d["phis"],
            })
        consistent3 = (d["sign_consistent"] and d["n_present"] == ALL_PRESENT)
        M_consistent[ia, ib] = consistent3
        M_consistent[ib, ia] = consistent3

    flat = M_value[np.isfinite(M_value)]
    vmax = float(np.nanmax(np.abs(flat))) if flat.size else 1.0
    vmax = max(vmax, 1e-8)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    # --------------------- figure -----------------------------------------
    fig = plt.figure(figsize=figsize_mm(CANVAS_W_MM, CANVAS_H_MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.4, 1.0], wspace=0.05,
                          left=0.13, right=0.97, top=0.83, bottom=0.18)
    ax = fig.add_subplot(gs[0, 0])
    ax_side = fig.add_subplot(gs[0, 1])

    for i in range(n):
        for j in range(n):
            v = M_value[i, j]
            y = n - 1 - i
            if not np.isfinite(v):
                ax.add_patch(Rectangle((j, y), 1, 1,
                                       fc="#FAFAFA",
                                       ec="#E6E6E6", lw=0.3))
                continue
            color = DIVERGING_CMAP(norm(v))
            ax.add_patch(Rectangle((j, y), 1, 1,
                                   fc=color, ec="white", lw=0.5))
            if abs(v) >= vmax * 0.03:
                tcolor = "white" if abs(norm(v) - 0.5) > 0.36 else "#222"
                ax.text(j + 0.5, y + 0.58, f"{v:+.2f}",
                        ha="center", va="center",
                        fontsize=5.0, fontweight="bold", color=tcolor)
                if M_present[i, j] >= 2:
                    chip_color = (PASTEL["green"]
                                  if M_present[i, j] == ALL_PRESENT
                                  else PASTEL["orange"])
                    ax.text(j + 0.5, y + 0.25,
                            f"{M_present[i, j]} topo",
                            ha="center", va="center",
                            fontsize=4.2, color="#222", fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.08",
                                      fc=chip_color, ec="none", alpha=0.95))
            if M_consistent[i, j]:
                ax.add_patch(Rectangle((j + 0.06, y + 0.06),
                                       0.88, 0.88,
                                       fc="none",
                                       ec=PASTEL["green"], lw=1.4))

    # --------------------- block borders ----------------------------------
    block_colors = {
        "all":  PASTEL["green"],
        "0x2B": TOPOLOGY_COLORS["0x2B"],
        "0x17": TOPOLOGY_COLORS["0x17"],
        "0x6D": TOPOLOGY_COLORS["0x6D"],
    }
    for bucket, (start, end) in blocks.items():
        w = end - start
        if w == 0:
            continue
        color = block_colors.get(bucket, MID_GRAY)
        ax.add_patch(Rectangle((-0.55, n - end), 0.20, w,
                               fc=color, ec="none", alpha=0.85, clip_on=False))
        ax.add_patch(Rectangle((start, n + 0.15), w, 0.20,
                               fc=color, ec="none", alpha=0.85, clip_on=False))
        if bucket == "all":
            txt = "conserved\n(all 3)"
        elif "+" in bucket:
            parts = [TOPOLOGY_DISPLAY[t] for t in bucket.split("+")]
            txt = "shared\n" + "+".join(parts)
        else:
            txt = f"{TOPOLOGY_DISPLAY[bucket]}\nonly"
        ax.text(start + w / 2, n + 0.55, txt,
                ha="center", va="bottom",
                fontsize=5.2, color=color, fontweight="bold",
                clip_on=False)

    # --------------------- axis labels ------------------------------------
    ticks = np.arange(n) + 0.5
    ax.set_xticks(ticks)
    ax.set_xticklabels(fp_order, rotation=45, ha="right",
                       fontsize=5.0)
    ax.set_yticks(ticks)
    ax.set_yticklabels(fp_order[::-1], fontsize=5.0)
    for label in ax.get_xticklabels():
        fam = label.get_text().split("@")[0]
        label.set_color(FAMILY_COLORS.get(fam, TEXT_GREY))
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        fam = label.get_text().split("@")[0]
        label.set_color(FAMILY_COLORS.get(fam, TEXT_GREY))
        label.set_fontweight("bold")
    ax.set_xlim(-0.8, n)
    ax.set_ylim(-0.2, n + 0.9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    # --------------------- colorbar (vertical, right of side panel) -------
    cax = fig.add_axes([0.965, 0.30, 0.010, 0.38])
    import matplotlib.cm as cm
    sm = cm.ScalarMappable(norm=norm, cmap=DIVERGING_CMAP)
    sm.set_array([])
    cb = plt.colorbar(sm, cax=cax)
    cb.set_label("mean Φᵢⱼ on log c",
                 fontsize=5.5, color=TEXT_GREY)
    cb.ax.tick_params(labelsize=5.0, labelcolor=TEXT_GREY,
                      color=DARK_GRAY, width=0.4, length=2.0)
    cb.outline.set_visible(False)

    # --------------------- side panel: shared pairs list ------------------
    ax_side.axis("off")
    ax_side.set_xlim(0, 1)
    ax_side.set_ylim(0, 1)
    ax_side.set_title("Shared across topologies",
                      loc="left", fontsize=6.5, color=TEXT_GREY,
                      fontweight="normal", pad=2)

    shared_sorted = sorted(
        shared_pairs,
        key=lambda r: (-r["n_present"], -abs(r["mean_phi"])),
    )
    conserved3 = [r for r in shared_sorted if r["n_present"] == 3]
    conserved2 = [r for r in shared_sorted if r["n_present"] == 2]

    yc = 0.96
    ax_side.text(0.02, yc,
                 f"✓ conserved in all 3 (sign-consistent): "
                 f"{sum(1 for r in conserved3 if r['sign_consistent'])}",
                 fontsize=5.5, fontweight="bold", color=PASTEL["green"])
    yc -= 0.04
    ax_side.text(0.02, yc,
                 f"  in 3, sign varies: "
                 f"{sum(1 for r in conserved3 if not r['sign_consistent'])}",
                 fontsize=5.2, color=PASTEL["green"])
    yc -= 0.04
    ax_side.text(0.02, yc,
                 f"★ appear in exactly 2 topologies: {len(conserved2)}",
                 fontsize=5.5, fontweight="bold", color=PASTEL["orange"])
    yc -= 0.05

    for row in conserved3 + conserved2:
        yc -= 0.030
        ax_side.text(0.02, yc,
                     f"{row['fp_a']}   ×   {row['fp_b']}",
                     fontsize=5.3, fontweight="bold", color="#222")
        yc -= 0.026
        parts = []
        for t, p in zip(row["topologies"], row["phis"]):
            parts.append(f"{TOPOLOGY_DISPLAY[t]}: {p:+.3f}")
        ax_side.text(0.04, yc, "   ".join(parts),
                     fontsize=4.8, color=TEXT_GREY,
                     style="italic", family="monospace")
        yc -= 0.020
        if row["sign_consistent"]:
            note = "sign-consistent ✓"
            note_c = PASTEL["green"]
        else:
            note = "sign-inconsistent ✗"
            note_c = PASTEL["red"]
        ax_side.text(0.04, yc, note,
                     fontsize=4.8, color=note_c, style="italic")
        if yc < 0.02:
            break

    # --------------------- header + footer --------------------------------
    n_pairs   = sum(1 for d in per_pair.values())
    n_in2     = sum(1 for d in per_pair.values() if d["n_present"] == 2)
    n_in3     = sum(1 for d in per_pair.values()
                    if d["n_present"] == ALL_PRESENT and d["sign_consistent"])
    headline = (
        f"Cross-topology interaction conservation · circuit (m_logic) — "
        f"{n_pairs} unique (family @ d_to_output) pairs across 5/6/7-reg; "
        f"{n_in2} in two topologies; {n_in3} conserved across all three "
        f"with consistent sign. The MLP's pair vocabulary is topology-specific."
    )
    fig.text(0.013, 0.96, headline,
             fontsize=6.5, color=TEXT_GREY, ha="left", va="top")

    fig.text(0.013, 0.04,
             "axis label colour = TF family · fingerprint = (family @ "
             "distance-to-output) · Φᵢⱼ from Shapley-Taylor 2-body decomp "
             "(Module B 04b) on valid-K manifold at each yellow dot.",
             fontsize=4.6, color=MID_GRAY, ha="left", va="bottom",
             style="italic")

    out_base = PANELS_VEC / "panel_d"
    written = save_panel(fig, out_base, dpi=600, close=True)
    print("Panel D written:")
    for fmt, path in written.items():
        print(f"  {fmt}: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    build_panel()
