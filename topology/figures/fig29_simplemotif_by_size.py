"""fig29 — Size-conditioned non-linear motif rankings (combined k=3/4/5).

Companion to fig27 (tier-conditioned) — same three metrics + same
linear-Pₖ filter, but the row axis is topology SIZE (4-reg / 5-reg /
6-reg / 7-reg) instead of design tier. Iso classes from k=3, k=4, AND
k=5 are pooled together; rankings within each size pick the top-6
motifs across the combined pool.

6 figures total: 3 metrics × 2 tasks. All land under output/simplemotif/.

Layouts:
  - partial_r:       8 rows (4 sizes × {HIGHEST +r, LOWEST -r}) × 6 motifs
  - log_odds:        4 rows (sizes) × 6 motifs, ranked by log_odds_top01_vs_bot01_S
  - presence_rate:   4 rows (sizes) × 6 motifs, ranked by presence_rate_top_01

The "_S" tier framing is matched-tail within size class — top_01 / bot_01
designs are defined by per-size-class percentile thresholds (see P23.2 /
data/G3/size_tier_designs/size_tier_thresholds.json).

Each motif in a row shows its own k (3/4/5) — `k=N` annotation in the
corner since different motifs in the same row may have different k.

Note: k=3 motifs tend to dominate presence-rate rankings because smaller
motifs appear in more positions per topology. This is the same dynamic
fig23 surfaced for tier-conditioned analysis. For a k≥4 variant, see
the (future) fig29v2 family if needed.
"""
from __future__ import annotations

import argparse
import math
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import _style as S


REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "topology_g3" / "motif_tier_analysis"

K_VALUES = [3, 4, 5]
# Display default: 5/6/7-reg only (4-reg dropped per 2026-05-20 user direction —
# 4-reg is a thin tail of the substrate (0.14% of designs) and its
# internal_only k=5 is unrenderable. Pass `--sizes 4 5 6 7` to include it.)
SIZE_CLASSES_DEFAULT = [5, 6, 7]
SIZE_CLASSES_ALL     = [4, 5, 6, 7]
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
    4: "#1A9850",  # green
    5: "#5AAE61",  # lighter green
    6: "#F0B90B",  # gold
    7: "#D7263D",  # red (largest = highest burden)
}
TASKS = {"circuit": "circuit", "growth": "toxicity"}
TASK_TITLE = {
    "circuit": "CIRCUIT  (log circuit_score)",
    "growth":  "GROWTH  (toxicity_score)",
}

# Structural DAG colors. All nodes use the same DeepCirc-paper accent blue
# (figures/styles/colors.py ACCENT_BLUE = #18A8E8). Shape-only
# emphasis preserved — orange/source and dark/sink distinctions were
# removed 2026-05-20 to avoid confusion with the fig19/fig20 atlas
# convention (yellow=IN, black=OUT); the all-blue palette continues that
# direction and matches the paper's highlight color for "active data".
NODE_FACE_INNER  = "#18A8E8"
NODE_FACE_SOURCE = NODE_FACE_INNER
NODE_FACE_SINK   = NODE_FACE_INNER
EDGE_COLOR       = "#222"


def iso_key_to_adj(iso_key: str, k: int) -> np.ndarray:
    return np.array([int(b) for b in iso_key], dtype=np.int8).reshape(k, k)


def layered_layout(adj: np.ndarray) -> dict[int, tuple[float, float]]:
    k = adj.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(k))
    for i in range(k):
        for j in range(k):
            if adj[i, j]:
                G.add_edge(i, j)
    sources = [n for n in G.nodes() if G.in_degree(n) == 0]
    depth = {s: 0 for s in sources}
    for n in nx.topological_sort(G):
        for s in G.successors(n):
            depth[s] = max(depth.get(s, 0), depth.get(n, 0) + 1)
    by_depth: dict[int, list[int]] = {}
    for n, d in depth.items():
        by_depth.setdefault(d, []).append(n)
    coords: dict[int, tuple[float, float]] = {}
    for d, ns in by_depth.items():
        ns_sorted = sorted(ns, key=lambda n: (G.in_degree(n), G.out_degree(n), n))
        n_at_d = len(ns_sorted)
        for i, n in enumerate(ns_sorted):
            y = (n_at_d - 1) / 2.0 - i
            coords[n] = (float(d), y)

    # Linear-look fix: when every depth column has exactly one node, the
    # backbone collapses to a single y=0 row and skip edges arc over it.
    # Zigzag the intermediate columns so the chain spreads vertically and
    # skip edges can route as straight diagonals (handled by the obstacle-
    # aware curve check in draw_structural_dag). Source (d=0) and sink
    # (d=max_depth) stay anchored on y=0.
    max_depth = max(depth.values()) if depth else 0
    if max_depth >= 2 and all(
        len(by_depth.get(d, [])) == 1 for d in range(max_depth + 1)
    ):
        has_skip = any(
            adj[i, j] and abs(depth[j] - depth[i]) >= 2
            for i in range(k) for j in range(k)
        )
        if has_skip:
            zigzag_offset = 0.55
            for d in range(1, max_depth):
                n = by_depth[d][0]
                x, _ = coords[n]
                y_new = zigzag_offset if (d % 2 == 1) else -zigzag_offset
                coords[n] = (x, y_new)
    return coords


def _cluster_has_path_overlap(cluster: list[tuple[int, float]],
                                j_coords: tuple[float, float],
                                source_coords_map: dict[int, tuple[float, float]],
                                node_radius: float) -> bool:
    """Return True if any two edges in this incoming cluster would
    visually overlap as straight lines (one source lies on another
    source's straight-line path to the destination).

    Without this check, we'd apply landing-point offsets to clusters
    whose sources happen to be at different y-coordinates — making
    same-height edges look unnecessarily tilted. With this check, we
    only disturb arrow paths when straight lines actually overlap.
    """
    n = len(cluster)
    if n < 2:
        return False
    jx, jy = j_coords
    sources = [(i, source_coords_map[i]) for i, _ in cluster]
    for a in range(n):
        i_a, (ax, ay) = sources[a]
        for b in range(n):
            if a == b:
                continue
            i_b, (bx, by) = sources[b]
            # Does source A lie on the line from source B to destination J?
            dx = jx - bx
            dy = jy - by
            chord_len_sq = dx * dx + dy * dy
            if chord_len_sq < 1e-12:
                continue
            chord_len = math.sqrt(chord_len_sq)
            # Distance from A to line through B-J (perpendicular distance)
            dist = abs(dx * (by - ay) - (bx - ax) * dy) / chord_len
            # Projection parameter t of A onto the B→J chord. t∈(0,1) means
            # A is "between" B and J in the chord direction.
            t = ((ax - bx) * dx + (ay - by) * dy) / chord_len_sq
            if 0.0 < t < 1.0 and dist < node_radius * 1.4:
                return True
    return False


def _cluster_incoming_by_angle(adj: np.ndarray,
                                coords: dict[int, tuple[float, float]],
                                k: int,
                                angle_tol: float = math.pi / 5.5,
                                ) -> dict[int, list[list[tuple[int, float]]]]:
    """For each destination node j, return its incoming edges partitioned
    into clusters of similar angle.

    Returns {j: [cluster_0, cluster_1, ...]} where each cluster is a list
    of (source_idx, incoming_angle_from_j) tuples sorted by angle.
    Clusters of size 1 are still returned (so the caller can no-op them).
    Wraparound handling: the first and last clusters get merged if their
    boundary angles span across ±π and are within the tolerance.
    """
    out: dict[int, list[list[tuple[int, float]]]] = {}
    for j in range(k):
        incoming: list[tuple[int, float]] = []
        xj, yj = coords[j]
        for i in range(k):
            if not adj[i, j] or i == j:
                continue
            xi, yi = coords[i]
            ang = math.atan2(yi - yj, xi - xj)
            incoming.append((i, ang))
        if not incoming:
            out[j] = []
            continue
        incoming.sort(key=lambda x: x[1])
        clusters: list[list[tuple[int, float]]] = []
        current = [incoming[0]]
        for idx in range(1, len(incoming)):
            prev_ang = current[-1][1]
            cur_ang = incoming[idx][1]
            d = cur_ang - prev_ang
            if d > math.pi:
                d -= 2 * math.pi
            elif d < -math.pi:
                d += 2 * math.pi
            if abs(d) < angle_tol:
                current.append(incoming[idx])
            else:
                clusters.append(current)
                current = [incoming[idx]]
        clusters.append(current)
        # Wraparound check
        if len(clusters) > 1:
            last_ang = clusters[-1][-1][1]
            first_ang = clusters[0][0][1]
            d = (first_ang + 2 * math.pi) - last_ang
            if abs(d) < angle_tol:
                clusters[0] = clusters[-1] + clusters[0]
                clusters.pop()
        out[j] = clusters
    return out


def _compute_cluster_curvatures(adj: np.ndarray,
                                 coords: dict[int, tuple[float, float]],
                                 k: int,
                                 node_radius: float = 0.20,
                                 ) -> dict[tuple[int, int], float]:
    """For each destination with multiple incoming edges that WOULD VISUALLY
    OVERLAP as straight lines, assign a curvature `rad` so the arrow PATHS
    fan out symmetrically.

    The visual-overlap gate (see `_cluster_has_path_overlap`) means clusters
    of edges whose sources sit at meaningfully different y-coordinates
    keep straight, naturally-distinct paths — only TRULY colliding paths
    get curved. This avoids the "same-height edges look tilted" artifact.

    Returns {(i, j): rad}. Edges not in an overlap-detected cluster are
    absent from the dict (caller uses 0.0 / obstacle-based logic).
    """
    RAD_MAX = 0.25
    clusters_by_dst = _cluster_incoming_by_angle(adj, coords, k)
    out: dict[tuple[int, int], float] = {}
    for j, clusters in clusters_by_dst.items():
        j_coords = coords[j]
        for cluster in clusters:
            n_in_cluster = len(cluster)
            # Curvature fan-out kicks in only for N≥3 where straight lines
            # from similar source positions would pile up regardless of
            # landing-point spread.
            if n_in_cluster < 3:
                continue
            # AND only if the cluster has actual visual path overlap.
            if not _cluster_has_path_overlap(cluster, j_coords, coords,
                                               node_radius):
                continue
            for idx, (i, _) in enumerate(cluster):
                rad = RAD_MAX - (idx / (n_in_cluster - 1)) * 2 * RAD_MAX
                out[(i, j)] = rad
    return out


def _compute_landing_offsets(adj: np.ndarray,
                              coords: dict[int, tuple[float, float]],
                              k: int,
                              node_radius: float = 0.20,
                              ) -> dict[tuple[int, int], float]:
    """For each destination node, spread incoming-arrow landing angles
    so multiple arrows arriving at the same node don't pile up on the
    same perimeter point — but ONLY when the cluster's sources would
    visually overlap as straight lines.

    Returns {(i, j): delta_angle_radians} — the radians to ADD to the
    natural "from j to i" angle when computing the landing point of
    edge i→j on j's perimeter. Edges in non-overlapping clusters get
    no offset, preserving natural geometry (e.g., perfectly horizontal
    edges between same-height nodes).
    """
    SPREAD = math.pi / 3.0   # ~60° total spread for a cluster

    offsets: dict[tuple[int, int], float] = {}
    clusters_by_dst = _cluster_incoming_by_angle(adj, coords, k)
    for j, clusters in clusters_by_dst.items():
        j_coords = coords[j]
        for cluster in clusters:
            n_in_cluster = len(cluster)
            if n_in_cluster < 2:
                continue
            # Only spread landings if straight-line paths in this cluster
            # would actually overlap. Otherwise leave the natural angle.
            if not _cluster_has_path_overlap(cluster, j_coords, coords,
                                               node_radius):
                continue
            cx = sum(math.cos(a) for _, a in cluster) / n_in_cluster
            cy = sum(math.sin(a) for _, a in cluster) / n_in_cluster
            mean_ang = math.atan2(cy, cx)
            for idx, (i, ang) in enumerate(cluster):
                if n_in_cluster == 2:
                    target = mean_ang + (SPREAD / 2.0) * (1 if idx == 1 else -1)
                else:
                    target = (mean_ang
                              + SPREAD * (idx - (n_in_cluster - 1) / 2.0)
                              / (n_in_cluster - 1))
                delta = target - ang
                while delta > math.pi:
                    delta -= 2 * math.pi
                while delta < -math.pi:
                    delta += 2 * math.pi
                offsets[(i, j)] = delta
    return offsets


def _draw_edge(ax, x1: float, y1: float, x2: float, y2: float,
                rad: float, *,
                node_radius: float = 0.20,
                color: str = "#222",
                lw: float = 1.2,
                head_length: float = 0.16,
                head_width: float = 0.10,
                arrow_t_frac: float = 0.5) -> None:
    """Draw a directed edge between two node centres (x1,y1) → (x2,y2)
    using a quadratic bezier with curvature ``rad`` (matplotlib arc3
    convention: +rad bows to the RIGHT of source→dest). The bezier is
    constructed from CENTRE TO CENTRE, then clipped numerically at each
    node's perimeter (radius ``node_radius``) by binary-searching the
    parameter at which the curve enters/leaves free space.

    Landing direction therefore reflects the actual curved path —
    bowed arrows land on the side of the destination they curve toward,
    not on the side the chord points to. Same on the source side.

    The arrowhead is placed at the midpoint of the VISIBLE arc segment
    (between the perimeter-exit and perimeter-entry parameters),
    oriented along the bezier's local tangent at that midpoint.
    """
    dx = x2 - x1
    dy = y2 - y1
    L = math.hypot(dx, dy)
    if L < 1e-9:
        return

    # arc3 control point: chord_mid + rad * (dy, -dx).
    cx_ctrl = (x1 + x2) / 2 + rad * dy
    cy_ctrl = (y1 + y2) / 2 + rad * (-dx)

    def bezier(t: float) -> tuple[float, float]:
        u = 1.0 - t
        return (u * u * x1 + 2 * u * t * cx_ctrl + t * t * x2,
                u * u * y1 + 2 * u * t * cy_ctrl + t * t * y2)

    def bezier_tan(t: float) -> tuple[float, float]:
        u = 1.0 - t
        return (2 * u * (cx_ctrl - x1) + 2 * t * (x2 - cx_ctrl),
                2 * u * (cy_ctrl - y1) + 2 * t * (y2 - cy_ctrl))

    # Binary-search t_start: smallest t for which the bezier point lies
    # OUTSIDE the source's perimeter. At t=0 we're at the source centre
    # (inside); at t=1 we're at the destination centre (also outside the
    # source's perimeter assuming nodes don't overlap), so the boundary
    # is monotone in t for any reasonable geometry.
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        bx, by = bezier(mid)
        if math.hypot(bx - x1, by - y1) < node_radius:
            lo = mid
        else:
            hi = mid
    t_start = hi

    # Binary-search t_end: largest t for which the bezier point lies
    # OUTSIDE the destination's perimeter.
    lo, hi = t_start, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        bx, by = bezier(mid)
        if math.hypot(bx - x2, by - y2) < node_radius:
            hi = mid
        else:
            lo = mid
    t_end = lo

    if t_end <= t_start + 1e-3:
        return  # Nodes effectively overlap; nothing visible to draw.

    # Polyline approximation of the visible bezier sub-segment.
    n_samples = 32
    xs = [0.0] * n_samples
    ys = [0.0] * n_samples
    for i in range(n_samples):
        t = t_start + (t_end - t_start) * (i / (n_samples - 1))
        xs[i], ys[i] = bezier(t)
    ax.plot(xs, ys, color=color, lw=lw,
              zorder=2, solid_capstyle="round")

    # Arrowhead at ``arrow_t_frac`` along the visible arc (default 0.5 =
    # geometric midpoint, original behaviour). The caller can pass a
    # different fraction (e.g., 0.3 for "20% upstream of midpoint") to
    # separate arrowheads on edges that participate in an X-pattern
    # crossing — at the geometric midpoint, two crossing arrows would
    # overlap; shifting toward the source separates them along their
    # respective tangents. Detection happens in draw_edges_on_dag.
    # NOTE: arrow_t_frac is relative to the VISIBLE segment (between
    # perimeter exit/entry), not the full centre-to-centre bezier.
    t_mid = t_start + arrow_t_frac * (t_end - t_start)
    mx, my = bezier(t_mid)
    tx, ty = bezier_tan(t_mid)
    tan_len = math.hypot(tx, ty)
    if tan_len < 1e-9:
        return
    tx /= tan_len
    ty /= tan_len
    tip_x = mx + tx * head_length * 0.5
    tip_y = my + ty * head_length * 0.5
    base_x = mx - tx * head_length * 0.5
    base_y = my - ty * head_length * 0.5
    perp_x = -ty
    perp_y = tx
    base_left = (base_x + perp_x * head_width, base_y + perp_y * head_width)
    base_right = (base_x - perp_x * head_width, base_y - perp_y * head_width)
    triangle = mpatches.Polygon(
        [(tip_x, tip_y), base_left, base_right],
        closed=True,
        facecolor=color, edgecolor=color, linewidth=0.4,
        zorder=3,
    )
    ax.add_patch(triangle)


def draw_edges_on_dag(ax, adj: np.ndarray,
                       coords: dict[int, tuple[float, float]],
                       k: int, *,
                       node_radius: float = 0.20,
                       edge_color: str = EDGE_COLOR,
                       lw: float = 1.2,
                       head_length_frac: float = 0.765,
                       head_width_frac: float = 0.45) -> None:
    """Draw the bezier-aware edges of a DAG. Factored out of
    draw_structural_dag so other renderers (fig30's typed expansion,
    etc.) can reuse the exact same connectivity logic without having to
    also reuse the node styling. Bidirectional pairs split, N≥3
    clusters fan out, and obstacle-aware skip edges bow away from
    intermediate nodes; _draw_edge then clips each bezier at the actual
    node perimeters so landing direction matches the curve's approach.
    """
    obstacle_clearance = node_radius * 1.4
    bidir = set()
    for i in range(k):
        for j in range(k):
            if adj[i, j] and adj[j, i] and i != j:
                bidir.add((i, j))
    incoming_cluster_curv = _compute_cluster_curvatures(adj, coords, k, node_radius)

    # X-pattern detection: collect edges (i,j) whose straight-line
    # segment crosses another edge's straight-line segment (no shared
    # endpoint, no bidirectional pair). Edges in this set get their
    # arrowhead shifted 20% upstream of midpoint so the two arrows on
    # an X don't overlap at the crossing point.
    edges = [(i, j) for i in range(k) for j in range(k)
              if adj[i, j] and (i, j) not in bidir]

    def _segments_cross(p1, p2, p3, p4):
        # Proper segment intersection excluding shared endpoints. Uses
        # the orientation (ccw) sign test — standard textbook algorithm.
        def _ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) - (B[1] - A[1]) * (C[0] - A[0])
        d1 = _ccw(p3, p4, p1)
        d2 = _ccw(p3, p4, p2)
        d3 = _ccw(p1, p2, p3)
        d4 = _ccw(p1, p2, p4)
        return ((d1 > 0 > d2 or d1 < 0 < d2)
                and (d3 > 0 > d4 or d3 < 0 < d4))

    crossing_edges: set[tuple[int, int]] = set()
    for a_idx, (i_a, j_a) in enumerate(edges):
        p1, p2 = coords[i_a], coords[j_a]
        for (i_b, j_b) in edges[a_idx + 1:]:
            if i_a in (i_b, j_b) or j_a in (i_b, j_b):
                continue  # Shared endpoint — not an X
            p3, p4 = coords[i_b], coords[j_b]
            if _segments_cross(p1, p2, p3, p4):
                crossing_edges.add((i_a, j_a))
                crossing_edges.add((i_b, j_b))

    for i in range(k):
        for j in range(k):
            if not adj[i, j]:
                continue
            x1, y1 = coords[i]
            x2, y2 = coords[j]
            dx, dy = x2 - x1, y2 - y1
            L = float(np.hypot(dx, dy))
            if L < 1e-6:
                continue

            rad = 0.0
            if (i, j) in bidir:
                rad = 0.18 if i < j else -0.18
            else:
                cluster_rad = incoming_cluster_curv.get((i, j))
                if cluster_rad is not None and abs(cluster_rad) > 1e-6:
                    rad = cluster_rad
                else:
                    best_rad_mag = 0.0
                    best_bow_sign = 0.0
                    for n in range(k):
                        if n == i or n == j:
                            continue
                        xn, yn = coords[n]
                        if not (min(x1, x2) - 0.05 < xn < max(x1, x2) + 0.05):
                            continue
                        if abs(x2 - x1) > 1e-9:
                            t = (xn - x1) / (x2 - x1)
                        else:
                            t = 0.0
                        y_line = y1 + t * (y2 - y1)
                        gap = abs(yn - y_line)
                        if gap < obstacle_clearance:
                            clearance_rad_mag = max(
                                0.0,
                                2.0 * (2.0 * node_radius - gap) / L,
                            )
                            base_rad_mag = 0.15 + 0.03 * abs(dx)
                            rad_mag = max(clearance_rad_mag, base_rad_mag)
                            cross = (x2 - x1) * (yn - y1) - (y2 - y1) * (xn - x1)
                            bow_sign = 1.0 if cross > 0 else -1.0
                            if rad_mag > best_rad_mag:
                                best_rad_mag = rad_mag
                                best_bow_sign = bow_sign
                    rad = best_bow_sign * best_rad_mag

            arrow_t_frac = 0.3 if (i, j) in crossing_edges else 0.5
            _draw_edge(ax, x1, y1, x2, y2, rad,
                        node_radius=node_radius,
                        color=edge_color, lw=lw,
                        head_length=node_radius * head_length_frac,
                        head_width=node_radius * head_width_frac,
                        arrow_t_frac=arrow_t_frac)


def draw_structural_dag(ax, iso_key: str, k: int, *,
                         node_radius: float = 0.20) -> None:
    adj = iso_key_to_adj(iso_key, k)
    coords = layered_layout(adj)
    xs = [c[0] for c in coords.values()]
    ys = [c[1] for c in coords.values()]
    pad_x, pad_y = 0.65, 0.7
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    ax.set_aspect("equal")
    ax.axis("off")

    draw_edges_on_dag(ax, adj, coords, k, node_radius=node_radius)

    in_deg = adj.sum(axis=0)
    out_deg = adj.sum(axis=1)
    for n in range(k):
        x, y = coords[n]
        if in_deg[n] == 0:
            face = NODE_FACE_SOURCE
        elif out_deg[n] == 0:
            face = NODE_FACE_SINK
        else:
            face = NODE_FACE_INNER
        ax.add_patch(plt.Circle((x, y), node_radius,
                                  facecolor=face, edgecolor="#222",
                                  linewidth=0.6, zorder=3))


def load_combined_rankings(task: str, motif_view: str = "full") -> pd.DataFrame:
    """Load + concatenate per-k iso rankings into one combined-k pool.

    ``motif_view`` picks the on-disk CSV suffix (matches P26's
    --motif_view output convention). When a per-k CSV is missing for the
    requested view (e.g., k=5 doesn't exist in internal_only mode if
    upstream P13/P26 didn't produce one), the function skips it with a
    warning rather than erroring.
    """
    suffix = VIEW_SUFFIX[motif_view]
    frames = []
    for k in K_VALUES:
        path = DATA / f"structural_iso_by_size_{task}_k{k}{suffix}.csv"
        if not path.exists():
            print(f"  [warn] missing {path.name}; skipping k={k} for "
                  f"view={motif_view}", file=sys.stderr)
            continue
        df = pd.read_csv(path, dtype={"iso_key": str})
        df["iso_key"] = df["iso_key"].str.zfill(k * k)
        # `k` is already a column from P26.
        frames.append(df)
    if not frames:
        raise SystemExit(
            f"no structural_iso_by_size_{task}_k*{suffix}.csv files found "
            f"under {DATA}; run P26 with --motif_view {motif_view} first"
        )
    return pd.concat(frames, ignore_index=True)


def top_n_by(df: pd.DataFrame, col: str, *, ascending: bool = False,
              n: int = 6, exclude_linear: bool = True) -> pd.DataFrame:
    sub = df[~df["is_linear"]].copy() if exclude_linear else df.copy()
    sub = sub.dropna(subset=[col])
    sub = sub.sort_values(col, ascending=ascending).head(n)
    return sub.reset_index(drop=True)


def render_row_cell(ax_grid, row_motifs: pd.DataFrame, *,
                     metric_col: str, value_fmt: str,
                     row_band_color: str | None,
                     tier_pair: str = "01",
                     show_k_badge: bool = True,
                     volume_direction: str = "both") -> None:
    """Render top-N iso classes in one row. Each motif's k may differ —
    DAG sized to its own k, with `k=N` corner annotation (suppressed when
    show_k_badge=False; downstream figures like fig34 use that for a
    cleaner combo layout).

    ``volume_direction`` controls which presence_designs count is shown
    in the per-cell "volume = …" line:
      - "both" (default, fig29 / fig30 / fig33 convention):
          v_top + v_bot — matched-tail combined design count.
      - "top": v_top only (presence_designs_top_<tier_pair>) — the
          numerator of presence_rate_top, useful when a row is sorted by
          that rate or by top-enriched log_odds.
      - "bot": v_bot only — symmetric for failure-direction rows.
    """
    n = len(row_motifs)
    if n == 0:
        ax_grid.text(0.5, 0.5, "(no non-linear motif passes filter)",
                       ha="center", va="center", transform=ax_grid.transAxes,
                       fontsize=8, color="#888", style="italic")
        ax_grid.axis("off")
        return
    ax_grid.axis("off")
    inner = ax_grid.inset_axes([0.02, 0.05, 0.96, 0.95])
    inner.axis("off")
    sub_axes = []
    for i in range(n):
        sub = inner.inset_axes([i / n, 0.0, 1.0 / n, 1.0])
        sub_axes.append(sub)
    for ax_sub, (_, row) in zip(sub_axes, row_motifs.iterrows()):
        iso_key = row["iso_key"]
        k = int(row["k"])
        ax_sub.set_xlim(0, 1)
        ax_sub.set_ylim(0, 1)
        dag = ax_sub.inset_axes([0.05, 0.35, 0.90, 0.60])
        draw_structural_dag(dag, iso_key, k, node_radius=0.20)
        # k-value annotation in the upper-left corner of the cell
        if show_k_badge:
            ax_sub.text(0.02, 0.97, f"k={k}",
                         ha="left", va="top",
                         fontsize=7.5, fontweight="bold",
                         color="#444",
                         bbox=dict(boxstyle="round,pad=0.18",
                                     facecolor="#f7f7f7",
                                     edgecolor="#bbb", linewidth=0.4))
        val = row[metric_col]
        ax_sub.text(0.5, 0.26, value_fmt.format(val),
                     ha="center", va="center",
                     fontsize=9.5, fontweight="bold",
                     color="#111")
        n_topo_motif = int(row.get("n_topo_with_motif_in_size", 0))
        n_topo_size  = int(row.get("n_topo_in_size", 0))
        ax_sub.text(0.5, 0.15,
                     f"n_topo = {n_topo_motif} / {n_topo_size}",
                     ha="center", va="center",
                     fontsize=7.5, color="#111")
        # Volume = number of designs in the matched-tail pool at the
        # chosen tier_pair that contain this motif. Default convention
        # (fig30 / fig33) is bidirectional (top + bot); fig34 overrides
        # to a single direction so the displayed volume matches the
        # numerator of the cell's ranking metric.
        top_col = f"presence_designs_top_{tier_pair}"
        bot_col = f"presence_designs_bot_{tier_pair}"
        v_top = int(row.get(top_col, 0) or 0)
        v_bot = int(row.get(bot_col, 0) or 0)
        tier_pct_short = {"01": "1", "05": "5",
                            "10": "10", "25": "25"}.get(tier_pair, tier_pair)
        if volume_direction == "top":
            volume_value = v_top
            volume_label = f"top-{tier_pct_short}% volume"
        elif volume_direction == "bot":
            volume_value = v_bot
            volume_label = f"bot-{tier_pct_short}% volume"
        else:
            volume_value = v_top + v_bot
            volume_label = "volume"
        ax_sub.text(0.5, 0.05,
                     f"{volume_label} = {volume_value}",
                     ha="center", va="center",
                     fontsize=7.5, color="#111")
        ax_sub.axis("off")


# ---------------------------------------------------------------------------
# Footer text wrapping
# ---------------------------------------------------------------------------

# Cells span left=0.07 to right=0.98 = 91% of a 17-inch canvas. At 9pt
# italic the comfortable per-line character budget is ~165 chars wide
# (verified empirically — anything beyond starts to overflow visibly).
FOOTER_LINE_WIDTH = 165


def _wrap_footer(text: str, width: int = FOOTER_LINE_WIDTH) -> str:
    """Wrap a multi-clause footer string so each line stays inside the
    cell table's horizontal extent. Clauses are joined with the
    middle-dot separator '·' in the calling code and split cleanly on
    word boundaries via textwrap.fill().
    """
    return textwrap.fill(
        text,
        width=width,
        break_long_words=False,
        replace_whitespace=True,
        drop_whitespace=True,
    )


# ---------------------------------------------------------------------------
# Size-group separators
# ---------------------------------------------------------------------------

def _draw_size_separators(fig, gs, n_sizes: int) -> None:
    """Draw a horizontal separator line in the gap below each size group's
    last row (the LOWEST row). With ``rows_per_size = 2`` (HIGH + LOW
    per size) and 3 sizes (5/6/7-reg), this yields exactly 2 lines.
    """
    n_rows = gs.nrows
    if n_sizes < 2 or n_rows % n_sizes != 0:
        return
    rows_per_size = n_rows // n_sizes
    bottoms, tops, _, _ = gs.get_grid_positions(fig)
    for i in range(1, n_sizes):
        last_above  = i * rows_per_size - 1   # last row of size i-1 (the LOW row)
        first_below = i * rows_per_size       # first row of size i (the HIGH row)
        y = (bottoms[last_above] + tops[first_below]) / 2
        line = Line2D([0.015, 0.98], [y, y],
                       transform=fig.transFigure,
                       color="#888", linewidth=0.8, linestyle="-")
        fig.add_artist(line)


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def build_partial_r_figure(task_disp: str, task_disk: str,
                            motif_view: str = "full",
                            sizes: list[int] | None = None,
                            min_n_topo: int = 1) -> None:
    """N rows (sizes × {HIGHEST, LOWEST}) × 1 col, combined-k pool."""
    sizes = list(sizes) if sizes else SIZE_CLASSES
    n_rows = len(sizes) * 2
    df = load_combined_rankings(task_disk, motif_view=motif_view)
    if min_n_topo > 1:
        df = df[df["n_topo_with_motif_in_size"] >= min_n_topo]
    # Canvas height scales with row count so each row keeps its breathing room.
    fig_h = 2.1 * n_rows + 0.6
    fig = plt.figure(figsize=(17.0, fig_h))
    gs = fig.add_gridspec(n_rows, 1, hspace=0.32,
                           left=0.07, right=0.98, top=0.95, bottom=0.04)
    row_idx = 0
    for s in sizes:
        df_s = df[df["size_class"] == s]
        color = SIZE_COLOR[s]
        for direction_label, ascending in (("HIGHEST  +r", False),
                                              ("LOWEST  -r", True)):
            top = top_n_by(df_s, "partial_r", ascending=ascending, n=6)
            ax = fig.add_subplot(gs[row_idx, 0])
            render_row_cell(ax, top, metric_col="partial_r",
                              value_fmt="r = {:+.3f}",
                              row_band_color=color,
                              tier_pair="01")
            label = f"{SIZE_LABEL[s]}\n{direction_label}"
            fig.text(0.015, 0.95 - (row_idx + 0.5) * (0.91 / n_rows),
                      label, ha="left", va="center", rotation=90,
                      fontsize=10, fontweight="bold", color="#111")
            row_idx += 1

    _draw_size_separators(fig, gs, len(sizes))
    S.figure_title(fig,
        f"Simplemotif by SIZE  ·  {TASK_TITLE[task_disp]}  ·  "
        f"k = 3 + 4 + 5 (combined)  ·  metric: within-size partial r  "
        f"(linear chains excluded){VIEW_TITLE_TAG[motif_view]}")
    footer_parts = [
        "Within each size class, top-6 non-linear iso-canonical motifs "
        "across the combined k=3+4+5 pool ranked by partial r of "
        "motif-count-per-topology with score, controlling for num_edges "
        "(over topologies in that size class only). Each motif's k shown "
        "in corner."
    ]
    if motif_view == "internal_only":
        footer_parts.append(
            "Internal-only view: IN sensors and OUT-* terminal gates "
            "excluded; only internal NOT + NOR participate. 4-reg has 4 "
            "internals (k=5 unrenderable); 5-/6-/7-reg have 5/6/7 (all k "
            "renderable). Sparseness = convergence, not candidate-node shortage."
        )
    elif motif_view == "no_in":
        footer_parts.append(
            "No-IN view: sensor inputs excluded; OUT-* gates retained.")
    elif motif_view == "no_out":
        footer_parts.append(
            "No-OUT view: OUT-* terminal gates excluded; IN sensors "
            "retained (symmetric counterpart to no_in).")
    footer = _wrap_footer("  ·  ".join(footer_parts))
    fig.text(0.5, 0.01, footer,
              ha="center", va="bottom", fontsize=9,
              color="#555", style="italic")
    view_tag = VIEW_SUFFIX[motif_view]
    S.save_figure(fig, f"fig29_size_partial_r_{task_disp}{view_tag}",
                    group="simplemotif")
    plt.close(fig)


def build_tier_figure(task_disp: str, task_disk: str, metric: str,
                       motif_view: str = "full",
                       sizes: list[int] | None = None,
                       min_n_topo: int = 1,
                       tier_pair: str = "01") -> None:
    """N rows (sizes × {HIGHEST, LOWEST}) × 1 col, combined-k pool.

    metric ∈ {"log_odds", "presence_rate"}

    For each size, emit two rows:
      - log_odds:      HIGHEST +log₂ (most elite-enriched) /
                       LOWEST  -log₂ (most failure-enriched)
      - presence_rate: TOP-1%  PRESENCE (most common in elite tier) /
                       BOT-1%  PRESENCE (most common in failure tier)

    Same size color is used for both rows in a pair — color = size identity.
    """
    sizes = list(sizes) if sizes else SIZE_CLASSES
    n_rows = len(sizes) * 2
    df = load_combined_rankings(task_disk, motif_view=motif_view)
    if min_n_topo > 1:
        df = df[df["n_topo_with_motif_in_size"] >= min_n_topo]
    fig_h = 2.1 * n_rows + 0.6
    fig = plt.figure(figsize=(17.0, fig_h))
    gs = fig.add_gridspec(n_rows, 1, hspace=0.32,
                           left=0.07, right=0.98, top=0.95, bottom=0.04)

    if tier_pair not in ("01", "05", "10", "25"):
        raise ValueError(f"tier_pair must be '01'|'05'|'10'|'25'; got {tier_pair!r}")
    tier_pct_label = {"01": "1%", "05": "5%",
                       "10": "10%", "25": "25%"}[tier_pair]
    if metric == "log_odds":
        col_log = f"log_odds_top{tier_pair}_vs_bot{tier_pair}"
        # log_odds is inherently bidirectional (top/bot ratio) → keep the
        # combined volume display so readers see both tails' contribution.
        direction_specs = (
            (f"HIGHEST  +log₂", col_log, False, "both"),
            (f"LOWEST  -log₂",  col_log, True,  "both"),
        )
        vf = "log₂ = {:+.2f}"
    elif metric == "presence_rate":
        col_top = f"presence_rate_top_{tier_pair}"
        col_bot = f"presence_rate_bot_{tier_pair}"
        # presence_rate rows are direction-specific → volume should match
        # the numerator (top-only for TOP-PRESENCE rows; bot-only for
        # BOT-PRESENCE rows). Same convention as fig34.
        direction_specs = (
            (f"TOP-{tier_pct_label}  PRESENCE", col_top, False, "top"),
            (f"BOT-{tier_pct_label}  PRESENCE", col_bot, False, "bot"),
        )
        vf = "{:.1%}"
    else:
        raise ValueError(metric)

    row_idx = 0
    for s in sizes:
        df_s = df[df["size_class"] == s]
        color = SIZE_COLOR[s]
        for direction_label, col, ascending, vol_dir in direction_specs:
            top = top_n_by(df_s, col, ascending=ascending, n=6)
            ax = fig.add_subplot(gs[row_idx, 0])
            render_row_cell(ax, top, metric_col=col, value_fmt=vf,
                              row_band_color=color,
                              tier_pair=tier_pair,
                              volume_direction=vol_dir)
            label = f"{SIZE_LABEL[s]}\n{direction_label}"
            fig.text(0.015, 0.95 - (row_idx + 0.5) * (0.91 / n_rows),
                      label, ha="left", va="center", rotation=90,
                      fontsize=10, fontweight="bold", color="#111")
            row_idx += 1

    view_tag_title = VIEW_TITLE_TAG[motif_view]
    tier_title_tag = f"  ·  TIER = top-{tier_pct_label} vs bot-{tier_pct_label}"
    if metric == "log_odds":
        title = (f"Simplemotif by SIZE  ·  {TASK_TITLE[task_disp]}  ·  "
                  f"k = 3 + 4 + 5 (combined)  ·  "
                  "metric: within-size log₂ matched-tail enrichment  "
                  f"(linear chains excluded){view_tag_title}{tier_title_tag}")
        footer_parts = [
            f"Within each size class, two rows: HIGHEST +log₂ "
            f"(top-{tier_pct_label}_S over bot-{tier_pct_label}_S, "
            f"elite-enriched) and LOWEST −log₂ (failure-enriched). Top-6 "
            "non-linear iso-canonical motifs per row across the combined "
            "k=3+4+5 pool. Size-stratified tiers from P23.2.",
        ]
    else:
        title = (f"Simplemotif by SIZE  ·  {TASK_TITLE[task_disp]}  ·  "
                  f"k = 3 + 4 + 5 (combined)  ·  "
                  "metric: within-size tier presence rate  "
                  f"(linear chains excluded){view_tag_title}{tier_title_tag}")
        footer_parts = [
            f"Within each size class, two rows: TOP-{tier_pct_label} "
            f"PRESENCE (common in elite tier) and BOT-{tier_pct_label} "
            f"PRESENCE (common in failure tier). Top-6 non-linear "
            "iso-canonical motifs per row across the combined k=3+4+5 "
            "pool. presence_rate measures ubiquity not discrimination — "
            "see log_odds for tier-specific shapes.",
        ]
    if motif_view == "internal_only":
        footer_parts.append(
            "Internal-only view: IN sensors and OUT-* terminal gates "
            "excluded; only internal NOT + NOR participate. 4-reg has 4 "
            "internals (k=5 unrenderable); 5-/6-/7-reg have 5/6/7 (all k "
            "renderable). Sparseness = convergence, not candidate-node shortage.")
    elif motif_view == "no_in":
        footer_parts.append(
            "No-IN view: sensor inputs excluded; OUT-* gates retained.")
    elif motif_view == "no_out":
        footer_parts.append(
            "No-OUT view: OUT-* terminal gates excluded; IN sensors "
            "retained (symmetric counterpart to no_in).")
    _draw_size_separators(fig, gs, len(sizes))
    footer = _wrap_footer("  ·  ".join(footer_parts))
    S.figure_title(fig, title)
    fig.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=9,
              color="#555", style="italic")
    view_tag = VIEW_SUFFIX[motif_view]
    # Append tier tag to filename only when non-default (tier_pair != "01")
    # so existing fig29_size_*_*_<view>.pdf files stay byte-stable for the
    # legacy default.
    tier_tag = "" if tier_pair == "01" else f"_top{tier_pair}_bot{tier_pair}"
    S.save_figure(fig,
                    f"fig29_size_{metric}_{task_disp}{view_tag}{tier_tag}",
                    group="simplemotif")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metric", default="all",
                     choices=["all", "partial_r", "log_odds", "presence_rate"])
    ap.add_argument("--task", default="all",
                     choices=["all", "circuit", "growth"])
    ap.add_argument("--motif_view",
                     choices=list(VIEW_SUFFIX.keys()), default="full",
                     help="Motif universe to read (mirrors P13/P24/P26). "
                          "'full' (default), 'no_in', 'no_out', or "
                          "'internal_only'. Outputs gain a matching filename "
                          "suffix; the title carries the view tag.")
    ap.add_argument("--sizes", type=int, nargs="+",
                     default=SIZE_CLASSES_DEFAULT,
                     help="Size classes to plot (regulator counts). "
                          "Default = 5 6 7 (4-reg dropped 2026-05-20 per "
                          "user direction). Pass `--sizes 4 5 6 7` to include "
                          "4-reg back in.")
    ap.add_argument("--min_n_topo", type=int, default=1,
                     help="Display filter: drop iso classes with "
                          "n_topo_with_motif_in_size < this value. Default 1 "
                          "(no filter — show every iso class with finite "
                          "log_odds). Higher values reproduce the historical "
                          "P13 min_motif_topologies=5 behaviour at iso level.")
    ap.add_argument("--tier_pair", choices=["01", "05", "10", "25"],
                     default="01",
                     help="Which matched-tail tier pair to use for log_odds "
                          "and presence_rate. '01' (default) = top-1%% vs "
                          "bot-1%%. '05' = top-5%% vs bot-5%%. '10' = "
                          "top-10%% vs bot-10%%. '25' = top-25%% vs "
                          "bot-25%%. Wider tails intersect more narrow-but-"
                          "enriched iso classes. Adds '_top<TP>_bot<TP>' to "
                          "output filenames when non-default.")
    args = ap.parse_args()
    S.setup_matplotlib()

    # Promote --sizes into the module-level SIZE_CLASSES so render_row_cell etc.
    # see the same list.
    global SIZE_CLASSES
    SIZE_CLASSES = list(args.sizes)

    tasks = list(TASKS) if args.task == "all" else [args.task]
    metrics = (["partial_r", "log_odds", "presence_rate"]
                 if args.metric == "all" else [args.metric])

    tier_tag = "" if args.tier_pair == "01" else f"_top{args.tier_pair}_bot{args.tier_pair}"
    for task_disp in tasks:
        task_disk = TASKS[task_disp]
        for metric in metrics:
            if metric == "partial_r":
                # partial_r doesn't depend on tier — same output regardless.
                if args.tier_pair != "01":
                    continue
                build_partial_r_figure(task_disp, task_disk,
                                        motif_view=args.motif_view,
                                        sizes=args.sizes,
                                        min_n_topo=args.min_n_topo)
            else:
                build_tier_figure(task_disp, task_disk, metric,
                                   motif_view=args.motif_view,
                                   sizes=args.sizes,
                                   min_n_topo=args.min_n_topo,
                                   tier_pair=args.tier_pair)
            view_tag = VIEW_SUFFIX[args.motif_view]
            print(f"  rendered fig29_size_{metric}_{task_disp}{view_tag}{tier_tag}")


if __name__ == "__main__":
    main()
