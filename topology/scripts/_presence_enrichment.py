"""Circuit-presence enrichment with explicit exposure denominators.

Each observation is one evaluated design row in a tier whose topology contains
the structural context. A part is either present or absent at that context;
multiple matching positions do not create additional observations.
"""
from __future__ import annotations

import numpy as np


PRESENCE_SCHEMA = "deepcirc.part_presence_enrichment.v2"
PRESENCE_METADATA = {
    "schema": PRESENCE_SCHEMA,
    "observation_unit": "evaluated_design_row_with_context",
    "numerator": "design_rows_with_part_at_any_position_matching_context",
    "denominator": "design_rows_whose_topology_contains_context",
    "smoothing": "binary_add_one",
    "probability_formula": "(part_design_count + 1) / (design_count + 2)",
    "enrichment_formula": "log2(top_probability / bottom_probability)",
    "tier_policy": "inclusive_quantile_thresholds",
    "quantile_method": "linear",
    "top_comparison": ">=",
    "bottom_comparison": "<=",
}


def presence_statistics(top_part, bottom_part, top_design, bottom_design) -> dict:
    """Return smoothed probabilities and their log2 ratio.

    Inputs may be scalars or broadcastable NumPy arrays of integer-valued
    counts. Denominators count evaluated circuits, not matching positions or
    occurrences summed across library parts. The two binary outcomes receive
    one pseudocount each, giving (n + 1) / (N + 2).

    Validation rejects missing exposure, nonfinite/fractional/negative counts,
    and a presence numerator greater than its denominator. Counts exceeding
    exact float64 integer precision are also rejected rather than rounded.
    """
    names = ("top_part", "bottom_part", "top_design", "bottom_design")
    values = (top_part, bottom_part, top_design, bottom_design)
    arrays = []
    for name, value in zip(names, values):
        original = np.asarray(value)
        if original.dtype.kind not in "iuf":
            raise ValueError(f"{name} must contain numeric integer counts")
        array = np.asarray(value, dtype=np.float64)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain finite counts")
        if np.any(array < 0) or np.any(array != np.floor(array)):
            raise ValueError(f"{name} must contain nonnegative integer counts")
        if np.any(array > 2**53 - 2):
            raise ValueError(f"{name} exceeds exact float64 count precision")
        arrays.append(array)
    try:
        top_part, bottom_part, top_design, bottom_design = np.broadcast_arrays(*arrays)
    except ValueError as exc:
        raise ValueError("presence counts must have broadcastable shapes") from exc
    if np.any(top_design <= 0) or np.any(bottom_design <= 0):
        raise ValueError("both tiers must have positive design exposure")
    if np.any(top_part > top_design) or np.any(bottom_part > bottom_design):
        raise ValueError("part-presence counts cannot exceed design exposure")
    top_probability = (top_part + 1.0) / (top_design + 2.0)
    bottom_probability = (bottom_part + 1.0) / (bottom_design + 2.0)
    return {
        "top_probability": top_probability,
        "bottom_probability": bottom_probability,
        "log2_presence_enrichment": np.log2(top_probability / bottom_probability),
    }
