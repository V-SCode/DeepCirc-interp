"""Validated circuit-presence enrichment for the current publication panels.

Historical count enrichment remains in the frozen G3 and setFinal inputs.
Each current dot is conditional on its structural context being present;
the denominator counts evaluated circuits, not matching positions.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "l2_presence_enrichment_v2.csv"
VALUE_COLUMN = "log2_presence_enrichment"


def load_presence_data(task: str) -> pd.DataFrame:
    """Load the complete, validated 65-context publication subset for one task."""
    if task not in {"toxicity", "circuit_log"}:
        raise ValueError(f"Unknown enrichment task: {task}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Build circuit-presence inputs first with "
            "setFinalvF/scripts/build_presence_enrichment.py"
        )
    data = pd.read_csv(DATA_PATH)
    required = {
        "task", "fp_key", "part_name", "part_idx", "n_topologies", "n_targets",
        "part_tier_support", "top_design_count", "bottom_design_count",
        "top_part_design_count", "bottom_part_design_count", "top_probability",
        "bottom_probability", VALUE_COLUMN,
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing presence-enrichment fields: {sorted(missing)}")
    data = data.loc[data["task"] == task].copy()
    if data.duplicated(["fp_key", "part_idx"]).any():
        raise ValueError("Duplicate part/context observations")
    for side in ("top", "bottom"):
        numerator = data[f"{side}_part_design_count"].to_numpy(dtype=float)
        denominator = data[f"{side}_design_count"].to_numpy(dtype=float)
        probability = data[f"{side}_probability"].to_numpy(dtype=float)
        if not (np.isfinite(numerator).all() and np.isfinite(denominator).all()):
            raise ValueError("Nonfinite circuit counts")
        if np.any(numerator < 0) or np.any(denominator <= 0) or np.any(numerator > denominator):
            raise ValueError("Invalid circuit-presence counts")
        if not (np.equal(numerator, np.floor(numerator)).all()
                and np.equal(denominator, np.floor(denominator)).all()):
            raise ValueError("Circuit counts must be integers")
        if not np.allclose(probability, (numerator + 1) / (denominator + 2),
                           rtol=1e-12, atol=1e-14):
            raise ValueError("Probabilities do not match circuit counts and binary add-one smoothing")
    expected = np.log2(data["top_probability"] / data["bottom_probability"])
    if not np.allclose(data[VALUE_COLUMN], expected, rtol=1e-12, atol=1e-12):
        raise ValueError("Enrichment does not match the normalized probability ratio")
    eligible = data.loc[
        (data["n_topologies"] >= 5)
        & (data["n_targets"] >= 3)
        & (data["part_tier_support"] >= 1000)
    ].copy()
    if (len(eligible) != 1300 or eligible["fp_key"].nunique() != 65
            or eligible["part_idx"].nunique() != 20
            or not eligible.groupby("fp_key")["part_idx"].nunique().eq(20).all()):
        raise ValueError("Publication inputs must contain 65 eligible contexts for all 20 parts")
    return eligible
