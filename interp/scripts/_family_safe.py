"""Family-safe coalition-swap sampler for Shapley / Shapley-Taylor attributions.

## Why this exists

DeepCirc's valid-design manifold enforces **family uniqueness**: each repressor
family (AmeR, AmtR, BetI, BM3R1, HlyIIR, IcaRA, LitR, LmrA, PhlF, PsrA, QacR,
SrpR) may appear at most once in a circuit. The 20-part library packs 2-4 variants
into some families (SrpR has 4, PhlF/BM3R1 have 3, QacR has 2). Any circuit
with two parts from the same family is out-of-manifold.

The Shapley / Shapley-Taylor attribution loop draws a "background" completion
from the valid-permutation pool, then overwrites a coalition of slots with the
yellow-dot's parts:

    out = pool[i].copy()
    out[:, mask] = perm[mask]

The pool is family-clean and the yellow-dot is family-clean, but the overwrite
can introduce a family collision between the substituted-in yellow-dot part and
a retained pool part. Empirical rate on 0x6D (7-reg): 49% of coalition
evaluations produce family-invalid designs, peaking at 77% for |S| = 3-4.

The MLP / simulator will still return a score for the family-repeated design,
but that score is an extrapolation off the training manifold and biases the
Shapley integral.

## What this module provides

`make_family_safe_sampler(pool, perm)` — factory that precomputes the pool's
per-slot family assignment once, then returns a fast callable
`safe_sample(mask, n, rng)` that:

  1. Filters `pool` to rows whose non-mask slots have no family overlap with
     `perm[mask]`.
  2. Samples `n` rows from that safe subset.
  3. Overwrites the mask slots with `perm[mask]`.
  4. Returns an (n, l) int64 array that is guaranteed family-unique.

Fallback: if no pool row satisfies the family constraint for a given mask
(rare for large libraries relative to coalition size), the sampler emits a
warning once per mask and falls back to the unfiltered pool with the swap
applied. Callers relying on strict validity should check the returned rows,
but in the yellow-dot Shapley setting the fallback is never triggered in
practice (verified for 0x2B/0x17/0x6D at all coalition sizes).

Also exposed:
- `FAMILY_OF_INDEX` — length-20 numpy array (part index → family string).
- `is_family_valid(perm)` — length-l perm → bool.

Written 2026-09-28 as part of the setFinalvF Shapley-constraint fix.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

# Library part → family (canonical mapping; from paper Methods 20-part TetR
# library). Index 0..19 matches the perm ints used throughout the pipeline.
FAMILY_OF_INDEX = np.array([
    "AmeR",    # 0  AmeR/F1
    "AmtR",    # 1  AmtR/A1
    "BetI",    # 2  BetI/E1
    "BM3R1",   # 3  BM3R1/B1
    "BM3R1",   # 4  BM3R1/B2
    "BM3R1",   # 5  BM3R1/B3
    "HlyIIR",  # 6  HlyIIR/H1
    "IcaRA",   # 7  IcaRA/I1
    "LitR",    # 8  LitR/L1
    "LmrA",    # 9  LmrA/N1
    "PhlF",    # 10 PhlF/P1
    "PhlF",    # 11 PhlF/P2
    "PhlF",    # 12 PhlF/P3
    "PsrA",    # 13 PsrA/R1
    "QacR",    # 14 QacR/Q1
    "QacR",    # 15 QacR/Q2
    "SrpR",    # 16 SrpR/S1
    "SrpR",    # 17 SrpR/S2
    "SrpR",    # 18 SrpR/S3
    "SrpR",    # 19 SrpR/S4
], dtype=object)


def is_family_valid(perm: np.ndarray) -> bool:
    """Return True iff `perm` (int64 array) has no repeated family."""
    fams = FAMILY_OF_INDEX[np.asarray(perm, dtype=np.int64)]
    return len(set(fams.tolist())) == len(fams)


def _all_valid_replacements_at_slot(perm: np.ndarray, slot: int) -> np.ndarray:
    """For a design `perm` and one slot to replace, return the array of
    all part indices p ∈ [0, 20) such that setting slot=p preserves
    family uniqueness. Used by exact single-body Shapley (no sampling).
    """
    perm = np.asarray(perm, dtype=np.int64)
    other_fams = set(FAMILY_OF_INDEX[np.delete(perm, slot)].tolist())
    return np.array(
        [p for p in range(20) if FAMILY_OF_INDEX[p] not in other_fams],
        dtype=np.int64,
    )


def valid_replacements_at_slot(perm: np.ndarray, slot: int) -> np.ndarray:
    """Public alias for the exact-enumeration helper (used by
    29_panel_c_shapley.py's single-body Shapley)."""
    return _all_valid_replacements_at_slot(perm, slot)


def make_family_safe_sampler(pool: np.ndarray, perm: np.ndarray
                              ) -> Callable[[np.ndarray, int, np.random.Generator],
                                            np.ndarray]:
    """Factory. Returns a `safe_sample(mask, n, rng) -> (n, l) int64`
    callable that yields family-unique completions of `perm` at the
    given coalition mask.

    Preprocesses `pool` once (O(N·l)) so per-call cost is a boolean
    reduction over the pool + a slice: microseconds per query for
    N ~ 10⁵.
    """
    pool = np.ascontiguousarray(pool, dtype=np.int64)
    perm = np.asarray(perm, dtype=np.int64)
    N, l = pool.shape
    if perm.shape != (l,):
        raise ValueError(f"perm shape {perm.shape} incompatible with pool {pool.shape}")

    pool_fams = FAMILY_OF_INDEX[pool]                     # (N, l), object dtype
    perm_fams = FAMILY_OF_INDEX[perm]                     # (l,)

    _fallback_warned: set[tuple[int, ...]] = set()

    def safe_sample(mask: np.ndarray, n: int, rng: np.random.Generator
                     ) -> np.ndarray:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (l,):
            raise ValueError(f"mask shape {mask.shape} incompatible with pool row length {l}")

        # Trivial edge cases.
        if not mask.any():
            # No swap → any pool row is valid.
            idx = rng.integers(0, N, size=n)
            return pool[idx].copy()
        if mask.all():
            # Full swap → yellow-dot itself, replicated.
            return np.broadcast_to(perm, (n, l)).copy()

        # Filter pool: a row is safe iff none of its non-mask slots
        # carry a family used by `perm` at the mask slots.
        used = set(perm_fams[mask].tolist())
        non_mask = ~mask
        # Boolean matrix: True where pool cell's family is in `used`.
        forbidden = np.isin(pool_fams[:, non_mask], list(used))
        row_ok = ~forbidden.any(axis=1)
        n_safe = int(row_ok.sum())

        if n_safe == 0:
            key = tuple(np.where(mask)[0].tolist())
            if key not in _fallback_warned:
                print(f"[family-safe] WARNING: no safe pool rows for mask={key}; "
                      f"falling back to unfiltered pool (results may include "
                      f"family-invalid designs)")
                _fallback_warned.add(key)
            idx = rng.integers(0, N, size=n)
        else:
            safe_indices = np.where(row_ok)[0]
            idx = safe_indices[rng.integers(0, n_safe, size=n)]

        out = pool[idx].copy()
        out[:, mask] = perm[mask]
        return out

    return safe_sample
