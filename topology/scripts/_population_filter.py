"""Centralized analysis-scope filter for the cross-topology pipeline.

Every downstream phase (P5, P7, P8, P9, P10, P11, P12, P13) loads
`population.pkl` and iterates topologies. The project's analysis scope
(locked 2026-05-06) is **agent-generated topologies at 4-/5-/6-/7-reg**:
no Hamming-1 baselines, no >7-reg drift cases. Both filters are enforced
here so every phase sees the same set.

Loosening either filter is supported via CLI flags so we can revisit
later without re-editing scripts:
  --include_baselines   include is_baseline=True topologies
  --size_classes 4 5 6 7 8 ...   size classes to retain (regulator_count)

Why filter here rather than at P3 (population assembly)?
- The population manifest stays the **complete** record of what was
  generated/trained. It includes baselines so downstream re-analyses
  with the loosened scope are possible without re-running P3/P4.
- Each phase opts into the analysis scope at load time. This makes the
  scope decision visible per script (a sanity-check moment when reading
  the code) rather than hidden upstream.
- For G1 in particular: 88 trained perm-MLPs + 4 markers cover both
  agent and baseline. Re-running downstream phases with the filter is
  cheap (minutes-to-hours of CPU). Re-training MLPs is not. So we keep
  P4's outputs and just narrow the analysis pass.

Scope decision history:
- 2026-05-06: locked agent-only, 4-7-reg as the canonical analysis scope.
  Memory: project_g1_cross_topology_complete.md, state-doc §2 (decision #7).
"""
from __future__ import annotations

import argparse
from typing import Iterable, Sequence


# Canonical analysis scope (locked 2026-05-06). Loosen via CLI, not by
# edit, so the audit trail of "what was analyzed" stays in script logs.
DEFAULT_SIZE_CLASSES: tuple[int, ...] = (4, 5, 6, 7)
DEFAULT_INCLUDE_BASELINES: bool = False


def add_scope_args(parser: argparse.ArgumentParser) -> None:
    """Attach `--include_baselines` and `--size_classes` to a parser.

    Defaults match the locked project scope (agent-only, 4-7-reg).
    Pass `--include_baselines` to add Hamming-1 baselines back in.
    Pass `--size_classes 4 5 6 7 8 9 ...` to widen the size filter.
    """
    parser.add_argument(
        "--include_baselines",
        action="store_true",
        default=DEFAULT_INCLUDE_BASELINES,
        help="Include Hamming-1 baseline topologies in the analysis. "
             "Default: False (agent-only — locked scope as of 2026-05-06).",
    )
    parser.add_argument(
        "--size_classes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SIZE_CLASSES),
        help="Regulator counts to include. Default: 4 5 6 7. Pass e.g. "
             "`--size_classes 4 5 6 7 8` to widen.",
    )


def in_scope(
    topology: dict,
    *,
    include_baselines: bool = DEFAULT_INCLUDE_BASELINES,
    size_classes: Sequence[int] = DEFAULT_SIZE_CLASSES,
) -> bool:
    """Predicate: is this topology in the active analysis scope?

    Test order matches scope-decision priority:
      1. baseline gate (cheapest reject)
      2. size-class gate

    A topology entry without `regulator_count` is treated as out-of-scope
    so we don't silently include malformed manifest rows.
    """
    if topology.get("is_baseline", False) and not include_baselines:
        return False
    rc = topology.get("regulator_count")
    if rc is None or rc not in size_classes:
        return False
    return True


def filter_topologies(
    topologies: Iterable[dict],
    *,
    include_baselines: bool = DEFAULT_INCLUDE_BASELINES,
    size_classes: Sequence[int] = DEFAULT_SIZE_CLASSES,
    verbose: bool = True,
    label: str = "scope filter",
) -> list[dict]:
    """Apply the scope filter to a topology list and (optionally) log the diff.

    Returns a new list — does not mutate the input.
    """
    src = list(topologies)
    kept: list[dict] = []
    drop_baseline = 0
    drop_size: dict[int, int] = {}
    for t in src:
        if t.get("is_baseline", False) and not include_baselines:
            drop_baseline += 1
            continue
        rc = t.get("regulator_count")
        if rc is None or rc not in size_classes:
            drop_size[rc] = drop_size.get(rc, 0) + 1
            continue
        kept.append(t)

    if verbose:
        size_set = sorted(set(size_classes))
        print(
            f"[{label}] {len(src)} → {len(kept)} topologies after scope filter "
            f"(include_baselines={include_baselines}, "
            f"size_classes={size_set}). "
            f"Dropped {drop_baseline} baselines, "
            f"{sum(drop_size.values())} out-of-size "
            f"({dict(sorted(drop_size.items(), key=lambda kv: (kv[0] is None, kv[0])))})."
        )
    return kept


def scope_summary(
    *,
    include_baselines: bool = DEFAULT_INCLUDE_BASELINES,
    size_classes: Sequence[int] = DEFAULT_SIZE_CLASSES,
) -> dict:
    """Stamp the active scope into output JSONs / summaries for traceability."""
    return {
        "include_baselines": bool(include_baselines),
        "size_classes":      sorted(set(int(s) for s in size_classes)),
        "scope_locked_on":   "2026-05-06",
        "scope_note":        ("Agent-only, 4-7-reg analysis scope. Baselines "
                              "remain in the population manifest but are not "
                              "analyzed. See PROJECT_STATE_TOPOLOGY.md §2."),
    }
