"""Phase 0 — select target Boolean functions for the cross-topology study.

Enumerates all 256 3-input Boolean functions, groups them into NPN equivalence
classes, and validates that the hand-picked 20-target diversity sample
(locked in PROJECT_STATE_TOPOLOGY.md §4.3) covers the structural space.

Outputs:
    configs/target_functions.yaml   — machine-readable selection + metadata
    reports/target_function_selection.md — human-readable narrative + coverage table

Usage:
    cd topology/scripts
    python 00_select_targets.py
    # or with custom output dirs:
    python 00_select_targets.py --out-dir ../configs --report-dir ../reports
"""
from __future__ import annotations
import argparse
import itertools
from pathlib import Path

N_INPUTS = 3
N_STATES = 1 << N_INPUTS    # 8
N_FUNCS = 1 << N_STATES     # 256

# 20 targets, partitioned into cumulative groups G1 / G2 / G3 (see
# PROJECT_STATE_TOPOLOGY.md §4.3-4.4). Order within the list follows the
# planned execution order.
HANDPICKED: list[tuple[int, str, str, str]] = [
    # ---- G1 (pilot, n=4): 3 anchors + 1 fresh small-size target ----
    (0x2B, "G1", "DeepCircMI anchor — known 5-reg yellow-dot",                "5-reg"),
    (0x17, "G1", "DeepCircMI anchor — known 6-reg yellow-dot",                "6-reg"),
    (0x6D, "G1", "DeepCircMI anchor — known 7-reg yellow-dot",                "7-reg"),
    (0xEE, "G1", "OR(A,B), 2-input function; fills 4-reg coverage anchors miss", "4-5-reg"),

    # ---- G2 (cumulative n=10; +6 new): NPN diversity + structural variety ----
    (0x4D, "G2", "NPN 0x17 variant (asym 3-input); 4th rep of well-fitting class", "5-6-reg"),
    (0xCC, "G2", "f=B (1-input degenerate, NPN-equivalent to 0x33 inverted)",     "4-reg"),
    (0xE8, "G2", "majority(A,B,C); biologically iconic",                      "5-6-reg"),
    (0x33, "G2", "XOR-like structure; distinct NPN class",                    "5-6-reg"),
    (0x66, "G2", "XOR(B,C), 2-input (A degenerate); 2nd rep of small NPN 0x3C", "4-5-reg"),
    (0x3C, "G2", "symmetric-pair; balance against asymmetric picks",          "5-6-reg"),

    # ---- G3 (cumulative n=20; +10 new): completes 13/13 non-trivial NPN coverage ----
    (0x2C, "G3", "smoke-test seed; verified through cluster pipeline",        "4-5-reg"),
    (0x03, "G3", "HW=2, two adjacent minterms",                               "4-5-reg"),
    (0x0F, "G3", "depends on 2 vars; degenerate-input handling",              "4-reg"),
    (0x06, "G3", "NPN class 0x06 representative — XOR(A,B)·NOT(C) structure", "5-reg"),
    (0x07, "G3", "NPN class 0x07 representative — three-minterm OR-like",     "4-5-reg"),
    (0x18, "G3", "NPN class 0x18 representative — distinct mid-complexity asymmetric", "5-reg"),
    (0xB4, "G3", "asymmetric mid-complexity; distinct NPN class",             "5-6-reg"),
    (0x77, "G3", "HW=6, mostly-1; NOT-side balance",                          "5-reg"),
    (0x47, "G3", "filler with distinct NPN class",                            "5-6-reg"),
    (0x1B, "G3", "filler with distinct NPN class",                            "5-6-reg"),
]


# ---------------------------------------------------------------------------
# Boolean-function helpers
# ---------------------------------------------------------------------------

def hamming_weight(f: int) -> int:
    return bin(f).count("1")


def truth_table_string(f: int) -> str:
    """LSB-first bit string, one char per input state 0..7."""
    return "".join(str((f >> i) & 1) for i in range(N_STATES))


def is_symmetric(f: int) -> bool:
    """Symmetric: output depends only on the count of 1-bits in the input."""
    by_weight: dict[int, int] = {}
    for state in range(N_STATES):
        w = bin(state).count("1")
        bit = (f >> state) & 1
        if w in by_weight and by_weight[w] != bit:
            return False
        by_weight[w] = bit
    return True


def depends_on(f: int) -> tuple[bool, ...]:
    """For each input position i (0..N_INPUTS-1), does f actually depend on input i?"""
    out: list[bool] = []
    for i in range(N_INPUTS):
        flips = False
        for state in range(N_STATES):
            other = state ^ (1 << i)
            if ((f >> state) & 1) != ((f >> other) & 1):
                flips = True
                break
        out.append(flips)
    return tuple(out)


def n_dependent_inputs(f: int) -> int:
    return sum(depends_on(f))


def apply_input_permutation(f: int, perm: tuple[int, ...]) -> int:
    """New function g where g's input position i is original f's input position perm[i]."""
    new_f = 0
    for old_state in range(N_STATES):
        bit = (f >> old_state) & 1
        new_state = 0
        for i in range(N_INPUTS):
            new_state |= ((old_state >> perm[i]) & 1) << i
        new_f |= bit << new_state
    return new_f


def apply_input_negation(f: int, neg_mask: int) -> int:
    """Negate inputs whose bit in neg_mask is set."""
    new_f = 0
    for state in range(N_STATES):
        bit = (f >> state) & 1
        new_f |= bit << (state ^ neg_mask)
    return new_f


def apply_output_negation(f: int) -> int:
    return f ^ ((1 << N_STATES) - 1)


def npn_canonical(f: int) -> int:
    """Smallest hex among all NPN-equivalent forms of f.

    Iterates over 6 input perms × 8 input-negation masks × 2 output negations = 96 forms.
    """
    canon = f
    for perm in itertools.permutations(range(N_INPUTS)):
        f1 = apply_input_permutation(f, perm)
        for neg_mask in range(1 << N_INPUTS):
            f2 = apply_input_negation(f1, neg_mask)
            for out_neg in (0, 1):
                f3 = apply_output_negation(f2) if out_neg else f2
                if f3 < canon:
                    canon = f3
    return canon


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_yaml(handpicked, canonical_of, npn_classes, canon_to_funcs,
                selected_canons, missing_canons) -> str:
    by_group: dict[str, list[int]] = {"G1": [], "G2": [], "G3": []}
    for i, (h, group, _, _) in enumerate(handpicked, 1):
        by_group[group].append(i)

    lines = [
        "# DeepCirc topology — Phase 0 target function selection",
        "# Generated by scripts/00_select_targets.py — do not edit by hand",
        "# Re-run the script to regenerate after editing the HANDPICKED list.",
        "",
        f"n_inputs: {N_INPUTS}",
        f"n_total_functions: {N_FUNCS}",
        f"n_npn_classes: {len(npn_classes)}",
        f"n_selected: {len(handpicked)}",
        f"npn_classes_covered: {len(selected_canons)}",
        f"all_npn_classes_covered: {str(len(missing_canons) == 0).lower()}",
        "",
        "# Phased rollout (cumulative groups; see PROJECT_STATE_TOPOLOGY.md §4.3):",
        f"#   G1 (pilot)      = ids {by_group['G1']}  (n={len(by_group['G1'])} cumulative)",
        f"#   G2 (scale-up)   = ids {by_group['G1'] + by_group['G2']}  (n={len(by_group['G1']) + len(by_group['G2'])} cumulative)",
        f"#   G3 (completion) = all ids 1..{len(handpicked)}  (n={len(handpicked)} cumulative)",
        "",
        "groups:",
        f"  G1: {by_group['G1']}",
        f"  G2: {by_group['G1'] + by_group['G2']}",
        f"  G3: {by_group['G1'] + by_group['G2'] + by_group['G3']}",
        "",
        "# Selection criteria (locked in PROJECT_STATE_TOPOLOGY.md §4.2):",
        "#   - Span Hamming weight 1-6 (skip trivial 0/7/8)",
        "#   - At least one representative from each of the 14 NPN classes",
        "#   - Mix of symmetric and asymmetric",
        "#   - Include 0x2B / 0x17 / 0x6D as DeepCircMI continuity anchors",
        "#   - Include 0x69 / 0x96 (parity, hard) and 0xE8 (majority, iconic)",
        "",
        "targets:",
    ]
    for i, (h, group, rationale, size) in enumerate(handpicked, 1):
        lines.extend([
            f"  - id: {i}",
            f"    hex: \"0x{h:02X}\"",
            f"    decimal: {h}",
            f"    group: \"{group}\"",
            f"    truth_table: \"{truth_table_string(h)}\"   # bit i = output for input state i (LSB first)",
            f"    hamming_weight: {hamming_weight(h)}",
            f"    n_dependent_inputs: {n_dependent_inputs(h)}",
            f"    depends_on_inputs: {list(depends_on(h))}",
            f"    is_symmetric: {str(is_symmetric(h)).lower()}",
            f"    npn_canonical: \"0x{canonical_of[h]:02X}\"",
            f"    expected_size_class: \"{size}\"",
            f"    rationale: \"{rationale}\"",
        ])
    lines.append("")
    if missing_canons:
        lines.append("# NPN classes NOT covered by the 20 picks (consider adding for full coverage):")
        for c in missing_canons:
            example_funcs = canon_to_funcs[c]
            ex_str = ", ".join(f"0x{f:02X}" for f in example_funcs[:6])
            lines.append(f"#   canonical=0x{c:02X}  ({len(example_funcs)} functions)  e.g. {ex_str}")
    return "\n".join(lines) + "\n"


def format_markdown(handpicked, canonical_of, npn_classes, canon_to_funcs,
                    selected_canons, missing_canons) -> str:
    md = [
        "# Phase 0 — Target Function Selection (3-input Boolean functions)",
        "",
        "**Generated by:** `scripts/00_select_targets.py`",
        "",
        "## Summary",
        "",
        f"- Total 3-input Boolean functions: **{N_FUNCS}**",
        f"- NPN equivalence classes: **{len(npn_classes)}** (expected 14 for n=3)",
        f"- Targets selected: **{len(handpicked)}**",
        f"- NPN classes covered by selection: **{len(selected_canons)} of {len(npn_classes)}**",
        f"- All NPN classes covered: **{'✅ YES' if not missing_canons else '⚠️ NO'}**",
        "",
        "## Selection criteria",
        "",
        "Locked in `PROJECT_STATE_TOPOLOGY.md` §4.2:",
        "",
        "- **Hamming weight diversity**: span 1–6 (skip trivial 0 / 7 / 8)",
        "- **NPN coverage**: at least one representative from each of the 14 NPN classes",
        "- **Symmetry mix**: include both symmetric (parity, majority) and asymmetric",
        "- **DeepCircMI continuity**: include 0x2B / 0x17 / 0x6D as anchors against existing exemplar work",
        "- **Hard cases**: include 0x69 / 0x96 (3-input parity) and 0xE8 (majority)",
        "",
        "## Phased rollout",
        "",
        "Per PROJECT_STATE_TOPOLOGY.md §4.3, the 20 targets are run in three",
        "**cumulative** groups, with explicit decision points between groups:",
        "",
        "| group | targets cumulative | new in this group | goal |",
        "|---|---|---|---|",
        "| G1 — pilot | 4 | 4 | Validate pipeline; calibrate QC thresholds; reproduce known exemplar R² |",
        "| G2 — scale-up | 10 | +6 | NPN diversity + statistical power for cross-topology rules |",
        "| G3 — completion | 20 | +10 | Full Tier-A scope; complete L4 rule book |",
        "",
        "## Selected targets",
        "",
        "| # | hex | group | bits (LSB→MSB) | HW | dep. inputs | symmetric | NPN class | expected size | rationale |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, (h, group, rationale, size) in enumerate(handpicked, 1):
        md.append(
            f"| {i} | `0x{h:02X}` | **{group}** | `{truth_table_string(h)}` | {hamming_weight(h)} | "
            f"{n_dependent_inputs(h)} | {'✓' if is_symmetric(h) else '–'} | "
            f"`0x{canonical_of[h]:02X}` | {size} | {rationale} |"
        )

    md.extend([
        "",
        "## NPN class coverage",
        "",
        f"There are {len(npn_classes)} NPN equivalence classes for 3-input Boolean functions.",
        "Each class groups functions identical up to input permutation, input negation,",
        "and output negation. Functions in the same class have identical logical complexity.",
        "",
        "| NPN canonical | # functions | covered? | example members |",
        "|---|---|---|---|",
    ])
    for c in npn_classes:
        funcs_in_class = canon_to_funcs[c]
        covered = "✅" if c in selected_canons else "❌"
        examples = ", ".join(f"`0x{f:02X}`" for f in funcs_in_class[:6])
        if len(funcs_in_class) > 6:
            examples += " …"
        md.append(f"| `0x{c:02X}` | {len(funcs_in_class)} | {covered} | {examples} |")

    if missing_canons:
        md.extend([
            "",
            "## Missing NPN classes",
            "",
            "The 20-target selection does **not** cover the following NPN classes:",
            "",
        ])
        for c in missing_canons:
            funcs_in_class = canon_to_funcs[c]
            ex = ", ".join(f"`0x{f:02X}`" for f in funcs_in_class[:8])
            md.append(f"- `0x{c:02X}` ({len(funcs_in_class)} functions): {ex}")
        md.extend([
            "",
            "Consider adding one representative from each missing class if full NPN",
            "coverage is desired. The 20-target plan was a starting size; per",
            "PROJECT_STATE_TOPOLOGY.md §3, we can scale to 30–50 targets if needed.",
        ])

    md.extend([
        "",
        "## Reproducibility",
        "",
        "```bash",
        "cd topology/scripts",
        "python 00_select_targets.py",
        "```",
        "",
        "Outputs are deterministic. The hand-picked target list is hard-coded in the",
        "`HANDPICKED` constant at the top of `00_select_targets.py`; to change the",
        "selection, edit that list and re-run the script.",
        "",
        "## Glossary",
        "",
        "- **Truth table** (`bits` column): for input state `s ∈ {0..7}`, character `s`",
        "  of the bit string is the output. Order is LSB-first: `s = (input_2, input_1, input_0)`.",
        "- **Hamming weight (HW)**: number of input states that produce output 1 (count of 1s in the bit string).",
        "- **Dependent inputs**: number of inputs that the function actually depends on (a function may not use all 3 inputs).",
        "- **Symmetric**: output depends only on the count of 1-inputs, not on which specific inputs are 1 (e.g., majority, parity).",
        "- **NPN class**: equivalence class under negation/permutation/negation. Two functions are in the same NPN class if one can be obtained from the other by permuting inputs, negating individual inputs, or negating the output.",
        "",
    ])

    return "\n".join(md) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    ap.add_argument("--out-dir",    type=Path, default=here.parent / "configs",
                    help="Where to write target_functions.yaml")
    ap.add_argument("--report-dir", type=Path, default=here.parent / "reports",
                    help="Where to write target_function_selection.md")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    print(f"Phase 0 — target function selection")
    print(f"  enumerating all {N_FUNCS} 3-input Boolean functions...")

    canonical_of = [npn_canonical(f) for f in range(N_FUNCS)]
    canon_to_funcs: dict[int, list[int]] = {}
    for f in range(N_FUNCS):
        canon_to_funcs.setdefault(canonical_of[f], []).append(f)
    npn_classes = sorted(canon_to_funcs.keys())

    print(f"  → {len(npn_classes)} NPN equivalence classes (expected 14)")
    if len(npn_classes) != 14:
        print(f"  ⚠️  WARNING: expected 14 classes, got {len(npn_classes)} — check NPN logic")

    selected_canons = sorted({canonical_of[h] for h, _, _, _ in HANDPICKED})
    missing_canons = [c for c in npn_classes if c not in selected_canons]

    print(f"  selection: {len(HANDPICKED)} targets, covering {len(selected_canons)} of {len(npn_classes)} NPN classes")
    if missing_canons:
        print(f"  ⚠️  missing NPN classes: " + ", ".join(f"0x{c:02X}" for c in missing_canons))
    else:
        print(f"  ✅ all NPN classes covered")

    out_yaml = args.out_dir / "target_functions.yaml"
    out_yaml.write_text(format_yaml(HANDPICKED, canonical_of, npn_classes,
                                    canon_to_funcs, selected_canons, missing_canons))
    print(f"\nWrote {out_yaml}")

    out_md = args.report_dir / "target_function_selection.md"
    out_md.write_text(format_markdown(HANDPICKED, canonical_of, npn_classes,
                                      canon_to_funcs, selected_canons, missing_canons))
    print(f"Wrote {out_md}")

    print()
    print("=" * 60)
    print("Phase 0 complete.")
    print(f"  selected: {len(HANDPICKED)} targets")
    print(f"  NPN coverage: {len(selected_canons)} / {len(npn_classes)}")
    if missing_canons:
        print(f"  ⚠️  next: decide whether to expand selection to cover the {len(missing_canons)} missing classes")
    else:
        print(f"  ✅ all NPN classes represented")
    print("=" * 60)


if __name__ == "__main__":
    main()
