"""figS05 v2 palette — red-tier "bad" emphasis (alternative to the
locked grey/blue convention) shared across Panels A, B, C v2.

Positive / headline stays ACCENT_BLUE. Negative tiers switch from
DARK_GRAY / MID_GRAY / near-black to a tiered red ramp so "bad" reads
chromatically rather than just darkness-wise.

Use only in v2 panel scripts under this folder; the rest of the
paper_figures pipeline still follows the canonical grey/blue palette
(see feedback_paper_grey_blue_palette).
"""

# Strongest "bad" emphasis — matches the SrpR/S4 maroon already in
# figS04 Panel C, so the cross-figure red identity is consistent.
BAD_DEEP   = "#A50026"
# Tier-2 negative.
BAD_TIER2  = "#E25C5C"
# Background region wash (use at low alpha ~0.08 so it tints, not floods).
BAD_BAND   = "#C62828"
# Diverging-cmap negative endpoint (same as deep).
BAD_CMAP_END = "#A50026"
