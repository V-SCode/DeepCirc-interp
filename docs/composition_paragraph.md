# Composition-to-function — main-text synthesis paragraph

Closing two-paragraph synthesis for the **"Learning composition to function in
complex gene circuits"** section of the DeepCirc paper. Walks figS10 → figS15
in order. Locked 2026-06-02.

---

> We next sought to understand the composition-to-function rules that DeepCirc
> had learned. To this end, we used the trained neural networks to score the
> full part-assignment space of all 215 trained G3 topologies (approximately
> 1.4 × 10⁹ candidate designs across 20 target Boolean functions and three
> regulator counts) and examined the structural and library composition of the
> highest- and lowest-scoring designs. Across topology sizes, predicted scores
> converged to a common upper limit near 2,200 RPU at the 99.99th percentile,
> but the elite tier thinned and the failure tail deepened as regulator count
> rose, indicating that adding regulators does not raise the achievable
> performance ceiling but instead reduces the density of high-performing
> designs (Fig. S10b, c). We then examined the structural motifs underlying
> the elite and failure tails (Fig. S10d). Enriched motifs were consistently
> 3- and 4-node subgraphs with balanced fan-out (maximum out-degree 2),
> whereas depleted motifs were either over-fanned (maximum out-degree 3) or
> collapsed to sparse linear chains. Notably, depletion magnitudes
> substantially exceeded enrichment magnitudes, with log₂ enrichment up to +3
> in the top 5% versus as low as −13 in the bottom 5%, indicating that
> specific structural compositions are more strongly associated with failure
> than success. Replacing each scaffold with its most-enriched NIG-typed
> variant sharpened these signals by an order of magnitude (Fig. S11): every
> top-5% typed motif placed NOR gates at input-adjacent positions paired with
> at least one NOT buffer at intermediate or output-adjacent depth, while
> every bottom-5% typed motif either doubled NOR at the output-adjacent slot
> or chained input-adjacent NORs without a NOT buffer. Beyond gate-composition
> motifs, we mapped the buildable design space at the topology-population
> scale by computing Pareto fronts (predicted circuit score vs. growth score)
> for all 215 topologies and aggregating them across the nine Boolean-function
> NPN equivalence classes represented in the substrate (Fig. S12). Pareto
> fronts within a class commonly overlapped across distinct target functions,
> consistent with NPN-equivariance in the agent's design solutions. This
> broad Pareto map served as the basis for the library-part composition
> analysis described below.

> To resolve library-part composition, we pre-specified two selection criteria
> with which to draw a curated design cohort from the broader Pareto map: the
> maximum-predicted-circuit-score design per topology and the joint-score
> Pareto-knee design per topology. Both criteria are our analytical choice
> rather than natural features of the design space. Applying them yielded
> 215 + 215 representative designs that we then analyzed for part composition
> (Fig. S13). The two cohorts separated within the buildable region by
> construction, the max-circuit cohort occupying a higher circuit-score /
> lower growth-score band and the Pareto-knee cohort the converse. Notably,
> the three published anchor designs from Fig. 3 sat within this cloud rather
> than at its upper-right corner, indicating that the published designs are
> simulator-confirmed picks rather than predicted maxima. Across the 30 best
> designs per target Boolean function (15 per criterion), one library part
> (PhlF/P2) appeared in every design and was systematically placed at the
> output-adjacent slot, four further parts (QacR/Q2, BetI/E1, AmtR/A1, and
> SrpR/S4) carried the bulk of the remaining positions, and the 20-part
> library effectively reduced to a small core within this curated cohort.
> The pivot between the two selection criteria reduced to a single
> substitution: PsrA/R1 appeared in 14 of 15 max-circuit picks but in none of
> the 15 knee picks, where it was consistently replaced by QacR/Q2. These
> rules persisted at the single-part and pairwise levels. On the growth axis,
> IcaRA/I1 was depleted from the top-5% tier at every position (median log₂
> enrichment = −4.35) while QacR/Q2 was universally enriched (median = +2.25;
> Fig. S14a); on the circuit axis, PhlF/P1 was the sole destructive-extreme
> outlier (median = −1.47) and PhlF/P2 led the constructive cluster
> (median = +0.56; Fig. S14b). PhlF/P1 therefore appeared simultaneously at
> the protective extreme of the growth landscape and the destructive extreme
> of the circuit landscape, exemplifying the part-level form of the
> joint-objective trade-off; the within-family RBS-variant flips of PhlF
> (P2 vs. P1 on circuit) and QacR (Q2 vs. Q1 on growth) further indicate that
> DeepCirc discriminates at the protein-RBS-pair level rather than the
> TF-family level. We then resolved the per-part circuit signal across cascade
> positions (Fig. S14c) and observed strong position dependence: the
> output-adjacent NOT slot carried both the strongest constructive cell in
> the matrix (PhlF/P2, log₂ = +1.03) and the strongest destructive cell
> (AmeR/F1, log₂ = −5.08), a position-specific failure for an otherwise
> neutral part. Single-body Shapley decomposition across the 30-design set
> was uniformly positive across all slots for predicted circuit score but
> flipped to uniformly negative across the 6- and 7-regulator max-circuit
> designs for predicted growth (Fig. S15a), providing a direct per-design
> visualization of the slot-by-slot growth tax that maximizing circuit margin
> imposes. Finally, simulator-valued Shapley-Taylor pairwise epistasis on the
> three published anchor designs showed circuit-side coefficients roughly
> three orders of magnitude larger than growth-side coefficients (maximum
> |Φᵢⱼ| = 29.7 vs. 0.018 on the 6-regulator anchor 0x17), with the single
> largest pair (PhlF-P2 × PsrA-R1, Φᵢⱼ = +29.7) exceeding every single-slot
> main effect except one, and every off-diagonal coefficient positive across
> all three anchors (Fig. S15b). Together, these analyses indicate that
> DeepCirc captures composition-to-function rules at multiple structural and
> library levels and narrows the 20-part library to a small core toolkit
> within our curated design region.
