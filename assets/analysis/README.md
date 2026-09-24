# Inputs for the equivalence analyses

`pooled_equivalence_cr2.json` contains global CR2 moment sums for the combined
Quantitative and Verbal comparisons, with and without repeat participants.
The regression coefficients, equivalence margins, confidence intervals and
TOST p-values are recomputed by `studentbench/primary_equivalence.py`.
`individual_equivalence_cr2.json` supplies the corresponding global moments for
12 individual AI tutors, both for the full cohort and after repeat-participant
exclusion. `studentbench/individual_equivalence.py` refits each comparison with
the same section weights, covariates and fixed full-cohort margin.

The study's dependence structure includes students who completed both sections
and human tutors who taught in both sections. The files supply sums needed to
account for this dependence without including direct identifiers, private
participant identity keys or per-cluster records. Anonymous tutor groups can span
both sections. Section-specific
fits use the anonymous tutor identifiers in the released dataset directly.

For design matrix `X`, contrast `c`, and cluster `g`, define
`B = (X'X)^-1`, `u_g = (I - X_g B X_g')^(-1/2) X_g B c`,
`D_g = X_g' u_g`, `r_g = u_g' y_g`, and `a_g = u_g' u_g`.
The released sums are:

| Field | Sum over all clusters |
| --- | --- |
| `design_outer_sum` | `D_g D_g'` |
| `norm_weighted_design_outer_sum` | `a_g D_g D_g'` |
| `response_design_sum` | `r_g D_g` |
| `response_square_sum` | `r_g^2` |
| `norm_sum` | `a_g` |
| `norm_square_sum` | `a_g^2` |

For the fitted coefficient vector `beta`, the CR2 variance is
`response_square_sum - 2 beta' response_design_sum + beta' design_outer_sum beta`.
The implementation derives the Satterthwaite degrees of freedom from the
design-only sums. Each input has a fingerprint of its released response and
design rows; changed observations cause the analysis to stop.

These are aggregate analysis inputs, not a release of the private real-world
identity mapping. Re-estimating moments for arbitrary new models requires the
underlying dependence structure.

The pooled input also contains study-level dependence counts: seven clusters
contain human sessions, and the largest contains 102 of the 140 human sessions.
These describe the restricted linkage calculation; they do not provide membership
lists.

`leave_one_tutor_out_cr2.json` defines 13 omitted-cohort comparisons using only
the anonymous tutor aliases already present in the dataset. The neutral labels
are ordered by those public aliases. Each fit re-estimates its regression from
the remaining public sessions and uses global CR2 moments for the remaining
dependence structure. The fixed full-cohort section weights and margins are
retained. All 13 comparisons are checked against the paper's results; no actual
tutor identity, participant-pair mapping or calendar timestamp is included.
