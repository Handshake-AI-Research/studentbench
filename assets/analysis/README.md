# Inputs for the pooled equivalence analysis

`pooled_equivalence_cr2.json` contains global CR2 moment sums for the combined
Quantitative and Verbal comparisons, with and without repeat participants.
The regression coefficients, equivalence margins, confidence intervals and
TOST p-values are recomputed by `studentbench/primary_equivalence.py`.

The study's dependence structure includes students who completed both sections
and human tutors who taught in both sections. These cross-section identity links
are private. The file supplies the sums needed to account for that dependence
without publishing identity links or individual-cluster records. Section-specific
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

These are aggregate analysis inputs, not a release of the private identity
mapping. Re-estimating the cross-section moments for a different model would
require access to those protected links.
