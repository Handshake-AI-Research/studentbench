# Verification targets

These files contain the paper's reported results and full-precision reference estimates. Analyses never read them. The verification command compares them with newly computed outputs:

```bash
python reproduce.py --data data --output results --verify
```

The checks cover learning and IRT analyses, expert reviews, conversation indicators, costs and reply times, engagement and practice, prompt comparisons, subgroup equivalence, dependence and repeat-participant checks, all statistical figure inputs, and every printed numeric cell in the empirical tables. The fixed study-design and interface illustrations are checked by file hash.

- Printed table values must match exactly at their reported precision.
- Full-precision estimates use explicit floating-point tolerances recorded in each target file or validator. Very small p-values are compared relatively, so zero cannot substitute for a small nonzero probability.
- The strict two-criterion practice-ranking sensitivity uses an analytic gradient; its archived finite-difference fit has a separately documented optimization tolerance. The reported ranking and correlation are checked.
- Dataset hashes cover the study records and supporting definitions. The Hugging Face card and illustration previews may change independently.

The run saves individual checks as they finish and produces machine-readable summaries in `results/verification/`. `results/run.json` is complete only after every requested stage succeeds. Formatting and byte-level checks are distinct from scientific comparisons.
