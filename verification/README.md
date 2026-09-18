# Verification targets

These files hold the reference results used by `--verify`. The command compares them with your analysis outputs:

```bash
python reproduce.py --data data --output results --verify
```

The checks cover learning and IRT analyses, expert reviews, conversation indicators, costs and reply times, engagement and practice, prompt comparisons, subgroup equivalence, dependence and repeat-participant checks, all statistical figure inputs, and every printed numeric cell in the empirical tables. The fixed study-design and interface illustrations are checked by file hash.

- Printed table values must match exactly at their reported precision.
- Full-precision estimates use explicit floating-point tolerances recorded in each target file or validator. Very small p-values are compared relatively, so zero cannot substitute for a small nonzero probability.
- The strict two-criterion practice-ranking sensitivity uses an analytic gradient; its archived finite-difference fit has a separately documented optimization tolerance. The reported ranking and correlation are checked.
- Dataset hashes cover the study records and supporting definitions. The Hugging Face card and illustration previews may change independently.

Checks are saved in `results/verification/`. `results/run.json` reports whether all requested stages finished.
