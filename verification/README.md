# Verification targets

These files hold the reference results used by `--verify`. The command compares them with your analysis outputs:

```bash
python reproduce.py --data data --output results --verify
```

The checks cover paper revision `47fa622`: learning and IRT analyses, expert reviews, conversation indicators, costs and reply times, engagement and practice, prompt comparisons, subgroup equivalence, dependence and repeat-participant checks, all statistical figure inputs, and every table cell. [Prose checks](../docs/paper_results.md) cover the numbers, inequalities and rankings stated in the main text and appendix. Fixed illustrations and all 15 table LaTeX fragments are checked by file hash.

- Complete table cells must match at their reported precision, including row labels, signs, inequalities and missing entries.
- Full-precision estimates use explicit floating-point tolerances recorded in each target file or validator. Very small p-values are compared relatively, so zero cannot substitute for a small nonzero probability.
- Dataset hashes cover the study records and supporting definitions. The Hugging Face card and illustration previews may change independently.

Checks are saved in `results/verification/`. `results/run.json` reports whether all requested stages finished.

Add `--verify-figure-bytes` to require byte-identical files for all 24 figures. This requires the paper's Arial fonts and [rendering environment](render_environment.json). Standard `--verify` checks the figure values independently of platform-dependent rendering.
