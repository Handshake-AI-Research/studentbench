These layout files preserve the final paper's typography: fonts, label positions,
legend alignment, annotation backings, provider logos, and export canvas sizes.
They contain no measured values or scientific paths. Figure 6 uses its native
Matplotlib export instead.

Each numbered Python figure computes its marks and label text from new analysis
outputs. `studentbench/publication_style.py` then fills character slots with that
new text and verifies its content has not changed. It also fingerprints every
scientific shape, including inherited transforms and clipping, before and after
styling. Only named layout elements (legends, logos, white label backings and
annotation guides) are outside this comparison.

The plots can also be generated without the publication styling using
`--native-figures`. Matching the final paper's PDF bytes requires the pinned
rendering tools and Arial fonts. The underlying calculations do not depend on
these layout files or fonts.
