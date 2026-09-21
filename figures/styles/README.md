# Figure layout

These files set the fonts, label positions, legends, logos and canvas sizes used
in the paper's figures. Figure 6 gets its layout directly from Matplotlib.

The plotting scripts calculate points, intervals and labels from the analysis
outputs. `studentbench/publication_style.py` applies the layout settings and
checks that the plotted values and label text are preserved.

Add `--native-figures` to use the original plotting layout when working with
modified data. Matching the paper's fonts requires Arial.
