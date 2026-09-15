"""Figure 1 -- the raw MFM stability coordinates X, Y and Z on the cortex, by outcome.

        python3 -W ignore 01_mfm_cortex_by_outcome.py
        python3 -W ignore 01_mfm_cortex_by_outcome.py --mesh=fsaverage     # full resolution

Writes `figures/mfm_XYZ_cortex_by_outcome.png`: three columns (X, Y, Z), two rows (good
outcome on top, poor underneath), each cell a 2x2 of lateral and medial views of both
hemispheres.

What is being painted
---------------------
Nothing is fitted or tested here. Each region's value is the plain cohort mean, within an
outcome group, of each patient's median X / Y / Z in that region over hours 12-48 since ROSC.
X, Y and Z are the Robinson corticothalamic neural-field stability-zone coordinates, and they
are gains of DIFFERENT loops -- X intracortical, Y corticothalamic, Z intrathalamic -- which
is why they are three separate maps rather than one "MFM" quantity.

The two decisions that make the figure readable
-----------------------------------------------
**One colour scale per column, shared by its two rows.** The point of the figure is whether a
coordinate's spatial pattern differs between the outcome groups, and that comparison is only
legible if good and poor sit on the same scale. Across columns the scales are different and
deliberately so: X, Y and Z occupy completely different ranges (X ~ 0.73, Y ~ 0.05, Z ~ 0.10),
so a single scale would flatten two of the three columns to a flat wash. **Colours are
comparable down a column, never across one** -- the per-column colourbar says so.

**Each scale is clipped to the 2nd-98th percentile** over both groups' regions. Between-region
variation in these coordinates is small, so a full min-max scale spends its whole range on one
or two extreme regions and flattens the pattern everywhere else.

The medial wall is left unpainted rather than filled, because it carries no regional value;
see `common.py`. The 18 subcortical regions have no cortical surface at all and are not in
this figure -- the cluster figure's companion heatmap is where deep gray appears.

**No test is computed.** Two group means painted side by side invite a "these look different"
reading; nothing here supports that, and the between-region SD printed under each map is the
scale against which any apparent difference should be judged.
"""

from __future__ import annotations

import sys

import numpy as np

from common import (ADC_DIRECTION, CMAP_DIV, CMAP_SEQ, FEATURE_WINDOW_LABEL, FIG_DPI,
                    FIGURES, MESH_FAST, MFM_PARAMS, OUTCOME_INK, OUTCOME_LABEL,
                    OUTCOME_ORDER, UNITS, apply_style, block_colorbar, block_title,
                    done, draw_brain_block, features_regional, is_subcortical,
                    make_block_grid, outcome_counts, outcomes, row_label, save, suptitle)

CLIP_PCT = (2, 98)
CLIP_LABEL = "2nd–98th percentile"


def group_means() -> dict[str, "object"]:
    """{outcome: DataFrame indexed by ROI, one column per MFM parameter}.

    A plain group mean of the (patient, ROI) table -- no statistic, which is why this script
    computes it inline instead of reading a saved results file.
    """
    d = features_regional().merge(outcomes(), on="patient_id")
    ctx = d[~d.roi.map(is_subcortical)]
    return {g: ctx[ctx.outcome == g].groupby("roi")[MFM_PARAMS].mean()
            for g in OUTCOME_ORDER}


def figure(mesh: str = MESH_FAST) -> None:
    means = group_means()
    n = outcome_counts()
    ncol = len(MFM_PARAMS)

    grid = make_block_grid(2 * ncol, ncol_max=ncol, block_h=2.95, header_in=0.95,
                           footer_in=0.10)
    for c, param in enumerate(MFM_PARAMS):
        # One scale per column, spanning both groups: the rows are only comparable if they
        # share it. Clipped, because the between-region spread is small (see the docstring).
        both = np.r_[means["good"][param].to_numpy(), means["poor"][param].to_numpy()]
        lo, hi = (float(v) for v in np.nanpercentile(both, CLIP_PCT))
        # A diverging ramp only where the values genuinely straddle zero; otherwise its white
        # midpoint would be a value the data never takes.
        cmap = CMAP_DIV if lo < 0 < hi else CMAP_SEQ

        for r, grp in enumerate(OUTCOME_ORDER):
            i = r * ncol + c
            vals = means[grp][param]
            draw_brain_block(grid, i, vals.to_dict(), cmap, lo, hi, mesh=mesh,
                             fill=float(np.nanmedian(vals)))
            if r == 0:
                block_title(grid, i, f"{param}   [{UNITS[param]}]",
                            f"{OUTCOME_LABEL[grp]} (n={n[grp]})  ·  between-region SD "
                            f"{vals.std():.3g}", subtitle_color=OUTCOME_INK[grp])
            else:
                block_title(grid, i, "",
                            f"{OUTCOME_LABEL[grp]} (n={n[grp]})  ·  between-region SD "
                            f"{vals.std():.3g}", subtitle_color=OUTCOME_INK[grp])
                # The colourbar goes under the lower row only: it is shared by both rows of
                # the column, and one per cell would say the opposite.
                block_colorbar(grid, i, cmap, lo, hi, f"{param} [{UNITS[param]}]")

    for r, grp in enumerate(OUTCOME_ORDER):
        row_label(grid, r, OUTCOME_LABEL[grp], color=OUTCOME_INK[grp])

    suptitle(
        grid,
        "MFM stability-zone coordinates on the cortex, by neurological outcome — cohort "
        f"mean of each patient's {FEATURE_WINDOW_LABEL} value per region\n"
        f"good n={n['good']}, poor n={n['poor']} · 68 cortical Desikan-Killiany ROIs · "
        f"{mesh} + DK aparc.annot · X intracortical, Y corticothalamic, Z intrathalamic "
        "loop gain\n"
        f"one colour scale per column (clipped at the {CLIP_LABEL}), shared by its two "
        "rows: colours are comparable down a column, never "
        "across one · no MRI and no test in this figure",
        fontsize=10.5)
    out = FIGURES / "mfm_XYZ_cortex_by_outcome.png"
    save(grid, out, FIG_DPI)
    done(out)


def main(argv: list[str]) -> None:
    mesh = MESH_FAST
    for a in argv:
        if a.startswith("--mesh"):
            mesh = a.split("=", 1)[1] if "=" in a else "fsaverage"
    apply_style()
    n = outcome_counts()
    print(f"cohort: {sum(n.values())} retained patients ({n['good']} good / {n['poor']} "
          f"poor) · mesh {mesh}")
    print(f"note: the MRI target used elsewhere in this folder is median ADC "
          f"({ADC_DIRECTION}); this figure has no MRI in it")
    figure(mesh)


if __name__ == "__main__":
    main(sys.argv[1:])
