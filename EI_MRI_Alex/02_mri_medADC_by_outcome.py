"""Figure 2 -- regional median ADC on the cortex, by outcome.

        python3 -W ignore 02_mri_medADC_by_outcome.py
        python3 -W ignore 02_mri_medADC_by_outcome.py --mesh=fsaverage   # full res, slow

Writes `figures/mri_medADC_cortex_by_outcome.png`: two rows, good outcome on top and poor
underneath, four views across (LH and RH, lateral and medial). Same row structure as figure 1
-- outcome down the left edge -- so the MFM maps and the injury maps read the same way; the
views are spread across the row here because there is only one quantity to show, not three.

What is being painted
---------------------
Each region's value is the plain cohort mean, within an outcome group, of that patient's
median ADC in that region -- the apparent diffusion coefficient from the MRI, which falls
where cytotoxic oedema develops after cardiac arrest. **Higher median ADC = healthier
tissue.** That is the opposite direction from what an "injury map" usually implies, and it is
why the caption and the colourbar both say so: the darker regions here are the LESS injured
ones.

The MRI is a single scan per patient, so unlike the EEG/MFM features there is no time window
to choose and nothing is interpolated.

Scale
-----
**Both rows share one colour scale**, because the entire point of the figure is the
difference between them. It is clipped to the 2nd-98th percentile over both groups' regions:
between-region variation in median ADC is as large as the between-group variation, so a full
min-max scale spends its whole range on a handful of extreme regions and flattens exactly the
group difference the figure exists to show.

The medial wall is left unpainted rather than filled -- it carries no regional value, and
filling it with the bottom of the scale would read as "most injured here". The 18 subcortical
regions have no cortical surface at all and so are absent from this figure; that matters more
here than anywhere else in this folder, because the deep gray nuclei carry the largest
good-versus-poor ADC difference in this cohort.

**No test is computed.** Two group means painted side by side are a description, not a
comparison; the between-region SD printed under each map is the scale against which the
apparent difference should be judged.
"""

from __future__ import annotations

import sys

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from common import (ADC_DIRECTION, ADC_LABEL, ADC_REGIONAL, ADC_UNITS, CMAP_SEQ, FIG_DPI,
                    FIGURES, MESH_FAST, OUTCOME_INK, OUTCOME_LABEL, OUTCOME_ORDER,
                    TEXT_SECONDARY, apply_style, done, draw_view, is_subcortical,
                    mri_regional, outcome_counts, outcomes, roi_values_to_vertices)

CLIP_PCT = (2, 98)
CLIP_LABEL = "2nd–98th percentile"


def group_means():
    """{outcome: Series indexed by cortical ROI} -- the cohort mean of the regional median ADC."""
    d = mri_regional().merge(outcomes(), on="patient_id")
    ctx = d[~d.roi.map(is_subcortical)]
    return {g: ctx[ctx.outcome == g].groupby("roi")[ADC_REGIONAL].mean()
            for g in OUTCOME_ORDER}


VIEWS = [("left", "lateral", "LH lateral"), ("left", "medial", "LH medial"),
         ("right", "lateral", "RH lateral"), ("right", "medial", "RH medial")]


def figure(mesh: str = MESH_FAST) -> None:
    means = group_means()
    n = outcome_counts()

    # One scale across both rows -- see the docstring. Clipped, not min-max.
    both = np.r_[means["good"].to_numpy(), means["poor"].to_numpy()]
    vmin, vmax = (float(v) for v in np.nanpercentile(both, CLIP_PCT))

    top, bottom, left, right = 0.84, 0.05, 0.05, 0.90
    row_h = (top - bottom) / len(OUTCOME_ORDER)
    fig = plt.figure(figsize=(13.0, 5.2))
    gs = fig.add_gridspec(len(OUTCOME_ORDER), len(VIEWS), wspace=0.0, hspace=0.0,
                          left=left, right=right, top=top, bottom=bottom)

    for r, grp in enumerate(OUTCOME_ORDER):
        vals = means[grp]
        # fill=NaN: the medial wall carries no regional value and must not be painted.
        arrays = roi_values_to_vertices(vals.to_dict(), mesh=mesh, fill=np.nan)
        for c, (hemi, view, vlabel) in enumerate(VIEWS):
            ax = fig.add_subplot(gs[r, c], projection="3d")
            draw_view(ax, hemi, view, arrays, CMAP_SEQ, vmin, vmax, mesh=mesh)
            if r == 0:
                ax.set_title(vlabel, fontsize=9, color=TEXT_SECONDARY, pad=-2)
        fig.text(0.018, top - (r + 0.5) * row_h,
                 f"{OUTCOME_LABEL[grp]} (n={n[grp]})", rotation=90, va="center",
                 ha="center", fontsize=10, color=OUTCOME_INK[grp], fontweight="medium")
        fig.text(right + 0.005, top - (r + 0.5) * row_h,
                 f"mean {vals.mean():.0f}\nSD {vals.std():.0f}", rotation=90, va="center",
                 ha="center", fontsize=7.5, color=TEXT_SECONDARY)

    h = (top - bottom) * 0.55
    cax = fig.add_axes([right + 0.055, bottom + (top - bottom - h) / 2, 0.012, h])
    cb = fig.colorbar(mpl.cm.ScalarMappable(
        cmap=CMAP_SEQ, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)), cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=2, color=TEXT_SECONDARY)
    cb.set_label(f"{ADC_LABEL} [{ADC_UNITS}]   ({ADC_DIRECTION})", fontsize=7.5,
                 color=TEXT_SECONDARY)

    fig.suptitle(
        f"Regional {ADC_LABEL} on the cortex, by neurological outcome — cohort mean per "
        "Desikan-Killiany region\n"
        f"{ADC_DIRECTION}, so the darker regions are the LESS injured ones · good "
        f"n={n['good']}, poor n={n['poor']} · 68 cortical ROIs · {mesh} + DK aparc.annot\n"
        f"one colour scale across both rows, clipped at the {CLIP_LABEL} · the medial wall "
        "is unpainted and the 18 subcortical regions are absent — they have no cortical "
        "surface · no test is computed",
        x=0.012, ha="left", fontsize=10.5, y=0.995, va="top")
    out = FIGURES / "mri_medADC_cortex_by_outcome.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    done(out)


def main(argv: list[str]) -> None:
    # fsaverage5 by default, the same mesh figure 1 uses -- the two are meant to be laid
    # side by side, and at block size the full-resolution mesh buys nothing visible while
    # costing ~10 minutes per figure instead of one.
    mesh = MESH_FAST
    for a in argv:
        if a.startswith("--mesh"):
            mesh = a.split("=", 1)[1] if "=" in a else "fsaverage"
    apply_style()
    n = outcome_counts()
    means = group_means()
    print(f"cohort: {sum(n.values())} retained patients ({n['good']} good / {n['poor']} "
          f"poor) · mesh {mesh}")
    print(f"cortical mean {ADC_LABEL}: good {means['good'].mean():.0f}, "
          f"poor {means['poor'].mean():.0f} {ADC_UNITS}  ({ADC_DIRECTION})")
    figure(mesh)


if __name__ == "__main__":
    main(sys.argv[1:])
