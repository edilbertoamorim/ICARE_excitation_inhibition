"""Figure 4 -- X, Y and Z against regional median ADC, computed inside each EEG cluster.

        python3 -W ignore 04_cluster_correlations.py              # compute (if needed) + draw
        python3 -W ignore 04_cluster_correlations.py --recompute  # force the statistics again
        python3 -W ignore 04_cluster_correlations.py X            # one coordinate only
        python3 -W ignore 04_cluster_correlations.py subcortical  # just the deep-gray panel

Writes one cortical figure per MFM coordinate plus one subcortical panel:

  clusters_medADC_<X|Y|Z>_corr.png    five blocks (four clusters + the unassigned group),
                                      TWO rows: every region's ρ on top, only the regions
                                      reaching p<0.05 underneath
  clusters_medADC_XYZ_subcortical.png the 18 regions with no cortical surface, as a heatmap

and caches the statistics in `data/derived/cluster_correlations_medADC.csv`.

The question
------------
Per region, across the patients *of one cluster*:

    x = the patient's median X / Y / Z in region r over hours 12-48
    y = the patient's median ADC in that same region r
    rho, p = spearmanr(x, y)                        # n = 7 to 14 patients

**The clusters are an outside input and were never fitted on MRI.** They come from a UMAP +
clustering of 542 ICARE patients on EEG alone; 45 of our 52 fall in it and the other 7 become
an explicit `unassigned` group, which is never dropped -- they are the most injured group in
the cohort, so excluding them would bias every cross-cluster comparison. Any MRI ordering
across the clusters is therefore an observed association, not a definition.

Why these figures colour by ρ and not by significance
-----------------------------------------------------
The five groups are 12 / 14 / 10 / 9 / 7 patients. At those n, |ρ| has to reach roughly
0.58 / 0.53 / 0.63 / 0.67 / 0.75 before p<0.05. `MIN_PAIRS` is lowered from the usual 10 to
**6** on purpose, so that cluster 3 (n=9) and the unassigned group (n=7) appear at all rather
than being silently dropped -- the cost being that those two are estimating a rank
correlation from 7-9 points, which is barely an estimate.

So a significance map at these n would be a map of what the study cannot see, and would read
as "no effect" where the truth is "no power". Every block is coloured by ρ and prints its own
n, and the second row shows how little of the first row clears p<0.05. **Never read the top
row alone**: a wall-to-wall coloured block routinely sits above a nearly empty one.

Ordering
--------
Blocks run left to right by **decreasing % good outcome**, and the `unassigned` group is
placed by that same rule rather than parked at the end, so position means the same thing in
every block of every cluster figure.

Nothing is FDR-corrected and the 86 regions are not independent tests.
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (ADC_DIRECTION, ADC_LABEL, ADC_REGIONAL, ALPHA, CMAP_DIV, DERIVED,
                    FEATURE_WINDOW_LABEL, FIG_DPI, FIGURES, MESH_FAST, MFM_PARAMS,
                    RHO_VMAX, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, apply_style,
                    block_title, cluster_table, done, draw_brain_block, is_subcortical,
                    make_block_grid, pretty_roi, regional_with_adc, rho_critical,
                    row_label, save, shared_colorbar, signed_log10p, spearman, suptitle)

TABLE = DERIVED / "cluster_correlations_medADC.csv"

# Deliberately BELOW the module-wide MIN_PAIRS of 10 -- see the docstring. The smallest group
# is 7 patients, so this is the floor that keeps every group on the figure.
CLUSTER_MIN_PAIRS = 6


# ==================================================================================
# statistics
# ==================================================================================
def compute() -> pd.DataFrame:
    """One Spearman per (cluster, coordinate, region). 5 x 3 x 86 = 1290 correlations."""
    d = regional_with_adc().merge(cluster_table()[["patient_id", "cluster"]],
                                  on="patient_id")
    rows = []
    for name, g in d.groupby("cluster"):
        n_pat = g.patient_id.nunique()
        rows += [{"cluster": name, "n_patients": n_pat, "feature": f, "roi": roi,
                  **spearman(cell[f], cell[ADC_REGIONAL], min_pairs=CLUSTER_MIN_PAIRS)}
                 for f in MFM_PARAMS for roi, cell in g.groupby("roi")]

    out = signed_log10p(pd.DataFrame(rows))
    out["subcortical"] = out.roi.map(is_subcortical)
    out = out.sort_values(["feature", "cluster", "roi"]).reset_index(drop=True)
    out.to_csv(TABLE, index=False)
    print(f"  wrote {TABLE.relative_to(DERIVED.parent.parent)}  {out.shape}")
    for c, g in out[out.rho.notna()].groupby("cluster"):
        n = int(g.n_patients.iloc[0])
        print(f"    cluster {c}: n={n} (|ρ| ≥ {rho_critical(n):.2f} for p<{ALPHA})  ·  "
              f"{int((g.p < ALPHA).sum())}/{len(g)} cells at p<{ALPHA}  ·  "
              f"median |ρ| {g.rho.abs().median():.2f}")
    return out


def table(recompute: bool = False) -> pd.DataFrame:
    if recompute or not TABLE.exists():
        return compute()
    print(f"  reusing {TABLE.relative_to(DERIVED.parent.parent)} (--recompute to rebuild)")
    return pd.read_csv(TABLE, dtype={"cluster": str})


def cluster_order() -> list[tuple[str, float, int]]:
    """(cluster, % good outcome, n) left to right by decreasing % good outcome.

    The one ordering rule for every cluster figure here, derived rather than hardcoded, so
    position means the same thing in all of them.
    """
    c = cluster_table()
    per = c.groupby("cluster").outcome
    pct_good = (per.apply(lambda s: (s == "good").sum()) / per.size() * 100.0)
    n = per.size()
    return [(k, float(pct_good[k]), int(n[k]))
            for k in sorted(pct_good.index, key=lambda k: -pct_good[k])]


def _name(cluster: str) -> str:
    return "no cluster" if cluster == "unassigned" else f"cluster {cluster}"


# ==================================================================================
# figures
# ==================================================================================
def figure_cortex(corr: pd.DataFrame, feat: str) -> None:
    """Two rows of the same numbers: all regions, then only those reaching p<ALPHA."""
    ok = corr[corr.rho.notna() & (corr.feature == feat) & (~corr.subcortical)]
    order = cluster_order()
    ncol = len(order)

    grid = make_block_grid(2 * ncol, ncol_max=ncol, block_w=3.9, block_h=2.75,
                           header_in=0.95, footer_in=0.70)
    for i, (cl, pct, n_pat) in enumerate(order):
        block = ok[ok.cluster == cl]
        sig = block[block.p < ALPHA]
        draw_brain_block(grid, i, dict(zip(block.roi, block.rho)), CMAP_DIV, -RHO_VMAX,
                         RHO_VMAX, mesh=MESH_FAST, fill=np.nan)
        block_title(grid, i, f"{_name(cl)}   (n={n_pat})",
                    f"{pct:.0f}% good outcome  ·  |ρ| ≥ {rho_critical(n_pat):.2f} for "
                    f"p<{ALPHA}")
        # Second row: same scale, but only the regions clearing ALPHA carry a value, so the
        # rest fall through to the plain background the medial wall already uses.
        draw_brain_block(grid, i + ncol, dict(zip(sig.roi, sig.rho)), CMAP_DIV, -RHO_VMAX,
                         RHO_VMAX, mesh=MESH_FAST, fill=np.nan)
        block_title(grid, i + ncol, "",
                    f"{len(sig)}/{len(block)} regions at p<{ALPHA} "
                    f"(chance ≈{int(round(ALPHA * len(block)))})")

    row_label(grid, 0, "all regions")
    row_label(grid, 1, f"p<{ALPHA} only")
    shared_colorbar(grid, CMAP_DIV, -RHO_VMAX, RHO_VMAX,
                    f"← lower {ADC_LABEL}      Spearman ρ of {feat} with the region's "
                    f"{ADC_LABEL}      higher {ADC_LABEL} →",
                    ticks=[-RHO_VMAX, -0.3, 0.0, 0.3, RHO_VMAX],
                    ticklabels=["−0.6", "−0.3", "0", "+0.3", "+0.6"], width=0.42)
    suptitle(
        grid,
        f"{feat} vs regional {ADC_LABEL}, within each EEG cluster — one Spearman per region "
        f"across that cluster's patients ({FEATURE_WINDOW_LABEL} feature value)\n"
        f"{ADC_DIRECTION} · blocks left to right by decreasing % good outcome · the clusters "
        "are EEG-derived and were never fitted on MRI, so any ordering here is an observed "
        "association\n"
        "coloured by ρ, NOT by significance: at n=7–14 a significance map is a map of what "
        "the study cannot see · never read the top row without the bottom one",
        fontsize=10.5)
    out = FIGURES / f"clusters_medADC_{feat}_corr.png"
    save(grid, out, FIG_DPI)
    done(out)


def figure_subcortical(corr: pd.DataFrame) -> None:
    """The 18 regions no cortical surface can show, as a (region x cluster) heatmap.

    Kept out of the surfaces rather than folded in: the thalamus and the rest of the deep
    gray are invisible on a pial mesh, and this cohort's ADC injury is largest there, so a
    surface-only treatment would silently drop the most affected tissue in the brain.
    """
    ok = corr[corr.rho.notna() & corr.subcortical]
    order = cluster_order()
    clusters = [c for c, _, _ in order]
    n_by = {c: n for c, _, n in order}
    feats = [f for f in MFM_PARAMS if f in set(ok.feature)]
    rois = sorted(ok.roi.unique())
    cols = [(f, c) for f in feats for c in clusters]

    piv = ok.set_index(["feature", "cluster", "roi"])
    mat = np.array([[piv.rho.get((f, c, r), np.nan) for f, c in cols] for r in rois])
    pv = np.array([[piv.p.get((f, c, r), np.nan) for f, c in cols] for r in rois])

    fig, ax = plt.subplots(figsize=(0.82 * len(cols) + 3.4, 0.32 * len(rois) + 2.4),
                           layout="constrained")
    im = ax.imshow(mat, cmap=CMAP_DIV, vmin=-RHO_VMAX, vmax=RHO_VMAX, aspect="auto")
    for i in range(len(rois)):
        for j in range(len(cols)):
            if np.isfinite(mat[i, j]):
                star = "*" if pv[i, j] < ALPHA else ""
                ax.text(j, i, f"{mat[i, j]:+.2f}{star}", ha="center", va="center",
                        fontsize=7, color=TEXT_PRIMARY,
                        fontweight="medium" if pv[i, j] < ALPHA else "normal")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{f}\n{_name(c)}\nn={n_by[c]}" for f, c in cols], fontsize=7.5)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(len(rois)), [pretty_roi(r) for r in rois], fontsize=7.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(len(cols) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(rois) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    # A heavier rule where the coordinate changes, so each feature reads as its own block.
    for k in range(1, len(feats)):
        ax.axvline(k * len(clusters) - 0.5, color=TEXT_PRIMARY, linewidth=1.4)

    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label(f"Spearman ρ with the region's {ADC_LABEL}  ({ADC_DIRECTION})", fontsize=8,
                 color=TEXT_SECONDARY)
    cb.ax.tick_params(labelsize=7, length=2)

    fig.suptitle(
        f"Subcortical regions — {', '.join(feats)} vs regional {ADC_LABEL}, within each EEG "
        f"cluster   (ρ, * marks p<{ALPHA})\n"
        "clusters left to right by decreasing % good outcome · the 18 regions with no "
        f"cortical surface · at n=7–14, |ρ| must reach {rho_critical(14):.2f}–"
        f"{rho_critical(7):.2f} for p<{ALPHA}, so read ρ and not the stars",
        ha="left", x=0.0, fontsize=10.5)
    out = FIGURES / f"clusters_medADC_{''.join(feats)}_subcortical.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    done(out)


def main(argv: list[str]) -> None:
    names = [a for a in argv if not a.startswith("--")]
    allowed = MFM_PARAMS + ["subcortical"]
    unknown = set(names) - set(allowed)
    if unknown:
        raise SystemExit(f"unknown target {sorted(unknown)}; pick from {allowed}")
    apply_style()
    order = cluster_order()
    print(f"clusters (left to right by % good outcome): "
          + " · ".join(f"{_name(c)} n={n} ({pct:.0f}% good)" for c, pct, n in order))
    print(f"target: regional {ADC_LABEL} ({ADC_DIRECTION}) · "
          f"{', '.join(MFM_PARAMS)} · min_pairs={CLUSTER_MIN_PAIRS}")
    corr = table(recompute="--recompute" in argv)

    for feat in [f for f in MFM_PARAMS if not names or f in names]:
        figure_cortex(corr, feat)
    if not names or "subcortical" in names:
        figure_subcortical(corr)


if __name__ == "__main__":
    main(sys.argv[1:])
