"""Stage 2 -- draw the paper figure from the tables `gbtm.py` wrote.

This module only READS results/. It fits nothing and computes nothing beyond cosmetics, so
changing a colour never re-runs the model.

`figures/GBTM_MRI_<tag>.png`, two rows, laid out to sit as close to square as the blocks
allow and carrying no figure title (the caption supplies it):

  row 1   the estimated class trajectory of every fitted variable, one wide panel each, with
          the two classes' outcome donuts -- n, and the good/poor split -- beside them;
  row 2   the raw hourly X, Y and Z of all 592 patients stacked in ONE narrow column on the
          left, and the cortical map of the between-class difference in median ADC filling
          the rest of the row.

The raw block departs from the usual spaghetti plot on purpose: all patients share one panel
per variable and the lines are coloured by CLASS, not by outcome, because the question this
figure answers is how far apart the two classes are in the raw data. Z is drawn even though
no model here fits it, as the companion series.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import brain
import config as cfg

# The class mean shares its colour with the patient lines underneath it, so it gets a thin
# white casing to stay readable at that line density. Thickness alone was not enough.
MEAN_W = 2.4
MEAN_STROKE = [pe.withStroke(linewidth=MEAN_W + 1.8, foreground="white")]
RC = {"font.family": "sans-serif",
      "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
      "axes.linewidth": 1.0, "figure.facecolor": "white", "axes.facecolor": "white",
      "savefig.facecolor": "white"}


def class_colors(classes) -> dict:
    """Class number -> line colour. Classes are numbered by decreasing % good, so the first
    is the blue (more-good) one and the last is the orange (more-poor) one."""
    classes = list(classes)
    if len(classes) == 1:
        return {classes[0]: cfg.COLOR_GOOD}
    out = {classes[0]: cfg.COLOR_GOOD, classes[-1]: cfg.COLOR_POOR}
    for c in classes[1:-1]:
        out[c] = cfg.COLOR_MID
    return out


def shared_ylim(values) -> tuple[float, float]:
    """3% padding, and never below zero for a series that is non-negative."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not v.size:
        return (0.0, 1.0)
    lo, hi = float(v.min()), float(v.max())
    pad = 0.03 * (hi - lo) if hi > lo else 0.05 * max(abs(hi), 1.0)
    lo, hi = lo - pad, hi + pad
    return (max(lo, 0.0) if float(v.min()) >= 0 else lo), hi


def donut(ax, title: str, n_good: int, n_poor: int, subtitle: str | None = None,
          title_color: str = "black", compact: bool = False) -> None:
    """The outcome ring and its three text lines. `compact` shrinks it for a narrow column."""
    ax.axis("off")
    f = 0.82 if compact else 1.0
    tx = 0.34 if compact else 0.38
    total = n_good + n_poor
    pct = 100.0 * n_good / total if total else 0.0
    inset = ax.inset_axes([0.0, 0.12, 0.30 if compact else 0.32, 0.76])
    inset.pie([n_good, n_poor], colors=[cfg.COLOR_GOOD, cfg.COLOR_POOR], startangle=90,
              counterclock=False,
              wedgeprops={"width": 0.48, "edgecolor": "white", "linewidth": 0.8})
    inset.text(0.0, 0.0, f"{pct:.0f}%", ha="center", va="center", fontsize=10 * f,
               fontweight="bold", color=cfg.COLOR_GOOD)
    ax.text(tx, 0.86, title, transform=ax.transAxes, fontsize=11 * f, fontweight="bold",
            va="center", color=title_color)
    ax.text(tx, 0.64, rf"$n$ = {total}", transform=ax.transAxes, fontsize=10 * f, va="center")
    ax.text(tx, 0.44, f"Good: {n_good} ({pct:.0f}%)" if compact
            else f"Good outcome: {n_good} ({pct:.0f}%)", transform=ax.transAxes,
            fontsize=9.5 * f, color=cfg.COLOR_GOOD, fontweight="bold", va="center")
    ax.text(tx, 0.24, f"Poor: {n_poor} ({100 - pct:.0f}%)" if compact
            else f"Poor outcome: {n_poor} ({100 - pct:.0f}%)", transform=ax.transAxes,
            fontsize=9.5 * f, color=cfg.COLOR_POOR, fontweight="bold", va="center")
    if subtitle:
        ax.text(tx, 0.05, subtitle, transform=ax.transAxes, fontsize=8 * f, color="#444444",
                va="center")


def _read(name: str, **kw) -> pd.DataFrame:
    path = cfg.RESULTS / name
    if not path.exists():
        raise SystemExit(f"{path} is missing -- run `python3 gbtm.py` (or run_all.py) first")
    return pd.read_csv(path, **kw)


def draw(tag: str | None = None, out_name: str | None = None):
    tag = cfg.check(tag)
    s = _read(f"subtype_{tag}_summary.csv").set_index("subtype")
    curves = _read(f"subtype_{tag}_curves.csv")
    hourly = _read(f"subtype_{tag}_hourly.csv", dtype={"patient_id": str})
    stats_all = _read(f"mri_roi_subtype_{tag}_means.csv")
    classes = sorted(s.index)
    if len(classes) != 2:
        raise SystemExit(f"this figure is built for 2 classes, got {classes}")
    colors = class_colors(classes)
    fit_vars = list(dict.fromkeys(curves.variable))

    # ---- the between-class difference in regional median ADC, cortex only
    stats = stats_all[(stats_all.metric == "medADC") & (~stats_all.subcortical)]
    wide = stats.pivot_table(index="roi", columns="subtype", values="mean")
    diff = wide[classes[-1]] - wide[classes[0]]
    # The two classes differ mostly by a GLOBAL offset, so a scale set by the extremes leaves
    # the bulk of regions in one flat shade. Clipping at the 85th percentile of |difference|
    # saturates the few strongest regions and spends the range on separating the rest; the
    # sign agreement and the saturated count are printed under the block, so the map is read
    # for WHERE the gap is widest rather than for its sign.
    dmax = float(np.percentile(np.abs(diff), 85)) or float(np.abs(diff).max())
    n_clipped = int((np.abs(diff) > dmax).sum())
    same_sign = 100.0 * float((np.sign(diff) == np.sign(diff.mean())).mean())

    n_traj = len(fit_vars)
    fig_w, fig_h = 12.8, 12.2
    with mpl.rc_context(RC):
        fig = plt.figure(figsize=(fig_w, fig_h))

        # ---- row 1, left: the estimated class trajectories
        traj_left, row1_top, row1_bottom = 0.055, 0.945, 0.615
        traj_right = traj_left + min(0.70, 0.325 * n_traj)
        donut_left = traj_right + 0.03
        donut_right = min(donut_left + 0.30, 0.995)
        # The donut column takes whatever the trajectories leave, so a two-variable model
        # leaves it narrow enough that the full-size text can overrun it.
        tight = (donut_right - donut_left) < 0.24
        gs = fig.add_gridspec(1, n_traj, left=traj_left, right=traj_right, top=row1_top,
                              bottom=row1_bottom, wspace=0.34)
        for v_idx, var in enumerate(fit_vars):
            ax = fig.add_subplot(gs[0, v_idx])
            for c in classes:
                g = curves[(curves.subtype == c) & (curves.variable == var)].sort_values("hour")
                se = g["se"] if "se" in g else (g["hi"] - g["lo"]) / (2 * 1.96)
                ax.fill_between(g.hour, g["fit"] - se, g["fit"] + se, color=colors[c],
                                alpha=0.30, linewidth=0, zorder=2)
                ax.plot(g.hour, g["fit"], color=colors[c], linewidth=2.8, zorder=3,
                        path_effects=MEAN_STROKE)
            ax.set_xlim(g.hour.min(), g.hour.max())
            ax.set_xticks(np.linspace(g.hour.min(), g.hour.max(), 5))
            ax.set_xlabel(cfg.X_LABEL, fontsize=12)
            ax.set_ylabel(cfg.Y_LABELS.get(var, var), fontsize=12)
            ax.tick_params(labelsize=10.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if v_idx == 0:
                ax.set_title("Estimated class trajectories", fontsize=12.5, loc="left", pad=8)

        # ---- row 1, right: one outcome donut per class
        gs_d = fig.add_gridspec(len(classes), 1, left=donut_left, right=donut_right,
                                top=row1_top + 0.02, bottom=row1_bottom - 0.005, hspace=0.30)
        for slot, c in enumerate(classes):
            r = s.loc[c]
            donut(fig.add_subplot(gs_d[slot, 0]), r.label, int(r.n_good), int(r.n_poor),
                  title_color=colors[c], subtitle=f"class {c} · n={int(r.n_patients)}",
                  compact=tight)

        # ---- row 2, left: the raw hourly series, every patient, coloured by class
        row2_top, raw_bottom, raw_right = 0.545, 0.082, 0.34
        hours = np.sort(hourly.hour.unique())
        cls = hourly.groupby("patient_id").subtype.first()
        gs_c = fig.add_gridspec(len(cfg.PLOT_VARS), 1, left=traj_left, right=raw_right,
                                top=row2_top, bottom=raw_bottom, hspace=0.28)
        for v_idx, var in enumerate(cfg.PLOT_VARS):
            ax = fig.add_subplot(gs_c[v_idx, 0])
            series = (hourly.pivot_table(index="patient_id", columns="hour", values=var)
                      .reindex(columns=hours))
            for c in classes:
                block = series.loc[cls[cls == c].index].to_numpy()
                for row in block:
                    ax.plot(hours, row, color=colors[c], alpha=0.13, linewidth=0.4, zorder=1)
                mean = np.nanmean(block, axis=0)
                n = np.sum(np.isfinite(block), axis=0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    se = np.nanstd(block, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
                se = np.where(n > 1, se, np.nan)
                ax.fill_between(hours, mean - se, mean + se, color=colors[c], alpha=0.35,
                                linewidth=0, zorder=2)
                ax.plot(hours, mean, color=colors[c], linewidth=MEAN_W, zorder=3,
                        path_effects=MEAN_STROKE)
            ax.set_xlim(hours.min(), hours.max())
            ax.set_ylim(*shared_ylim(series.to_numpy()))
            ax.set_xticks(np.linspace(hours.min(), hours.max(), 5))
            # Only the bottom panel carries the time axis: three copies of it in a narrow
            # column is three labels saying the same thing.
            if v_idx == len(cfg.PLOT_VARS) - 1:
                ax.set_xlabel(cfg.X_LABEL, fontsize=11.5)
            else:
                ax.set_xticklabels([])
            ax.set_ylabel(cfg.Y_LABELS.get(var, var), fontsize=11.5)
            ax.tick_params(labelsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if var not in fit_vars:
                ax.set_facecolor("#fbfbfb")        # companion series, not fitted
            if v_idx == 0:
                ax.set_title(f"Observed hourly series, all {int(s.n_patients.sum())} patients",
                             fontsize=12, loc="left", pad=8)

        # ---- row 2, right: the difference map
        block = brain.Block(fig=fig, left=raw_right + 0.06, right=0.98,
                            top=row2_top + 0.013, bottom=0.018)
        brain.draw(block, dict(zip(wide.index, diff)), brain.CMAP_DIFF, -dmax, dmax,
                   fill=np.nan)
        brain.colorbar(block, brain.CMAP_DIFF, -dmax, dmax,
                       f"Δ regional median ADC  (class {classes[-1]} − class {classes[0]})")
        cx = block.left + block.w / 2
        fig.text(cx, block.bottom + block.h * 0.975, "Difference in cortical median ADC",
                 ha="center", va="center", fontsize=13, fontweight="bold")
        fig.text(cx, block.bottom + block.h * 0.928,
                 f"red = higher in class {classes[-1]}, blue = higher in class {classes[0]}"
                 f"  ({int(s.n_with_mri.sum())} scans · same sign in {same_sign:.0f}% of "
                 f"regions · {n_clipped} saturated)",
                 ha="center", va="center", fontsize=10, color="#444444")

        # The class key sits under the raw column: the centre of the bottom edge belongs to
        # the brain block's colourbar.
        fig.legend(handles=[plt.Line2D([], [], color=colors[c], lw=MEAN_W,
                                       label=f"class {c} · {s.loc[c, 'label']}")
                            for c in classes],
                   loc="lower left", ncol=1, frameon=False, fontsize=11,
                   bbox_to_anchor=(traj_left - 0.01, -0.002))

        cfg.FIGURES.mkdir(parents=True, exist_ok=True)
        out = cfg.FIGURES / (out_name or f"GBTM_MRI_{tag}.png")
        fig.savefig(out, dpi=cfg.DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print(f"  wrote {out.relative_to(cfg.ROOT)}")
    return out


if __name__ == "__main__":
    draw()
