"""Figure 3 -- every feature's correlation with whole-brain median ADC, hour by hour.

        python3 -W ignore 03_correlations_overtime.py              # compute (if needed) + draw
        python3 -W ignore 03_correlations_overtime.py --recompute  # force the statistics again
        python3 -W ignore 03_correlations_overtime.py rho          # just one figure

Writes three figures, all from one table:

  patientovertime_medADC_rho.png       one panel per feature: the whole cohort and the two
                                       outcome groups as curves over hours 12-72
  patientovertime_medADC_heatmap.png   the same numbers as feature x hour blocks, blanked
                                       where p>=0.05, with a recording-coverage strip
  patientovertime_medADC_basis.png     the whole-cohort curve on the interpolated series
                                       against the observed-only one -- the sensitivity check

and caches the statistics in `data/derived/overtime_correlations_medADC.csv`, which takes
about a minute to build and is reused on later runs.

The question
------------
One number per patient on each side, correlated across the 52 patients, separately at every
hour:

    x = the patient's ROI-averaged feature value AT THAT HOUR
    y = the patient's whole-brain median ADC            (fixed -- one MRI scan per patient)
    rho, p = spearmanr(x, y)                            # n = 52, or 27 / 25 by outcome

Only x moves with time. The MRI is a single scan, so these curves are entirely about *when*
the EEG carries the signal, not about the MRI changing. Spearman rather than Pearson: n is 52
at most and several of the features are strongly skewed.

The vertical rule at hour 48 on every panel is where the 12-48 h feature window used
elsewhere in this folder ends -- i.e. how much of each curve a single window median would
have seen.

The two bases, which are not a detail
-------------------------------------
Hourly coverage is partial: recordings start anywhere from hour 12 to 46 and end from 29 to
72. So the correlation is computed twice, carried in the `basis` column:

  interpolated  all 52 patients at every hour, using the Kalman-filled series. n is constant
                across hours, which is the only thing that makes the curve comparable from
                hour to hour -- but the early and late hours lean on extrapolation, and a
                random-walk Kalman smoother is close to holding the last observed value, so
                those ends carry mid-window information wearing an early/late label.
  observed      only the patients actually recording at that hour (14-38 of 52). Honest about
                what was measured, at the cost of an n that moves with the hour, so a change
                in the curve can be a change in the cohort rather than in the signal.

The first two figures use `interpolated`; `_basis` is the explicit check on it. Neither basis
is right on its own -- where the two agree, the timing is real.

Reading rules, which are part of the figures and not decoration
---------------------------------------------------------------
* **Neighbouring hours are not independent tests.** The series are Kalman-smoothed and the
  MRI target never moves, so a run of significant hours is ONE finding, not a dozen. Nothing
  is corrected for multiplicity: read runs, never a single hour.
* **Splitting by outcome halves n** (52 -> 27 / 25), raising the |rho| needed for p<0.05 from
  0.27 to 0.38 / 0.40. A group whose dots disappear has less power; it has not been shown to
  differ. No between-group test is computed here.
* Higher median ADC means HEALTHIER tissue, so a positive rho means the feature is high in
  the patients whose brains are less injured.
"""

from __future__ import annotations

import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (ADC_DIRECTION, ADC_GLOBAL, ADC_LABEL, ALPHA, CMAP_SIG, CORR_VMAX,
                    DERIVED, FEATURE_FAMILIES, FEATURE_WINDOW, FEATURES, FEATURES_HOURLY,
                    FIG_DPI, FIGURES, GRID, OUTCOME_INK, OUTCOME_ORDER, SEQUENTIAL_BLUE,
                    TEXT_PRIMARY, TEXT_SECONDARY, TRAJECTORY_HOURS, X_LABEL, apply_style,
                    done, mri_global, outcomes, rho_critical, signed_log10p, spearman)

TABLE = DERIVED / "overtime_correlations_medADC.csv"
HOURS = list(range(TRAJECTORY_HOURS[0], TRAJECTORY_HOURS[1] + 1))

GROUP_ORDER = ["all"] + OUTCOME_ORDER
GROUP_LABEL = {"all": "whole cohort", "good": "good outcome", "poor": "poor outcome"}

# `all` is the reference the two outcome groups are read against, so it is neutral in colour
# and dotted. The two outcome curves are BOTH solid: these panels are read as shapes over
# time, and a dash pattern breaks the shape up. It is the one place in this folder where good
# and poor are told apart by colour alone.
GROUP_STYLE = {"all": dict(color=TEXT_SECONDARY, dashes=(1, 1.6)),
               "good": dict(color=OUTCOME_INK["good"], dashes=(None, None)),
               "poor": dict(color=OUTCOME_INK["poor"], dashes=(None, None))}

WB_LABEL = f"whole-brain {ADC_LABEL}"


# ==================================================================================
# statistics
# ==================================================================================
def compute() -> pd.DataFrame:
    """One Spearman per (feature, group, basis, hour). ~7k rank correlations, under a minute."""
    h = pd.read_csv(FEATURES_HOURLY, dtype={"patient_id": str})
    h = h[h.Hour.between(*TRAJECTORY_HOURS)]
    d = h.merge(outcomes(), on="patient_id").merge(mri_global(), on="patient_id")

    rows = []
    for basis, sub in (("interpolated", d), ("observed", d[d.observed])):
        groups = {"all": sub, **{g: sub[sub.outcome == g] for g in OUTCOME_ORDER}}
        for name, g in groups.items():
            for (feat, hour), cell in g.groupby(["Feature_Name", "Hour"]):
                rows.append({"feature": feat, "group": name, "basis": basis,
                             "hour": int(hour), **spearman(cell.value, cell[ADC_GLOBAL])})
    out = signed_log10p(pd.DataFrame(rows))
    out.to_csv(TABLE, index=False)
    print(f"  wrote {TABLE.relative_to(DERIVED.parent.parent)}  {out.shape}")
    for (b,), g in out.groupby(["basis"]):
        print(f"    {b}: {int((g.p < ALPHA).sum())}/{int(g.p.notna().sum())} cells at "
              f"p<{ALPHA} (n {int(g.n.min())}-{int(g.n.max())})")
    return out


def table(recompute: bool = False) -> pd.DataFrame:
    if recompute or not TABLE.exists():
        return compute()
    print(f"  reusing {TABLE.relative_to(DERIVED.parent.parent)} (--recompute to rebuild)")
    return pd.read_csv(TABLE)


def _family_order(available: set[str]) -> tuple[list[str], list[tuple[str, int, int]]]:
    """Feature order, plus (family title, start, stop) spans.

    Ordering by FEATURE_FAMILIES rather than by |rho| keeps the rows structured the same way
    in all three figures, and stops a figure from silently ranking features by the very
    statistic it is displaying.
    """
    feats, spans = [], []
    for spec in FEATURE_FAMILIES.values():
        block = [f for f in spec["features"] if f in available]
        if block:
            spans.append((spec["title"], len(feats), len(feats) + len(block)))
            feats += block
    feats += sorted(available - set(feats))       # anything not claimed by a family
    return feats, spans


def _series(corr: pd.DataFrame, feat: str, group: str, basis: str) -> pd.DataFrame:
    """One (feature, group, basis) curve on the full hour grid, NaN where the cell is absent.

    The four FOOOF features have no observed patient at hour 72, so the `observed` basis is
    not rectangular; reindexing here saves every caller from having to notice.
    """
    g = corr[(corr.feature == feat) & (corr.group == group) & (corr.basis == basis)]
    return g.set_index("hour").reindex(HOURS)


# ==================================================================================
# the curve grid, shared by `rho` and `basis`
# ==================================================================================
def _grid(feats: list[str], panel_h: float = 1.95):
    """One panel per feature, 4 to a row, plus ONE spare slot for the legend.

    A figure-level legend below the panels lands on the x labels -- constrained layout puts
    an `outside lower` legend and the bottom row's xlabel in the same place -- so the legend
    goes in the spare slot instead, which also uses dead space.
    """
    ncol = 4
    nrow = int(np.ceil((len(feats) + 1) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, panel_h * nrow),
                            squeeze=False, layout="constrained", sharex=True)
    return fig, axes, nrow, ncol


def _frame(ax, feat: str, vmax: float) -> None:
    """Everything that is the same in every panel: zero line, window marker, limits."""
    ax.axhline(0.0, color=TEXT_SECONDARY, linewidth=0.8, zorder=1)
    ax.axvline(FEATURE_WINDOW[1], color=GRID, linewidth=1.4, zorder=1)
    ax.set_title(feat, loc="left")
    ax.set_xlim(*TRAJECTORY_HOURS)
    ax.set_ylim(-vmax, vmax)
    ax.grid(alpha=0.7)
    ax.set_axisbelow(True)


def _finish(fig, axes, nrow: int, ncol: int, n_feats: int, handles, title: str) -> None:
    flat = axes.ravel()
    for r in range(nrow):
        for c in range(1, ncol):
            axes[r][c].tick_params(labelleft=False)      # the rho scale is shared
        axes[r][0].set_ylabel("Spearman ρ")
    # x label on the last panel of each column that actually holds a feature -- the legend
    # slot is not one, so it must not be what the search lands on.
    for c in range(ncol):
        rows = [r for r in range(nrow) if r * ncol + c < n_feats]
        if rows:
            axes[rows[-1]][c].set_xlabel(X_LABEL)

    legend_ax = flat[n_feats]
    legend_ax.axis("off")
    legend_ax.legend(handles=handles, loc="center left", frameon=False, handlelength=2.6,
                     fontsize=8.5)
    for ax in flat[n_feats + 1:]:
        ax.set_visible(False)
    fig.suptitle(title, ha="left", x=0.0, fontsize=10.5)


def figure_rho(corr: pd.DataFrame) -> None:
    """The interpolated basis, whole cohort and both outcome groups, one panel per feature."""
    corr = corr[corr.basis == "interpolated"]
    feats, _ = _family_order(set(corr.feature))
    ns = {g: int(corr[corr.group == g].n.max()) for g in GROUP_ORDER}
    vmax = float(np.ceil(corr.rho.abs().max() * 10) / 10)

    fig, axes, nrow, ncol = _grid(feats)
    for ax, feat in zip(axes.ravel(), feats):
        _frame(ax, feat, vmax)
        for grp in GROUP_ORDER:
            g = _series(corr, feat, grp, "interpolated")
            style = GROUP_STYLE[grp]
            ax.plot(g.index, g.rho, linewidth=1.5, zorder=3, **style)
            # A dot on every hour clearing p<0.05. n is fixed within a group, so this is a
            # |rho| threshold: it marks where the curve is strong, not where it moved.
            sig = g[g.p < ALPHA]
            ax.scatter(sig.index, sig.rho, s=11, color=style["color"], zorder=4,
                       linewidths=0)

    handles = [plt.Line2D([], [], label=f"{GROUP_LABEL[g]} (n={ns[g]}, |ρ|≥"
                                        f"{rho_critical(ns[g]):.2f} for p<{ALPHA})",
                          linewidth=1.5, **GROUP_STYLE[g]) for g in GROUP_ORDER]
    _finish(fig, axes, nrow, ncol, len(feats), handles,
            f"Correlation with {WB_LABEL} over time   (Spearman ρ · dots mark p<{ALPHA} · "
            f"{ADC_DIRECTION} · rule at hour {FEATURE_WINDOW[1]})")
    out = FIGURES / "patientovertime_medADC_rho.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    done(out)


def figure_basis(corr: pd.DataFrame) -> None:
    """Sensitivity check: the whole-cohort curve, interpolated vs observed patients only."""
    corr = corr[corr.group == "all"]
    feats, _ = _family_order(set(corr.feature))
    vmax = float(np.ceil(corr.rho.abs().max() * 10) / 10)
    # Both solid, and a bigger significance dot than the other figures: this pair is read as
    # "do the two curves agree", which a dash pattern only gets in the way of.
    bases = {"interpolated": dict(color=TEXT_SECONDARY, dashes=(None, None)),
             "observed": dict(color=OUTCOME_INK["poor"], dashes=(None, None))}
    n_obs = corr[corr.basis == "observed"].n

    fig, axes, nrow, ncol = _grid(feats)
    for ax, feat in zip(axes.ravel(), feats):
        _frame(ax, feat, vmax)
        for basis, style in bases.items():
            g = _series(corr, feat, "all", basis)
            ax.plot(g.index, g.rho, linewidth=1.4, zorder=3, **style)
            sig = g[g.p < ALPHA]
            ax.scatter(sig.index, sig.rho, s=26, color=style["color"], zorder=4,
                       linewidths=0)

    handles = [plt.Line2D([], [], label="interpolated (all 52 at every hour)", linewidth=1.4,
                          **bases["interpolated"]),
               plt.Line2D([], [], label=f"observed only ({int(n_obs.min())}–"
                                        f"{int(n_obs.max())} per hour)", linewidth=1.4,
                          **bases["observed"])]
    _finish(fig, axes, nrow, ncol, len(feats), handles,
            f"Correlation with {WB_LABEL} over time — interpolated vs observed-only   "
            f"(Spearman ρ · dots mark p<{ALPHA})")
    out = FIGURES / "patientovertime_medADC_basis.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    done(out)


def figure_heatmap(corr: pd.DataFrame) -> None:
    """One feature x hour block per group, blanked where p>=ALPHA, for reading bands.

    A coloured stripe is a stretch of hours in which the feature tracked injury. The strip
    underneath each block is the recording coverage: the correlation itself is computed on
    the interpolated series, and this is what says how much of a given hour was actually
    measured.
    """
    feats, spans = _family_order(set(corr.feature))
    interp = corr[corr.basis == "interpolated"]
    obs = corr[corr.basis == "observed"]
    ns = {g: int(interp[interp.group == g].n.max()) for g in GROUP_ORDER}

    # Explicit geometry, not constrained layout: the family names sit outside the feature tick
    # labels, and constrained layout does not reserve room for text drawn with clip_on=False,
    # so the left margin has to be set here or the two overprint.
    fig = plt.figure(figsize=(15.0, 6.6))
    gs = fig.add_gridspec(2, len(GROUP_ORDER), height_ratios=[len(feats), 4.2], left=0.225,
                          right=0.925, top=0.90, bottom=0.105, hspace=0.06, wspace=0.045)

    for c, grp in enumerate(GROUP_ORDER):
        signed = np.vstack([_series(interp, f, grp, "interpolated").signed_log10p
                            for f in feats])
        pval = np.vstack([_series(interp, f, grp, "interpolated").p for f in feats])
        shown = np.clip(np.where(pval < ALPHA, signed, 0.0), -CORR_VMAX, CORR_VMAX)

        ax = fig.add_subplot(gs[0, c])
        ax.imshow(np.nan_to_num(shown), cmap=CMAP_SIG, vmin=-CORR_VMAX, vmax=CORR_VMAX,
                  aspect="auto", interpolation="nearest",
                  extent=(HOURS[0] - 0.5, HOURS[-1] + 0.5, len(feats) - 0.5, -0.5))
        ax.set_title(f"{GROUP_LABEL[grp]}  (n={ns[grp]}, |ρ|≥{rho_critical(ns[grp]):.2f})",
                     fontsize=9, loc="left",
                     color=TEXT_PRIMARY if grp == "all" else OUTCOME_INK[grp])
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(feats if c == 0 else [], fontsize=7.5)
        ax.tick_params(length=0, labelbottom=False)
        for s in ax.spines.values():
            s.set_visible(False)
        for _, i0, _ in spans:                     # a rule between feature families
            if i0:
                ax.axhline(i0 - 0.5, color=TEXT_SECONDARY, linewidth=0.7)
        if c == 0:
            for title, i0, _ in spans:
                ax.text(-0.42, i0, title, transform=ax.get_yaxis_transform(), ha="right",
                        va="center", fontsize=7, color=TEXT_SECONDARY, fontweight="medium",
                        clip_on=False)

        # Coverage is identical across features, so one strip per block says it for the block.
        cov = _series(obs, feats[0], grp, "observed").n
        axc = fig.add_subplot(gs[1, c])
        axc.fill_between(cov.index, 0, cov.to_numpy(), color=SEQUENTIAL_BLUE[2],
                         edgecolor=SEQUENTIAL_BLUE[5], linewidth=0.9)
        axc.set_xlim(HOURS[0] - 0.5, HOURS[-1] + 0.5)
        axc.set_ylim(0, ns[grp])
        axc.set_xlabel(X_LABEL)
        axc.set_yticks([0, ns[grp]])
        axc.tick_params(labelsize=7)
        if c == 0:
            axc.set_ylabel("patients\nrecording", fontsize=7.5)
        else:
            axc.tick_params(left=False, labelleft=False)
        axc.grid(False)
        for s in axc.spines.values():
            s.set_visible(False)

    sig = np.log10(1 / ALPHA)
    cax = fig.add_axes([0.945, 0.30, 0.011, 0.50])
    cb = fig.colorbar(mpl.cm.ScalarMappable(
        cmap=CMAP_SIG, norm=mpl.colors.Normalize(-CORR_VMAX, CORR_VMAX)), cax=cax,
        orientation="vertical")
    cb.outline.set_visible(False)
    cb.set_ticks([-CORR_VMAX, -2, -sig, 0, sig, 2, CORR_VMAX])
    cb.set_ticklabels(["0.001", "0.01", "0.05", "n.s.", "0.05", "0.01", "0.001"])
    cb.ax.tick_params(labelsize=7, length=2)
    cb.set_label("← negative      Spearman p      positive →", fontsize=7.5,
                 color=TEXT_SECONDARY)

    fig.suptitle(f"Correlation with {WB_LABEL} over time   (coloured only where p<{ALPHA} · "
                 f"{ADC_DIRECTION})", ha="left", x=0.01, y=0.975, fontsize=10.5)
    out = FIGURES / "patientovertime_medADC_heatmap.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    done(out)


FIGURES_BY_NAME = {"rho": figure_rho, "heatmap": figure_heatmap, "basis": figure_basis}


def _runs(hours: list[int]) -> str:
    """`[38,39,40,52]` -> `38–40, 52`. Significant hours are only meaningful as runs."""
    out, start = [], hours[0]
    for a, b in zip(hours, hours[1:] + [None]):
        if b != a + 1:
            out.append(f"{start}–{a}" if a != start else f"{start}")
            start = b
    return ", ".join(out)


def _headline(corr: pd.DataFrame) -> None:
    """Print what the figures are about to show, so the run is self-documenting."""
    sub = corr[(corr.basis == "interpolated") & (corr.group == "all")]
    n = int(sub.n.max())
    print(f"\nwhole cohort (n={n}) vs {WB_LABEL}, interpolated basis — "
          f"|ρ| ≥ {rho_critical(n):.2f} for p<{ALPHA}:")
    any_sig = False
    for feat, g in sub.groupby("feature"):
        sig = g[g.p < ALPHA]
        if sig.empty:
            continue
        any_sig = True
        print(f"  {feat:<20s} {len(sig):>2d}/{len(g)} hours  ρ {g.rho.min():+.2f}.."
              f"{g.rho.max():+.2f}  ·  hours {_runs(sorted(sig.hour))}")
    if not any_sig:
        print("  no feature clears p<0.05 at any hour")
    print("  neighbouring hours are not independent tests — read runs, not single hours")


def main(argv: list[str]) -> None:
    names = [a for a in argv if not a.startswith("--")]
    unknown = set(names) - set(FIGURES_BY_NAME)
    if unknown:
        raise SystemExit(f"unknown figure {sorted(unknown)}; "
                         f"pick from {list(FIGURES_BY_NAME)}")
    apply_style()
    print(f"target: {WB_LABEL} ({ADC_DIRECTION}) · {len(FEATURES)} features · "
          f"hours {TRAJECTORY_HOURS[0]}-{TRAJECTORY_HOURS[1]} · both bases")
    corr = table(recompute="--recompute" in argv)
    for name in names or list(FIGURES_BY_NAME):
        FIGURES_BY_NAME[name](corr)
    _headline(corr)


if __name__ == "__main__":
    main(sys.argv[1:])
