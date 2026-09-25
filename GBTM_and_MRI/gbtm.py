"""Stage 1 -- fit the trajectory classes and write every table the figure needs.

What happens here, in order:

  1. read the 592 patients' hourly MFM trajectories (12-72 h since ROSC);
  2. fit a 2-class group-based trajectory model to the chosen variables (`lcga.py`);
  3. number the classes by decreasing % good outcome and name them by curve shape;
  4. bootstrap the fit to get a standard error on each class curve and a stability ARI;
  5. join the 52 patients who also have an MRI and compare regional ADC between the classes.

The classes are fitted on the EEG-derived trajectories ALONE. Outcome and MRI are never
inputs, which is the whole point: anything they show across classes is an observed
association, not a definition.

Outputs, all tagged with the variable set (results/):
  subtype_<tag>_assignments.csv   one row per patient: class, posteriors, outcome, whole-brain ADC
  subtype_<tag>_hourly.csv        the raw series the figure draws, with the class joined on
  subtype_<tag>_curves.csv        (class, variable, hour) fitted value, bootstrap SE and band
  subtype_<tag>_summary.csv       one row per class: name, size, outcome mix, ADC
  subtype_<tag>_stability.csv     bootstrap ARI quartiles and how many starts found the optimum
  mri_roi_subtype_<tag>_means.csv (ROI, metric, class) ADC, the between-class test and its BH q
  mri_macro_subtype_<tag>_means.csv the same over four cortical macroareas
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import mannwhitneyu

import config as cfg
import lcga

H0, H1 = cfg.HOURS
HOUR_GRID = np.arange(H0, H1 + 1, dtype=float)
REGIONAL_TARGETS = {"pctADC650": "Percentage.ADC.650", "medADC": "Median.ADC"}
LABEL_RISE, LABEL_FALL = 0.03, -0.03      # the shape thresholds the class names use

# Four macroareas over the 68 cortical parcels, both hemispheres pooled. 68 regional tests on
# ~25 scans per class survive no multiplicity correction, and the parcels are neither
# independent nor individually of interest. Two assignments are judgement calls, made here so
# they can be changed in one place: `paracentral` joins the sensorimotor strips it continues
# medially, and the cingulate parcels and the insula -- which have no lobe in this scheme --
# join `frontoparietal` as the medial and perisylvian association cortex.
MACROAREAS = {
    "sensorimotor": ("precentral", "postcentral", "paracentral"),
    "temporal": ("bankssts", "entorhinal", "fusiform", "inferiortemporal", "middletemporal",
                 "parahippocampal", "superiortemporal", "temporalpole", "transversetemporal"),
    "occipital": ("cuneus", "lateraloccipital", "lingual", "pericalcarine"),
    "frontoparietal": ("caudalmiddlefrontal", "frontalpole", "lateralorbitofrontal",
                       "medialorbitofrontal", "parsopercularis", "parsorbitalis",
                       "parstriangularis", "rostralmiddlefrontal", "superiorfrontal",
                       "inferiorparietal", "precuneus", "superiorparietal", "supramarginal",
                       "caudalanteriorcingulate", "isthmuscingulate", "posteriorcingulate",
                       "rostralanteriorcingulate", "insula"),
}
MACROAREA_ORDER = ["frontoparietal", "sensorimotor", "temporal", "occipital"]
_MACRO_BY_PARCEL = {p: a for a, ps in MACROAREAS.items() for p in ps}


def out_path(tag: str, name: str):
    return cfg.RESULTS / f"subtype_{tag}_{name}.csv"


def norm_hour(hours) -> np.ndarray:
    """Hours since ROSC rescaled to [0, 1] over the 12-72 h window the model is specified on."""
    return (np.asarray(hours, float) - H0) / (H1 - H0)


def is_cortical(roi: str) -> bool:
    return roi.startswith("ctx_")


def macroarea(roi: str) -> str:
    if not is_cortical(roi):
        return ""
    parcel = roi.split("_", 2)[2]
    assert parcel in _MACRO_BY_PARCEL, f"{roi}: no macroarea for parcel {parcel!r}"
    return _MACRO_BY_PARCEL[parcel]


# ---- data -------------------------------------------------------------------------------
def load_hourly(subsample: int | None = None, seed: int = 0) -> pd.DataFrame:
    """The hourly table, keyed on a zero-padded 4-character patient id.

    CPC 1-2 is a good outcome and 3-5 a poor one; 11 of the 592 have no CPC and keep a null
    outcome, so they are never counted as either.
    """
    h = pd.read_csv(cfg.HOURLY_CSV)
    h["patient_id"] = h.pat_ID.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    h = h[h.hour.between(H0, H1)]
    h["outcome"] = np.where(h.cpc.isin([1.0, 2.0]), "good",
                            np.where(h.cpc.notna(), "poor", None))
    h = h[["patient_id", "hour", "cpc", "outcome"] + cfg.PLOT_VARS]
    if subsample:
        # The smoke test keeps every patient who has an MRI -- they are what the second half
        # of the pipeline runs on -- and fills up to `subsample` with the rest.
        with_mri = set(pd.read_csv(cfg.MRI_GLOBAL_CSV, dtype={"patient_id": str}).patient_id)
        ids = sorted(set(h.patient_id))
        keep = [p for p in ids if p in with_mri]
        rest = [p for p in ids if p not in with_mri]
        keep += list(np.random.default_rng(seed).permutation(rest)[:max(0, subsample - len(keep))])
        h = h[h.patient_id.isin(set(keep))]
    return h.sort_values(["patient_id", "hour"]).reset_index(drop=True)


def to_array(h: pd.DataFrame, patients: list[str], variables: list[str]) -> np.ndarray:
    """Long hourly rows -> (n_patients, n_hours, n_vars), NaN where that hour was not recorded."""
    Y = np.full((len(patients), len(HOUR_GRID), len(variables)), np.nan)
    pi = {p: i for i, p in enumerate(patients)}
    ti = {hh: j for j, hh in enumerate(HOUR_GRID)}
    rows = h.patient_id.map(pi).to_numpy()
    cols = h.hour.astype(float).map(ti).to_numpy()
    for v, var in enumerate(variables):
        Y[rows, cols, v] = h[var].to_numpy()
    return Y


def patient_meta(h: pd.DataFrame) -> pd.DataFrame:
    """One row per patient: outcome, coverage, and whole-brain ADC for the 52 with an MRI."""
    meta = (h.groupby("patient_id")
            .agg(cpc=("cpc", "first"), outcome=("outcome", "first"),
                 first_hour=("hour", "min"), last_hour=("hour", "max"),
                 n_hours=("hour", "size")).reset_index())
    wb = pd.read_csv(cfg.MRI_GLOBAL_CSV, dtype={"patient_id": str})
    wb["pct_adc_650"] = wb.mean_pct_adc_650 * 100.0
    meta = meta.merge(wb[["patient_id", "wb_Median_adc", "wb_Average_adc", "pct_adc_650"]],
                      on="patient_id", how="left")
    meta["has_mri"] = meta.wb_Median_adc.notna()
    return meta


# ---- naming and numbering ----------------------------------------------------------------
def shape_names(curves: np.ndarray, var_idx: int = 0) -> list[str]:
    """Name each class by the shape of its predicted curve (rise 0.03 / fall -0.03)."""
    lo, hi = curves[:, 0, var_idx], curves[:, -1, var_idx]
    delta = hi - lo
    a, b = np.argsort(curves[:, :, var_idx].mean(axis=1))    # lower mean first
    names = [""] * len(curves)
    if delta[a] > LABEL_RISE and delta[b] > LABEL_RISE:
        names[a], names[b] = "Delayed-Rise", "Early-Rise"
    elif delta[a] < LABEL_FALL and delta[b] < LABEL_FALL:
        names[a], names[b] = "Lower-Declining", "Higher-Declining"
    elif delta[b] > LABEL_RISE or delta[a] < LABEL_FALL:
        names[a], names[b] = "Declining", "Persistent-High"
    else:
        names[a], names[b] = "Lower", "Higher"
    return [f"{n} trajectory" for n in names]


def class_order(model: lcga.Fit, meta: pd.DataFrame) -> np.ndarray:
    """Fitted class index -> 1..K by decreasing % good outcome (ties: larger class first).

    So class 2 is always the more-poor class, and position means the same thing in every
    figure. Patients without a CPC are NaN here, not "not good".
    """
    m = model.modal
    good = np.where(meta.outcome.isna(), np.nan, (meta.outcome == "good").astype(float))
    key = [(-np.nanmean(good[m == j]) if np.isfinite(good[m == j]).any() else 1.0,
            -(m == j).sum()) for j in range(model.k)]
    relabel = np.empty(model.k, int)
    for rank, j in enumerate(sorted(range(model.k), key=lambda j: key[j])):
        relabel[j] = rank + 1
    return relabel


def _boot_one(Z, t, k, ref_modal, n_starts, seed):
    """One bootstrap refit: resample patients, refit, then re-assign the ORIGINAL patients.

    Re-assigning the originals is what makes the ARI comparable to the observed classes, and
    what lets the refitted curves be matched to them class by class.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, Z.shape[0], Z.shape[0])
    f = lcga.fit(Z[idx], t, k, degree=cfg.DEGREE, n_starts=max(2, n_starts // 10),
                 seed=seed, re_terms=cfg.RE_TERMS)
    modal = lcga.posterior(f, Z, t).argmax(axis=1)
    curves = np.empty_like(f.curve(t))
    curves[_align(ref_modal, modal, k)] = f.curve(t)
    return _ari(ref_modal, modal), curves


def _align(ref_modal, modal, k) -> np.ndarray:
    """Match a bootstrap refit's arbitrary class indices to the original fit's, by overlap."""
    overlap = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            overlap[i, j] = np.sum((ref_modal == i) & (modal == j))
    ref_idx, new_idx = linear_sum_assignment(-overlap)
    perm = np.empty(k, int)
    perm[new_idx] = ref_idx
    return perm


# ---- MRI --------------------------------------------------------------------------------
def _bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values for ONE family of tests. NaN p-values pass through."""
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if not m:
        return q
    order = np.flatnonzero(ok)[np.argsort(p[ok], kind="stable")]
    ranked = p[order] * m / np.arange(1, m + 1)
    q[order] = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    return q


def _mri_long(a: pd.DataFrame) -> pd.DataFrame:
    d = pd.read_csv(cfg.MRI_REGIONAL_CSV, dtype={"patient_id": str}).merge(
        a.loc[a.has_mri, ["patient_id", "subtype"]], on="patient_id")
    d["Percentage.ADC.650"] = d["Percentage.ADC.650"] * 100.0
    return d


def roi_means(a: pd.DataFrame) -> pd.DataFrame:
    """Per-ROI injury by class, plus the two-sided Mann-Whitney between the two classes.

    A rank test rather than a t-test because each class holds only ~20-30 scans and regional
    ADC is skewed. BH-FDR runs within one metric and one compartment (the 68 cortical ROIs,
    the 18 subcortical ones, separately) -- that is the set one figure shows.
    """
    d = _mri_long(a)
    classes = sorted(d.subtype.unique())
    rows = []
    for key, col in REGIONAL_TARGETS.items():
        g = d.groupby(["roi", "roi_adc", "subtype"])[col]
        rows.append(pd.DataFrame({"mean": g.mean(), "median": g.median(), "sd": g.std(),
                                  "n_patients": g.size()}).reset_index().assign(metric=key))
    out = pd.concat(rows, ignore_index=True)
    out["subcortical"] = ~out.roi.map(is_cortical)

    tests = []
    for key, col in REGIONAL_TARGETS.items():
        for roi, g in d.groupby("roi"):
            x0 = g.loc[g.subtype == classes[0], col].dropna()
            x1 = g.loc[g.subtype == classes[-1], col].dropna()
            p = (float(mannwhitneyu(x1, x0, alternative="two-sided").pvalue)
                 if len(classes) == 2 and len(x0) > 1 and len(x1) > 1 else np.nan)
            tests.append({"roi": roi, "metric": key, "p_diff": p})
    t = pd.DataFrame(tests)
    t["q_diff"] = np.nan
    for _, idx in t.groupby(["metric", ~t.roi.map(is_cortical)]).groups.items():
        t.loc[idx, "q_diff"] = _bh_fdr(t.loc[idx, "p_diff"].to_numpy())
    return out.merge(t, on=["roi", "metric"], how="left").sort_values(
        ["metric", "subtype", "roi"])


def macro_means(a: pd.DataFrame) -> pd.DataFrame:
    """The same comparison over four cortical macroareas -- four tests instead of 68.

    Each patient contributes ONE value per area: the unweighted mean over that area's
    parcels, both hemispheres pooled (unweighted because the ROI table carries no parcel
    volumes). BH then runs over the four areas within a metric.
    """
    d = _mri_long(a)
    d["macroarea"] = d.roi.map(macroarea)
    d = d[d.macroarea != ""]
    classes = sorted(d.subtype.unique())
    n_rois = d.groupby("macroarea").roi.nunique()
    rows = []
    for key, col in REGIONAL_TARGETS.items():
        per = d.groupby(["patient_id", "subtype", "macroarea"])[col].mean().reset_index()
        for area in MACROAREA_ORDER:
            g = per[per.macroarea == area]
            x0 = g.loc[g.subtype == classes[0], col].dropna()
            x1 = g.loc[g.subtype == classes[-1], col].dropna()
            p = (float(mannwhitneyu(x1, x0, alternative="two-sided").pvalue)
                 if len(classes) == 2 and len(x0) > 1 and len(x1) > 1 else np.nan)
            for c in classes:
                x = g.loc[g.subtype == c, col].dropna()
                rows.append({"macroarea": area, "n_rois": int(n_rois[area]), "metric": key,
                             "subtype": c, "n_patients": len(x), "mean": x.mean(),
                             "median": x.median(), "sd": x.std(), "p_diff": p})
    out = pd.DataFrame(rows)
    out["q_diff"] = np.nan
    for _, idx in out.groupby("metric").groups.items():
        # One p per AREA, repeated on both class rows -- BH over the four areas, not the
        # eight rows, or the correction is applied twice over.
        u = out.loc[idx].drop_duplicates("macroarea")
        q = dict(zip(u.macroarea, _bh_fdr(u.p_diff.to_numpy())))
        out.loc[idx, "q_diff"] = out.loc[idx, "macroarea"].map(q)
    return out


# ---- the run ----------------------------------------------------------------------------
def summary(a: pd.DataFrame, labels: dict) -> pd.DataFrame:
    rows = []
    for c in sorted(labels):
        s = a[a.subtype == c]
        r = {"subtype": c, "label": labels[c], "n_patients": len(s),
             "n_good": int((s.outcome == "good").sum()),
             "n_poor": int((s.outcome == "poor").sum()),
             "n_no_cpc": int(s.outcome.isna().sum()),
             "pct_good": 100.0 * (s.outcome == "good").mean(),
             "mean_post": s.post_max.mean(), "n_with_mri": int(s.has_mri.sum())}
        for key, col in {"wbMedADC": "wb_Median_adc", "wbPctADC650": "pct_adc_650"}.items():
            x = s[col].dropna()
            r[f"{key}_n"], r[f"{key}_median"] = len(x), x.median()
        rows.append(r)
    return pd.DataFrame(rows)


def run(tag: str | None = None, n_starts: int | None = None, n_boot: int | None = None,
        subsample: int | None = None, quiet: bool = False) -> dict:
    """Fit, characterise and write every table. Returns the paths it wrote."""
    tag = cfg.check(tag)
    fit_vars = cfg.VAR_SETS[tag]
    n_starts = cfg.N_STARTS if n_starts is None else n_starts
    n_boot = cfg.N_BOOT if n_boot is None else n_boot
    say = (lambda *a_: None) if quiet else print

    h = load_hourly(subsample)
    meta = patient_meta(h)
    patients = sorted(meta.patient_id)
    meta = meta.set_index("patient_id").loc[patients].reset_index()
    t = norm_hour(HOUR_GRID)

    Yall = to_array(h, patients, cfg.PLOT_VARS)
    Y = Yall[:, :, [cfg.PLOT_VARS.index(v) for v in fit_vars]]
    # Standardised per variable so a joint X+Y fit is not dominated by whichever has the
    # larger raw spread; the curves are put back on the original scale before anything is
    # written, so every number in the outputs is in MFM units.
    mu, sd = np.nanmean(Y, axis=(0, 1)), np.nanstd(Y, axis=(0, 1))
    Z = (Y - mu) / sd
    say(f"  vars={tag} ({'+'.join(fit_vars)}): {len(patients)} patients, "
        f"{int(np.isfinite(Y).sum())} patient-hours; {int(meta.has_mri.sum())} have MRI")

    model = lcga.fit(Z, t, cfg.N_CLASSES, degree=cfg.DEGREE, n_starts=n_starts,
                     seed=cfg.SEED, re_terms=cfg.RE_TERMS)
    say(f"  K={model.k}: loglik {model.loglik:.1f}, BIC {model.bic:.1f}, "
        f"entropy {model.entropy:.2f}, {model.n_replicated}/{model.n_starts} starts found it")

    curves_raw = model.curve(t) * sd + mu
    relabel = class_order(model, meta)
    labels = {int(relabel[j]): n for j, n in enumerate(shape_names(curves_raw))}

    a = meta.copy()
    a["subtype"] = relabel[model.modal]
    post_sorted = np.empty_like(model.post)
    post_sorted[:, relabel - 1] = model.post
    for c in range(1, model.k + 1):
        a[f"post_{c}"] = post_sorted[:, c - 1]
    a["post_max"] = post_sorted.max(axis=1)
    a["label"] = a.subtype.map(labels)
    a.to_csv(out_path(tag, "assignments"), index=False)

    hourly = h.merge(a[["patient_id", "subtype", "label"]], on="patient_id", how="left")
    hourly.to_csv(out_path(tag, "hourly"), index=False)

    # ---- bootstrap: the SE band on each class curve, and how reproducible the split is.
    # Each refit is independent, so they run in parallel where joblib is available -- 100
    # refits at a tenth of the main fit's starts is otherwise the longest part of the run.
    say(f"  bootstrap: {n_boot} refits at {max(2, n_starts // 10)} starts each")
    jobs = [(Z, t, model.k, model.modal, n_starts, 1000 + b) for b in range(n_boot)]
    boots = None
    if n_boot >= 8:                    # below that, starting the workers costs more than it saves
        try:
            from joblib import Parallel, delayed

            boots = Parallel(n_jobs=-1)(delayed(_boot_one)(*j) for j in jobs)
        except ImportError:
            pass
    if boots is None:
        boots = [_boot_one(*j) for j in jobs]
    aris = [b[0] for b in boots]
    bc = [b[1] for b in boots]
    rows = []
    bc = np.stack(bc) * sd + mu if bc else None
    for j in range(model.k):
        for v, var in enumerate(fit_vars):
            for hh, hour in enumerate(HOUR_GRID):
                at = bc[:, j, hh, v] if bc is not None else np.array([np.nan])
                rows.append({"subtype": int(relabel[j]), "variable": var, "hour": hour,
                             "fit": curves_raw[j, hh, v],
                             "se": float(np.std(at, ddof=1)) if len(at) > 1 else np.nan,
                             "lo": np.percentile(at, 2.5) if len(at) > 1 else np.nan,
                             "hi": np.percentile(at, 97.5) if len(at) > 1 else np.nan})
    pd.DataFrame(rows).to_csv(out_path(tag, "curves"), index=False)
    aris = np.array(aris) if aris else np.array([np.nan])
    pd.DataFrame([{"n_boot": n_boot, "ari_median": np.median(aris),
                   "ari_q1": np.percentile(aris, 25) if len(aris) > 1 else np.nan,
                   "ari_q3": np.percentile(aris, 75) if len(aris) > 1 else np.nan,
                   "n_starts": model.n_starts, "n_replicated": model.n_replicated,
                   "loglik": model.loglik, "bic": model.bic, "entropy": model.entropy}]
                 ).to_csv(out_path(tag, "stability"), index=False)

    s = summary(a, labels)
    s.to_csv(out_path(tag, "summary"), index=False)
    roi_means(a).to_csv(cfg.RESULTS / f"mri_roi_subtype_{tag}_means.csv", index=False)
    m = macro_means(a)
    m.to_csv(cfg.RESULTS / f"mri_macro_subtype_{tag}_means.csv", index=False)

    say("\n  classes (numbered by decreasing % good outcome):")
    say(s[["subtype", "label", "n_patients", "n_good", "n_poor", "pct_good", "n_with_mri",
           "wbMedADC_median"]].to_string(index=False, float_format="%.1f"))
    say(f"\n  bootstrap stability ARI {np.median(aris):.2f}")
    say("\n  cortical macroareas, median ADC (class 2 - class 1):")
    w = (m[m.metric == "medADC"]
         .pivot_table(index=["macroarea", "n_rois", "p_diff", "q_diff"], columns="subtype",
                      values="mean").reset_index())
    w["diff"] = w[w.columns[-1]] - w[w.columns[-2]]
    say(w.to_string(index=False, float_format="%.2f"))
    return {"summary": s, "macro": m, "ari": float(np.median(aris))}


def _ari(a, b) -> float:
    """Adjusted Rand index between two labellings (no sklearn dependency)."""
    a, b = np.asarray(a), np.asarray(b)
    tab = pd.crosstab(a, b).to_numpy()
    n = tab.sum()
    comb = lambda x: x * (x - 1) / 2.0                                   # noqa: E731
    sij, sa, sb = comb(tab).sum(), comb(tab.sum(1)).sum(), comb(tab.sum(0)).sum()
    exp = sa * sb / comb(n)
    return float((sij - exp) / (0.5 * (sa + sb) - exp)) if (0.5 * (sa + sb) - exp) else 1.0


if __name__ == "__main__":
    run()
