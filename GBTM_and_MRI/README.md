# GBTM trajectory classes and MRI injury

Two latent trajectory classes are fitted to post-cardiac-arrest patients' hourly MFM
(Robinson corticothalamic neural-field) parameters, and the classes are then read against
outcome and against MRI-measured cytotoxic oedema.

The model is a **group-based trajectory model** (Nagin's GBTM / a growth mixture model):
quadratic class curves on time since ROSC, with a per-patient random intercept, slope and
quadratic, fitted by EM. It matches the R `lcmm::hlme` specification
`fixed = mixture = random = ~ NormHour + I(NormHour^2)`, `ng = 2`.

**The classes are fitted on the EEG-derived trajectories alone.** Outcome and MRI are never
inputs, so anything they show across classes is an observed association, not a definition.

---

## Run it

```bash
python3 run_all.py         # ~25 min (X) or ~35 min (XY)
python3 run_all.py X XY    # or just do both, one after the other
```

`config.py` is the only file to edit. The one choice is which trajectories the model is
fitted on:

| `VARS` | fitted on | what it is |
|---|---|---|
| `"X"` | X | the cortical excitation/inhibition gain alone |
| `"XY"` | X and Y jointly | adding the corticothalamic gain |

Z (the intrathalamic gain) is drawn in the figure as a companion series but is never fitted.

Every output is tagged with that choice, so the two runs coexist rather than overwrite each
other. The figure is `figures/GBTM_MRI_X.png` / `figures/GBTM_MRI_XY.png`.

Requirements: `numpy`, `pandas`, `scipy`, `matplotlib`, `nibabel`, `nilearn`. `joblib` is
optional and only speeds the bootstrap up. Nothing is downloaded at run time.

---

## The figure

Two rows, no figure title (the caption supplies it):

* **row 1** — the estimated class trajectory of each fitted variable, with a ± 1 SE band from
  the bootstrap refits, and the two classes' outcome donuts beside them;
* **row 2** — the raw hourly X, Y and Z of every patient in one column on the left, coloured
  by class, and the cortical map of the between-class difference in median ADC on the right.

Class colours are the outcome palette: **orange** is the class with the larger share of poor
outcomes, **blue** the other. Classes are numbered 1..K by decreasing % good outcome, so
class 2 is always the more-poor one and position means the same thing in every figure.

The difference map's scale is clipped at the 85th percentile of |difference|, because the two
classes differ mostly by a global offset and a scale set by the extremes would leave every
region in one flat shade. The caption prints the sign agreement and the number of saturated
regions, so the map is read for *where* the gap is widest rather than for its sign.
Subcortical territory and the medial wall are left grey: they carry no cortical value on this
surface, and painting them would read as a measurement.

---

## Input data (`data/`)

| file | what it is |
|---|---|
| `hourly_mfm_long.csv` | 592 patients × hours 12–72, sensor-level MFM `X`/`Y`/`Z` (χ² < 4) and CPC. One row per patient-hour actually recorded — an unrecorded hour is simply an absent row, never interpolated. |
| `mri_regional.csv` | the 52 patients who also have an MRI × 86 Desikan-Killiany ROIs: mean and median ADC, and the fraction of the region below 450 / 650 ×10⁻⁶ mm²/s. |
| `mri_global.csv` | the same 52 patients' whole-brain ADC summaries. |
| `lh.aparc.annot`, `rh.aparc.annot` | the FreeSurfer Desikan-Killiany parcellation, used to paint ROI values onto the cortical surface. Copied in so nothing has to download a FreeSurfer subject. |

Patient ids are zero-padded to 4 characters everywhere (`"0001"`). CPC 1–2 is a good outcome
and 3–5 a poor one; 11 of the 592 have no CPC and are counted as neither.

**Only 52 of the 592 have an MRI**, so every MRI statement is an n=52 result inside a
592-patient class structure — roughly 20–30 scans per class.

---

## Output files (`results/`)

| file | one row per |
|---|---|
| `subtype_<tag>_assignments.csv` | patient: class, posterior probabilities, outcome, coverage, whole-brain ADC |
| `subtype_<tag>_hourly.csv` | patient-hour: the raw series with the class joined on |
| `subtype_<tag>_curves.csv` | class × variable × hour: the fitted value, its bootstrap SE and 95% band |
| `subtype_<tag>_summary.csv` | class: name, size, outcome mix, median whole-brain ADC |
| `subtype_<tag>_stability.csv` | the run: bootstrap ARI quartiles, how many starts found the optimum, BIC, entropy |
| `mri_roi_subtype_<tag>_means.csv` | ROI × metric × class: ADC, the between-class Mann-Whitney p and its BH-FDR q |
| `mri_macro_subtype_<tag>_means.csv` | macroarea × metric × class: the same over four cortical macroareas |

### Why four macroareas as well as 68 ROIs

68 regional tests on ~25 scans per class survive no multiplicity correction, and the parcels
are neither independent nor individually interesting. `gbtm.MACROAREAS` therefore also pools
both hemispheres into four areas — `frontoparietal` (36 parcels), `temporal` (18),
`occipital` (8) and `sensorimotor` (6: pre-, post- and paracentral) — and runs one test per
area with BH over the four. The cingulate parcels and the insula have no lobe in this scheme
and join `frontoparietal`; every cortical parcel lands in exactly one area, and the code
asserts it.

**This changes the conclusion in neither direction**: on the full data no region and no
macroarea reaches q < 0.05 for either model or either MRI metric. That is the honest
read-out at this sample size, and it is why the figure shows the difference itself rather
than a thresholded map.

---

## Three things in `lcga.py` that are load-bearing

`lcga.py` is the model. It was tuned against planted-class simulations, and three findings in
it are easy to undo by accident:

1. **The likelihood never forms a patient's covariance matrix** (Woodbury identity + the
   matrix determinant lemma on q×q blocks), which is what makes hourly data as cheap as
   binned data. `test_loglik_matches_dense()` checks that identity against
   `scipy.stats.multivariate_normal`.
2. **EM must run to convergence.** Stopping at 2000 iterations cost 600 log-likelihood units
   at K=3, which *inverted* the BIC ordering between K=2 and K=3. The cap is 20 000 with a
   tight tolerance, and `fit` warns if a run ever reaches it.
3. **Do not screen starts.** Ranking starts by their likelihood after a few dozen iterations
   — `lcmm`'s `gridsearch` strategy — is actively misleading on this surface: on planted data
   the best-screened starts all converged to the same *inferior* optimum while the good one
   was found by 2 starts in 40 that screening discarded. Every start runs to convergence,
   which is affordable because the EM is SQUAREM-accelerated (~5× fewer iterations).

`python3 lcga.py` runs both self-checks (~8 min). If anything downstream ever looks wrong,
that is the first thing to run — it is a correctness gate on the model itself, independent of
this dataset.

---

## Reading the result

`subtype_<tag>_stability.csv` is the first thing to look at. `n_replicated` of `n_starts` is
how many random starts reached the reported optimum, and `ari_median` is how reproducible the
partition is under resampling — an ARI near 1 means the split is a property of the data, not
of the starting values. A low value there invalidates everything downstream, however good the
figure looks.

K is fixed at 2 on purpose. BIC falls with every added class in a mixture whose within-class
spread is not exactly Gaussian, so a lower BIC at K=3 is not on its own a reason to prefer
three classes; two also keeps the variable sets readable against each other and matches the
published `ng = 2`.
