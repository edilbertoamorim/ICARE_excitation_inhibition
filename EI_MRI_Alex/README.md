# EI_MRI_Alex — EEG / MFM features against MRI injury

A small, self-contained extract of the **CCMRC** study: do EEG source features — and the
Robinson corticothalamic neural-field (MFM) parameters fitted to them — relate to MRI injury
severity in a 52-patient post-cardiac-arrest cohort (I-CARE)?

Everything runs from this folder with no other part of the thesis codebase present.

```
EI_MRI_Alex/
  common.py                       shared foundation: paths, cohort, atlas→surface, style
  01_mfm_cortex_by_outcome.py     raw X, Y, Z on the cortex, good vs poor outcome
  02_mri_medADC_by_outcome.py     regional median ADC on the cortex, good vs poor outcome
  03_correlations_overtime.py     every feature vs whole-brain median ADC, hour by hour
  04_cluster_correlations.py      X/Y/Z vs regional median ADC, inside each EEG cluster
  data/                           inputs, copied from the CCMRC pipeline (+ derived/)
  figures/                        all PNG output
  README.md
```

Four figure scripts, one per figure family, each runnable on its own. `common.py` is not a
fifth figure script — it is the layer underneath all four, so the Desikan–Killiany → fsaverage
mapping, the cohort definition and the palette exist once instead of four times.

## One MRI target

**Median ADC only.** Regionally that is `Median.ADC` per Desikan–Killiany region; at the
patient level it is `wb_Median_adc`, the whole-brain median. The parent pipeline also carries
an ADC-below-650 fraction and a whole-brain mean; both are deliberately absent here.

> **Higher median ADC = healthier tissue.** ADC falls where cytotoxic oedema develops, so a
> *positive* correlation between a feature and ADC means that feature is high where the tissue
> is **less** injured. This is the opposite of the "higher = worse" reading an injury map
> usually invites, and every caption in the folder says so.

## Running it

```bash
python3 -W ignore 01_mfm_cortex_by_outcome.py        # ~1 min
python3 -W ignore 02_mri_medADC_by_outcome.py        # ~30 s
python3 -W ignore 03_correlations_overtime.py        # ~4 min first run, ~10 s after
python3 -W ignore 04_cluster_correlations.py         # ~6 min first run, ~5 min after
```

Scripts 3 and 4 compute their statistics once and cache them in `data/derived/`; later runs
reuse the table unless given `--recompute`. Everything else is surface rendering, which is
where the time goes — so the slow scripts are narrowable:

```bash
python3 -W ignore 03_correlations_overtime.py rho        # one of the three figures
python3 -W ignore 04_cluster_correlations.py X           # one coordinate
python3 -W ignore 04_cluster_correlations.py subcortical # just the deep-gray panel
python3 -W ignore 01_mfm_cortex_by_outcome.py --mesh=fsaverage   # full resolution, ~10x slower
```

Requires `numpy pandas scipy matplotlib nibabel nilearn mne`. The first run downloads the
fsaverage surfaces (nilearn → `~/nilearn_data`) and the DK `aparc.annot` parcellation
(MNE → `~/mne_data`) and caches them; after that it is offline.

## The four figures

### 1 · `mfm_XYZ_cortex_by_outcome.png`

Cohort-mean X, Y and Z per cortical region, three columns, **two rows: good outcome on top,
poor underneath.** X, Y and Z are the Robinson stability-zone coordinates and are gains of
*different* loops — X intracortical, Y corticothalamic, Z intrathalamic — which is why they
are three maps and not one. Each value is the mean, within an outcome group, of each
patient's median in that region over hours 12–48.

One colour scale per **column**, shared by its two rows, so good and poor are comparable. The
three columns occupy completely different ranges (X ≈ 0.73, Y ≈ 0.05, Z ≈ 0.10), so colours
are comparable **down a column and never across one**. No MRI, and no test, in this figure.

### 2 · `mri_medADC_cortex_by_outcome.png`

The same shape for the MRI: cohort-mean regional median ADC, good row on top of poor row,
four views across. Both rows share one colour scale — the difference between them is the whole
point — clipped to the 2nd–98th percentile, because between-region variation is as large as
the between-group variation and a min-max scale would flatten the group difference. Cortical
means: **good 1051, poor 950** ×10⁻⁶ mm²/s.

### 3 · `patientovertime_medADC_{rho,heatmap,basis}.png`

The patient-level correlation with whole-brain median ADC, recomputed **at every hour** from
12 to 72:

```
x = the patient's ROI-averaged feature value at that hour
y = the patient's whole-brain median ADC        (fixed — one MRI scan per patient)
rho, p = spearmanr(x, y)                        # n = 52, or 27 / 25 by outcome
```

Only *x* moves with time, so these curves are about **when** the EEG carries the signal.
`_rho` gives one panel per feature with the whole cohort and both outcome groups; `_heatmap`
gives the same numbers as feature × hour blocks, blanked where p ≥ 0.05, with a
recording-coverage strip; `_basis` is the sensitivity check described below. The vertical rule
at hour 48 is where the 12–48 h window used by figures 1, 2 and 4 ends.

Result on the whole cohort: **X is the only feature that clears p<0.05 at any hour, and it
does so as one run, hours 38–53** (16 of 61 hours, ρ up to +0.34). Everything else stays
inside the band.

### 4 · `clusters_medADC_{X,Y,Z}_corr.png` + `clusters_medADC_XYZ_subcortical.png`

The per-region correlation of X / Y / Z with **that region's** median ADC, computed separately
inside each EEG cluster. The clusters are an outside input — a UMAP + clustering of 542 ICARE
patients fitted on **EEG alone, never on MRI** — so any MRI ordering across them is an observed
association, not a definition. 45 of our 52 are in it; the other 7 form an explicit
`unassigned` group and are never dropped.

Blocks run left to right by **decreasing % good outcome**: cluster 0 (92 %) → 2 (80 %) →
no cluster (57 %) → 3 (22 %) → 1 (14 %). Each figure is **two rows of the same numbers** —
every region on top, only the p<0.05 regions underneath.

## Data

`data/` is copied verbatim from the CCMRC pipeline's `Data/Data_preprocessed/` and
`atlas_utils/` (~4.9 MB, small enough to track):

| file | contents |
|---|---|
| `patients.csv` | the cohort: IDs, CPC, outcome, whole-brain ADC, QC, interpolation coverage |
| `features_regional.csv` | `(patient, ROI)` × 19 features, each the median over hours 12–48 |
| `features_hourly.csv` | `(patient, hour)` × 19 ROI-averaged features, with an `observed` flag |
| `mri_regional.csv` | `(patient, ROI)` × ADC metrics |
| `mri_global.csv` | `(patient)` whole-brain ADC summaries |
| `cluster_assignments.csv` | the EEG-derived UMAP clustering (outside input) |
| `DesikanKilliany-LabelID-LabelName-1mm.txt` | the 86-row atlas label table |
| `derived/` | the two statistics tables scripts 3 and 4 cache |

**The cohort is the 52 `retained` patients** — those with EEG, an MFM fit, and an MRI that
passes QC: 27 good outcome, 25 poor. Nobody is dropped for thin EEG sampling
(`observed_fraction` records that instead).

**The 19 features** are 16 curated EEG features (amplitude, burst-continuity, absolute and
relative band power, four aperiodic exponents) plus X, Y and Z. All of them — and the ADC —
live on the same **86-region Desikan–Killiany atlas**, which is the structural fact the whole
design rests on: the analysis can stay regional instead of collapsing to one number per
patient.

**Every feature value is a median over hours 12–48 since ROSC.** `Hour` is a clock shared by
all patients (recordings start anywhere from hour 12 to 46), not hours since that patient's
recording began. The MRI is a single scan, so these are associations *across patients*.

## Things that will mislead you if you skip them

**Never read an unmasked ρ map on its own.** Every region gets a colour whether or not there
is evidence for it, so a wall-to-wall coloured brain reads as a wall-to-wall real effect. Both
correlation figures show the same numbers twice, and the gap between the rows is the message:
cluster 0's uniformly blue X map holds **2 significant regions of 68**; cluster 1's uniformly
red one holds **4**.

**At these n, absence of stars is absence of power.** |ρ| must reach **0.27** at n=52, and
**0.53–0.75** at the cluster sizes of 14 down to 7. That is why the cluster figures colour by
ρ rather than by significance: a significance map at n=7 is a map of what the study cannot
see, and would read as "no effect" where the truth is "no power". `MIN_PAIRS` is lowered from
10 to **6** there on purpose, so cluster 3 (n=9) and the unassigned group (n=7) appear at
all — the cost being that both are estimating a rank correlation from 7–9 points.

**Nothing is corrected for multiplicity, and the tests are not independent.** Across 86
regions per cluster-coordinate cell, ~13 of 258 would clear α = 0.05 by chance. Clusters 0, 1
and 2 land at or below that (4, 8, 4); **cluster 3 (19) and the unassigned group (27) land
above it — and they are the two smallest groups**, which is the opposite of what power would
predict. Treat that as an artefact of tiny-n rank correlation plus spatial dependence between
neighbouring regions, not as a finding, unless a permutation test says otherwise. Likewise for
the over-time figures: neighbouring hours are not independent tests — the series are
Kalman-smoothed and the MRI target never moves — so **a run of significant hours is one
finding, not a dozen.** Read runs and patches, never a single hour or a single region.

**The interpolated basis extrapolates at the ends.** The over-time correlation is computed on
Kalman-filled series so that n stays fixed at 52/27/25 and the curve is comparable hour to
hour. At hours 12–20 and 60–72 that rests on as few as 14 genuinely recording patients, and a
random-walk smoother is close to holding the last observed value — so those ends carry
mid-window information wearing an early/late label. `_basis.png` is the check: where the
interpolated and observed-only curves agree, the timing is real.

**Splitting by outcome halves n**, taking the required |ρ| from 0.27 to 0.38 / 0.40. A group
whose dots disappear has less power; it has not been shown to differ. No between-group test is
computed anywhere in this folder.

**The medial wall is left unpainted on purpose**, and the 18 subcortical regions are never
folded into a surface. Neither carries a cortical value, and filling them with the bottom of
the scale reads as "most negative here" while filling them with the zero of a diverging ramp
reads as "no effect here". Deep gray gets its own panel in figure 4 — it matters, because the
deep gray nuclei carry the largest good-versus-poor ADC difference in this cohort.

**Colour is never the only carrier of identity.** Outcome groups carry a dash pattern as well
as a colour everywhere except the over-time curves, where the shape over time is what is being
read and a dash pattern breaks it up. The red/green significance ramp on the over-time heatmap
is the one colour-vision-unsafe pair, kept to match the parent codebase; the signed values are
always in `data/derived/`.

## Provenance

Extracted from the CCMRC study folder of the `Excitation-Inhibition-Modeling` thesis codebase.
Both statistics tables reproduce that pipeline's results **exactly** — max |Δρ| = 0 across all
6942 over-time cells and all 860 shared cluster cells — so figures made here and there are
directly comparable. The plotting toolkit is the same geometry and palette, rewritten as one
self-contained module.

`data/` is copied input, not generated here: regenerating it means re-running steps 01–02 of
the parent pipeline, not anything in this folder.
