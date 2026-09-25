"""=======================  THE ONLY FILE YOU NEED TO EDIT  =======================

Set VARS below, then run:   python3 run_all.py

  VARS = "X"    the GBTM is fitted on the MFM cortical gain X alone (Zack's model)
  VARS = "XY"   fitted jointly on X and the corticothalamic gain Y

Everything else -- the classes, the outcome split, the MRI comparison and the figure --
follows from that choice, and every output file is tagged with it, so the two runs can sit
side by side without overwriting each other:

  results/subtype_<VARS>_*.csv          the fitted classes and their tables
  figures/GBTM_MRI_<VARS>.png           the paper figure

A full run is ~25 min for X and ~35 min for XY on a laptop, almost all of it in the 200
random starts and the 100 bootstrap refits. `python3 smoke_test.py` does the same thing
end to end in under a minute on a subsample, and is what to run first.
================================================================================"""

from pathlib import Path

# ---- the choice ------------------------------------------------------------------
VARS = "XY"                  # "X" or "XY"

# ---- how hard to work ------------------------------------------------------------
# K is fixed at 2 deliberately: it is what the published comparison uses, and a larger K
# always has the lower BIC in a mixture whose within-class spread is not exactly Gaussian,
# so BIC on its own is not a reason to prefer one.
N_CLASSES = 2
N_STARTS = 200               # EVERY start is run to convergence -- see lcga.py on why
N_BOOT = 100                 # bootstrap refits behind the +/- 1 SE bands and the stability ARI
SEED = 0

# ---- fixed by the data -----------------------------------------------------------
HOURS = (12, 72)             # hours since ROSC that the trajectories cover
DEGREE = 2                   # quadratic class curves, as in lcmm::hlme
RE_TERMS = 3                 # per-patient random intercept + slope + quadratic
PLOT_VARS = ["X", "Y", "Z"]  # the raw series drawn under the model; Z is never fitted here
VAR_SETS = {"X": ["X"], "XY": ["X", "Y"]}

ROOT = Path(__file__).resolve().parent
DATA, RESULTS, FIGURES = ROOT / "data", ROOT / "results", ROOT / "figures"
HOURLY_CSV = DATA / "hourly_mfm_long.csv"     # 592 patients x hours, MFM X/Y/Z + CPC
MRI_REGIONAL_CSV = DATA / "mri_regional.csv"  # 52 patients x 86 ROIs, ADC metrics
MRI_GLOBAL_CSV = DATA / "mri_global.csv"      # 52 patients, whole-brain ADC
ANNOT = {"lh": DATA / "lh.aparc.annot", "rh": DATA / "rh.aparc.annot"}

# ---- palette ---------------------------------------------------------------------
# Good/poor outcome colours, reused for class identity: the class with the larger share of
# poor outcomes takes the poor colour, so a class line carries the same association as the
# patient lines underneath it.
COLOR_GOOD, COLOR_POOR, COLOR_MID = "#1974CD", "#FE8000", "#8C8C8C"
ALPHA_GOOD, ALPHA_POOR = 0.60, 0.40
Y_LABELS = {"X": "X (cortical E/I)", "Y": "Y (corticothalamic E/I)", "Z": "Z (intrathalamic)"}
X_LABEL = "Time after ROSC (h)"
DPI = 150


def check(tag: str | None = None) -> str:
    """Validate VARS (or an override) and make sure the inputs are all present."""
    tag = (tag or VARS).upper()
    if tag not in VAR_SETS:
        raise SystemExit(f"VARS={tag!r} in config.py is not one of {sorted(VAR_SETS)}")
    missing = [p.name for p in (HOURLY_CSV, MRI_REGIONAL_CSV, MRI_GLOBAL_CSV, *ANNOT.values())
               if not p.exists()]
    if missing:
        raise SystemExit(f"missing input files in {DATA}: {', '.join(missing)}")
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    return tag
