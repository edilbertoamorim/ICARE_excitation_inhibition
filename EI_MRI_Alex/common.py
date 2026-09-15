"""Shared foundation for the four figure scripts: paths, data loaders, surfaces, style.

This is not one of the four figure scripts -- it is the layer under all of them, so that the
Desikan-Killiany -> fsaverage mapping, the cohort definition and the palette exist exactly
once instead of four times. Every script starts with `from common import ...` and nothing
here draws a figure of its own.

The one MRI target
------------------
`Median.ADC` per region, and `wb_Median_adc` whole-brain, are the only MRI numbers used
anywhere in this folder. The parent pipeline also carries an ADC-below-650 fraction and a
whole-brain mean; both are deliberately absent here. **Higher median ADC = healthier tissue**,
so a positive correlation with a feature means that feature is high where the tissue is
LESS injured. Every caption states that direction, because it is the opposite of what
"higher = worse" intuition expects from an injury map.

How a per-region number becomes a picture
-----------------------------------------
EEG features, MFM parameters and ADC all live on the same 86-region Desikan-Killiany atlas,
so every quantity here is a `{roi: value}` dict:

    values = {"ctx_lh_precentral": 0.74, ...}
    arrays = roi_values_to_vertices(values, mesh="fsaverage5")   # -> {"lh": (N,), "rh": (N,)}

`roi_values_to_vertices` uses the FreeSurfer `aparc.annot` parcellation to find which mesh
vertices belong to each cortical region and writes the region's value into all of them.
Vertices with no value -- the medial wall, and every subcortical region, which has no cortical
surface at all -- stay NaN, and nilearn renders those as plain sulcal background. That is
deliberate: filling them with `vmin` reads as "most negative here" and filling them with the
zero of a diverging ramp reads as "no effect here", when the truth is "not measured". The 18
subcortical regions are therefore always shown separately, never folded into a surface.

fsaverage5's 10242 vertices per hemisphere are the first 10242 of fsaverage's 163842
(icosahedral nesting), so the full-resolution annot is simply truncated to address the coarse
mesh. Full resolution is worth it for a headline anatomical figure and not for a thumbnail.

Naming: the ADC files and the atlas label table say `Left-Thalamus-Proper*` while the EEG
source features say `Left_Thalamus`. Everything here keys on the EEG form, via
`normalize_roi()`. Getting this wrong silently zeroes both thalami.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt                          # noqa: E402
import numpy as np                                       # noqa: E402
import pandas as pd                                      # noqa: E402
from matplotlib.colors import LinearSegmentedColormap    # noqa: E402

# ----------------------------------------------------------------------------------
# paths -- everything resolves from this file, so the folder can be moved or renamed
# ----------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DERIVED = DATA / "derived"          # tables the scripts compute and cache
FIGURES = ROOT / "figures"

PATIENTS = DATA / "patients.csv"                      # cohort, outcome, whole-brain ADC, QC
FEATURES_REGIONAL = DATA / "features_regional.csv"    # patient x ROI x 19 features
FEATURES_HOURLY = DATA / "features_hourly.csv"        # patient x hour x feature, ROI-averaged
MRI_REGIONAL = DATA / "mri_regional.csv"              # patient x ROI x ADC
MRI_GLOBAL = DATA / "mri_global.csv"                  # patient x whole-brain ADC
CLUSTERS = DATA / "cluster_assignments.csv"           # EEG-derived UMAP clustering
ATLAS_LABELS = DATA / "DesikanKilliany-LabelID-LabelName-1mm.txt"

for _d in (DERIVED, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

MESH_FAST = "fsaverage5"
MESH_FULL = "fsaverage"
FIG_DPI = 200
ALPHA = 0.05

# ----------------------------------------------------------------------------------
# the MRI target -- the only one in this folder
# ----------------------------------------------------------------------------------
ADC_REGIONAL = "Median.ADC"        # per region, in mri_regional.csv
ADC_GLOBAL = "wb_Median_adc"       # whole brain, in mri_global.csv
ADC_LABEL = "median ADC"
ADC_UNITS = "×10⁻⁶ mm²/s"
ADC_DIRECTION = "higher = healthier tissue"

# ----------------------------------------------------------------------------------
# feature schema
# ----------------------------------------------------------------------------------
# The three MFM stability-zone coordinates. In the Robinson corticothalamic neural field they
# are gains of DIFFERENT loops -- X intracortical, Y corticothalamic, Z intrathalamic -- which
# is why they are worth mapping separately rather than as one "MFM" quantity.
MFM_PARAMS = ["X", "Y", "Z"]
MFM_FOCUS = ["X", "Y"]             # the pair the cluster figures stratify

# The 16 curated EEG features that go with them; 19 in total. Families drive the row grouping
# in the over-time figures.
FEATURE_FAMILIES = {
    "amplitude_burden": {"title": "Amplitude and burst-continuity",
                         "features": ["Amp_Mean", "BCI"]},
    "power_absolute": {"title": "Absolute band power",
                       "features": [f"PowerAbs_{b}" for b in
                                    ["Delta", "Theta", "Alpha", "Beta", "Gamma"]]},
    "power_relative": {"title": "Relative band power",
                       "features": [f"PowerRel_{b}" for b in
                                    ["Delta", "Theta", "Alpha", "Beta", "Gamma"]]},
    "aperiodic_exponent": {"title": "Aperiodic exponent (FOOOF)",
                           "features": ["ap_exp_1to20_fixed", "ap_exp_1to40_fixed",
                                        "ap_exp_20to40_fixed", "ap_exp_1to30_knee"]},
    "mfm_stability": {"title": "MFM stability-zone coordinates", "features": MFM_PARAMS},
}
FEATURES = [f for spec in FEATURE_FAMILIES.values() for f in spec["features"]]

# Units as the source tables record them; printed in raw-value block titles, because a
# magnitude map is unreadable without one (`X = 0.75` is a dimensionless loop gain).
UNITS = {"Amp_Mean": "µV", "BCI": "a.u.",
         **{f"PowerAbs_{b}": "dB" for b in ["Delta", "Theta", "Alpha", "Beta", "Gamma"]},
         **{f"PowerRel_{b}": "%" for b in ["Delta", "Theta", "Alpha", "Beta", "Gamma"]},
         **{f: "a.u." for f in ["ap_exp_1to20_fixed", "ap_exp_1to40_fixed",
                                "ap_exp_20to40_fixed", "ap_exp_1to30_knee"] + MFM_PARAMS}}

# Every value in features_regional.csv is the median of that patient's Kalman-interpolated
# trajectory over these hours-since-ROSC. `Hour` is a clock shared by all patients -- they
# start recording anywhere from hour 12 to 46 -- not hours since that patient's recording
# began. TRAJECTORY_HOURS is wider on purpose: it is the full stretch any patient records, so
# the over-time figures can show both sides of the window the static values stop at.
FEATURE_WINDOW = (12, 48)
FEATURE_WINDOW_LABEL = f"median {FEATURE_WINDOW[0]}–{FEATURE_WINDOW[1]} h"
TRAJECTORY_HOURS = (12, 72)
X_LABEL = "hour since ROSC"

# ----------------------------------------------------------------------------------
# palette and style
# ----------------------------------------------------------------------------------
# Pastel throughout; light blue = good outcome, orange = poor. Identity is never carried by
# colour alone -- each outcome group also gets a dash pattern, and the pastel fills are paired
# with a deeper "ink" colour that carries lines and text, because the pastels themselves sit
# below 3:1 contrast against a light surface.
OUTCOME_COLORS = {"good": "#74b2ea", "poor": "#f59f63"}
OUTCOME_INK = {"good": "#3d8fd6", "poor": "#e07c33"}
OUTCOME_ORDER = ["good", "poor"]
OUTCOME_DASH = {"good": (None, None), "poor": (4, 2)}
OUTCOME_LABEL = {"good": "good outcome", "poor": "poor outcome"}

SEQUENTIAL_BLUE = ["#eaf3fe", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
                   "#5598e7"]
DIVERGING_MID = "#ffffff"

# Signed-significance ramp: red = negative, white = not significant, green = positive, with
# intensity encoding -log10(p). Red/green is the one pair colour-vision deficiency cannot
# separate; it is kept to match the parent codebase's atlases, and the signed values are
# always written to CSV in data/derived/ alongside.
PASTEL_NEG = ["#c0392b", "#e05c4f", "#f08a80", "#f8bdb6", DIVERGING_MID]
PASTEL_POS = [DIVERGING_MID, "#b9e0c2", "#8ecf9e", "#5cb974", "#2e9e4f"]

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e3e2de"
SURFACE = "#fcfcfb"

# Sequential = one hue light->dark (magnitude). Diverging = two hues + a neutral midpoint
# (polarity), so it is only ever used where the values genuinely straddle zero.
CMAP_SEQ = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
CMAP_DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#0d366b", "#256abf", "#9ec5f4", DIVERGING_MID, "#f0a5a5", "#d03b3b",
                     "#7a1a1a"])
CMAP_SIG = LinearSegmentedColormap.from_list("neg_white_pos", PASTEL_NEG + PASTEL_POS[1:])

CORR_VMAX = 3.0        # -log10(p) = 3 -> p = 0.001, the significance colour ceiling
RHO_VMAX = 0.6         # the colour ceiling for a raw-rho map, not a significance one


def apply_style() -> None:
    """Recessive axes/grid, text in the ink tokens, thin marks. Applied to every figure."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlesize": 9, "axes.titleweight": "medium", "axes.titlecolor": TEXT_PRIMARY,
        "axes.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.frameon": False, "legend.fontsize": 8, "font.size": 8,
        "lines.linewidth": 2.0,
    })


def done(path: Path) -> None:
    print(f"  wrote {path.relative_to(ROOT)}")


# ----------------------------------------------------------------------------------
# ROI naming
# ----------------------------------------------------------------------------------
def normalize_roi(name: str) -> str:
    """ADC / atlas ROI naming -> the EEG naming everything downstream keys on."""
    return name.replace("-", "_").replace("*", "").replace("Thalamus_Proper", "Thalamus")


def is_subcortical(roi: str) -> bool:
    """True for the 18 subcortical / cerebellar ROIs -- the ones with no `ctx_` prefix."""
    return not roi.startswith("ctx_")


def pretty_roi(roi: str) -> str:
    """`ctx_lh_superiorfrontal` -> `LH superiorfrontal`; `Left_Thalamus` -> `L Thalamus`."""
    if roi.startswith("ctx_"):
        _, hemi, name = roi.split("_", 2)
        return f"{hemi.upper()} {name}"
    return roi.replace("Left_", "L ").replace("Right_", "R ").replace("_", " ")


# ----------------------------------------------------------------------------------
# data loading -- cached, because four scripts want the same joins
# ----------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _cohort() -> pd.DataFrame:
    """The retained cohort only: needs EEG + an MFM fit + an MRI that passes QC.

    Nobody is dropped for thin EEG sampling -- `observed_fraction` records that instead --
    so this flag is the single cohort definition for the whole folder.
    """
    p = pd.read_csv(PATIENTS, dtype={"icare_id": str, "site_id": str})
    p = p[p.retained].rename(columns={"icare_id": "patient_id"})
    assert not p.empty, f"no retained patients in {PATIENTS.name}"
    return p.reset_index(drop=True)


def cohort() -> pd.DataFrame:
    return _cohort().copy()


def outcomes() -> pd.DataFrame:
    """patient_id -> outcome, keyed the way the feature and MRI tables are keyed."""
    return _cohort()[["patient_id", "outcome"]].copy()


def outcome_counts() -> dict[str, int]:
    c = _cohort().outcome.value_counts().to_dict()
    return {g: int(c[g]) for g in OUTCOME_ORDER}


@lru_cache(maxsize=1)
def _features_regional() -> pd.DataFrame:
    d = pd.read_csv(FEATURES_REGIONAL, dtype={"patient_id": str})
    missing = set(FEATURES) - set(d.columns)
    assert not missing, (f"{sorted(missing)} not in {FEATURES_REGIONAL.name} -- the schema "
                         "in FEATURE_FAMILIES disagrees with the data")
    return d[d.patient_id.isin(set(_cohort().patient_id))].reset_index(drop=True)


def features_regional() -> pd.DataFrame:
    """(patient, ROI) x 19 features, each the median over FEATURE_WINDOW."""
    return _features_regional().copy()


@lru_cache(maxsize=1)
def _mri_regional() -> pd.DataFrame:
    d = pd.read_csv(MRI_REGIONAL, dtype={"patient_id": str})
    return d[d.patient_id.isin(set(_cohort().patient_id))].reset_index(drop=True)


def mri_regional() -> pd.DataFrame:
    """(patient, ROI) median ADC for the retained cohort."""
    return _mri_regional()[["patient_id", "roi", ADC_REGIONAL]].copy()


def mri_global() -> pd.DataFrame:
    """(patient) whole-brain median ADC for the retained cohort."""
    d = pd.read_csv(MRI_GLOBAL, dtype={"patient_id": str})
    d = d[d.patient_id.isin(set(_cohort().patient_id))]
    return d[["patient_id", ADC_GLOBAL]].reset_index(drop=True)


def regional_with_adc() -> pd.DataFrame:
    """(patient, ROI): the 19 features + that region's median ADC + the patient's outcome."""
    d = (_features_regional()
         .merge(mri_regional(), on=["patient_id", "roi"])
         .merge(outcomes(), on="patient_id"))
    assert not d.empty, "the feature / ADC join produced no rows -- check the ROI naming"
    return d.reset_index(drop=True)


@lru_cache(maxsize=1)
def _clusters() -> pd.DataFrame:
    """The EEG-derived cluster assignment, joined to the retained cohort.

    An OUTSIDE input: a UMAP + clustering of 542 ICARE patients fitted on EEG alone, never on
    MRI, so any MRI ordering across the clusters is an observed association and not a
    definition. Its `patient_id` is the icare_id with leading zeros stripped ("0354" -> "354"),
    so it is re-padded before joining.

    45 of the 52 retained patients are in it; the other 7 become an explicit `unassigned`
    group and are never dropped. They are not a random remainder -- they are the most injured
    group in the cohort -- so silently excluding them would bias every cross-cluster
    comparison.
    """
    c = pd.read_csv(CLUSTERS, dtype={"patient_id": str})
    c["patient_id"] = c.patient_id.str.zfill(4)
    ids = _cohort()[["patient_id", "outcome", "cpc"]]
    d = ids.merge(c[["patient_id", "cluster", "outcome", "cpc"]], on="patient_id",
                  how="left", suffixes=("", "_clu"))
    # The key is only trustworthy if the labels it carries agree with ours. They do on every
    # matched row today; assert so that a future re-export which shifts the ID convention is
    # caught here rather than silently producing a plausible-looking regrouping of the cohort.
    m = d[d.cluster.notna()]
    assert (m.outcome == m.outcome_clu).all() and (m.cpc == m.cpc_clu).all(), (
        f"{CLUSTERS.name} disagrees with {PATIENTS.name} on outcome/CPC for a matched "
        "patient -- the patient_id convention has probably changed")
    d = d.drop(columns=["outcome_clu", "cpc_clu"])
    # A string label keeps the 7 unassigned patients in the same column as the four clusters,
    # instead of forcing a sentinel integer that a later mean would happily average over.
    d["cluster"] = d.cluster.map(lambda c: "unassigned" if pd.isna(c) else str(int(c)))
    return d.reset_index(drop=True)


def cluster_table() -> pd.DataFrame:
    """patient_id -> cluster ('0'..'3' or 'unassigned') + outcome, for the retained cohort."""
    return _clusters().copy()


# ----------------------------------------------------------------------------------
# atlas -> surface
# ----------------------------------------------------------------------------------
HEMIS = ("lh", "rh")


@lru_cache(maxsize=3)
def fsaverage(mesh: str = MESH_FAST):
    from nilearn import datasets

    return datasets.fetch_surf_fsaverage(mesh=mesh)


@lru_cache(maxsize=3)
def n_vertices(mesh: str = MESH_FAST) -> int:
    from nilearn.surface import load_surf_mesh

    return int(load_surf_mesh(fsaverage(mesh)["pial_left"]).coordinates.shape[0])


@lru_cache(maxsize=1)
def _annot_names() -> dict[str, np.ndarray]:
    """Per-vertex ROI name from the full-resolution fsaverage DK parcellation.

    MNE downloads this once into ~/mne_data and caches it; the first run on a new machine
    fetches it, every run after that is offline.
    """
    import mne
    import nibabel as nib

    label_dir = (Path(mne.datasets.fetch_fsaverage(verbose="ERROR")).parent
                 / "fsaverage" / "label")
    out = {}
    for hemi in HEMIS:
        labels, _, names = nib.freesurfer.read_annot(str(label_dir / f"{hemi}.aparc.annot"))
        names = [n.decode() if isinstance(n, bytes) else n for n in names]
        out[hemi] = np.array(
            [f"ctx_{hemi}_{names[i]}" if 0 <= i < len(names) else "" for i in labels])
    return out


@lru_cache(maxsize=3)
def annot_roi_index(mesh: str = MESH_FAST) -> dict[str, dict[str, np.ndarray]]:
    """{hemi: {roi: vertex indices}} for `mesh`, computed once and reused.

    One sort at first call replaces a full 163k-element string comparison per ROI per map;
    painting 19 features x 2 hemispheres would otherwise be ~2600 such scans.
    """
    nv = n_vertices(mesh)
    idx = {}
    for hemi, names in _annot_names().items():
        truncated = names[:nv]       # fsaverage5 vertices are the first 10242 of fsaverage
        order = np.argsort(truncated, kind="stable")
        srt = truncated[order]
        uniq, starts = np.unique(srt, return_index=True)
        bounds = list(starts) + [len(srt)]
        idx[hemi] = {name: order[bounds[i]:bounds[i + 1]]
                     for i, name in enumerate(uniq) if name}
    return idx


def roi_values_to_vertices(values: dict[str, float], mesh: str = MESH_FAST,
                           fill: float = np.nan) -> dict[str, np.ndarray]:
    """{roi: value} -> per-hemisphere vertex arrays. Cortical ROIs only; the rest stay `fill`."""
    nv = n_vertices(mesh)
    index = annot_roi_index(mesh)
    out = {}
    for hemi in HEMIS:
        arr = np.full(nv, fill, dtype=float)
        for roi, v in values.items():
            ids = index[hemi].get(roi)
            if ids is not None and v is not None and np.isfinite(v):
                arr[ids] = v
        out[hemi] = arr
    return out


def draw_view(ax, hemi: str, view: str, arrays: dict[str, np.ndarray], cmap, vmin: float,
              vmax: float, mesh: str = MESH_FAST, fill: float = np.nan,
              zoom: float = 1.5) -> None:
    """One brain view into an existing 3d axes -- the single rendering primitive here.

    A NaN `fill` is passed straight through to nilearn, which leaves those vertices as the
    plain sulcal background instead of painting them; see the module docstring. A finite
    `fill` still means what it says, for callers that want one.
    """
    from nilearn import plotting as niplot

    fs = fsaverage(mesh)
    data = arrays["lh" if hemi == "left" else "rh"]
    if np.isfinite(fill):
        data = np.nan_to_num(data, nan=fill)
    niplot.plot_surf(fs[f"pial_{hemi}"], data, hemi=hemi, view=view,
                     bg_map=fs[f"sulc_{hemi}"], bg_on_data=True, cmap=cmap, vmin=vmin,
                     vmax=vmax, colorbar=False, axes=ax, alpha=1.0)
    ax.set_box_aspect(None, zoom=zoom)


# ----------------------------------------------------------------------------------
# block grid -- a grid of "blocks", each block a 2x2 of brain views with its own title
# ----------------------------------------------------------------------------------
# The header and footer are reserved in INCHES, not in figure fractions. That is the only way
# the spacing survives a change in row count: with fractions, a one-row grid collides with the
# suptitle and a three-row grid drops titles onto the previous row's colourbar.
VIEWS_2x2 = [("left", "lateral"), ("right", "lateral"),
             ("left", "medial"), ("right", "medial")]


@dataclass(frozen=True)
class BlockGrid:
    """Figure-fraction geometry for a grid of brain blocks."""

    fig: plt.Figure
    ncol: int
    nrow: int
    left: float
    right: float
    top: float
    bottom: float
    height_in: float

    @property
    def bw(self) -> float:
        return (self.right - self.left) / self.ncol

    @property
    def bh(self) -> float:
        return (self.top - self.bottom) / self.nrow

    def origin(self, i: int) -> tuple[float, float]:
        """Bottom-left corner of block i, filling left-to-right then top-to-bottom."""
        r, c = divmod(i, self.ncol)
        return self.left + c * self.bw, self.top - (r + 1) * self.bh


def make_block_grid(n_blocks: int, ncol_max: int = 3, block_w: float = 4.3,
                    block_h: float = 3.15, header_in: float = 0.78,
                    footer_in: float = 0.62) -> BlockGrid:
    """Size a figure for `n_blocks` brain blocks, reserving header/footer in inches."""
    ncol = min(ncol_max, n_blocks)
    nrow = int(np.ceil(n_blocks / ncol))
    fig_h = block_h * nrow + header_in + footer_in
    fig = plt.figure(figsize=(block_w * ncol, fig_h))
    return BlockGrid(fig=fig, ncol=ncol, nrow=nrow, left=0.012, right=0.988,
                     top=1 - header_in / fig_h, bottom=footer_in / fig_h, height_in=fig_h)


def draw_brain_block(grid: BlockGrid, i: int, values: dict[str, float], cmap, vmin: float,
                     vmax: float, mesh: str = MESH_FAST, fill: float = np.nan,
                     zoom: float = 1.5) -> None:
    """Render one 2x2 (LH/RH x lateral/medial) block of cortical surfaces at slot i."""
    arrays = roi_values_to_vertices(values, mesh=mesh, fill=fill)
    x0, y0 = grid.origin(i)
    panel_w, panel_h = grid.bw * 0.48, grid.bh * 0.40
    for j, (hemi, view) in enumerate(VIEWS_2x2):
        r, c = divmod(j, 2)
        ax = grid.fig.add_axes([x0 + grid.bw * 0.02 + c * panel_w,
                                y0 + grid.bh * 0.50 - r * panel_h, panel_w, panel_h],
                               projection="3d")
        draw_view(ax, hemi, view, arrays, cmap, vmin, vmax, mesh=mesh, fill=fill, zoom=zoom)


def block_title(grid: BlockGrid, i: int, title: str, subtitle: str | None = None,
                subtitle_color: str = TEXT_SECONDARY) -> None:
    """Title above block i. `subtitle_color` lets a paired grid carry group identity in the
    subtitle ink, so two blocks of the same quantity are told apart at a glance."""
    x0, y0 = grid.origin(i)
    grid.fig.text(x0 + grid.bw / 2, y0 + grid.bh * 0.95, title, ha="center", va="center",
                  fontsize=10.5, color=TEXT_PRIMARY, fontweight="medium")
    if subtitle:
        grid.fig.text(x0 + grid.bw / 2, y0 + grid.bh * 0.885, subtitle, ha="center",
                      va="center", fontsize=7.5, color=subtitle_color, fontweight="medium")


def row_label(grid: BlockGrid, row: int, text: str, color: str = TEXT_PRIMARY) -> None:
    """A rotated label down the left edge of one row of blocks."""
    _, y0 = grid.origin(row * grid.ncol)
    grid.fig.text(0.004, y0 + grid.bh * 0.52, text, rotation=90, va="center", ha="center",
                  fontsize=9.5, color=color, fontweight="medium")


def block_colorbar(grid: BlockGrid, i: int, cmap, vmin: float, vmax: float, label: str = "",
                   nbins: int = 4) -> None:
    """Horizontal colourbar under one block, for figures with a per-block scale."""
    x0, y0 = grid.origin(i)
    cax = grid.fig.add_axes([x0 + grid.bw * 0.22, y0 + grid.bh * 0.11, grid.bw * 0.56,
                             grid.bh * 0.018])
    cb = grid.fig.colorbar(mpl.cm.ScalarMappable(
        cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)), cax=cax,
        orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5, length=2, pad=1.5, color=TEXT_SECONDARY)
    cb.locator = mpl.ticker.MaxNLocator(nbins=nbins)
    cb.update_ticks()
    if label:
        cb.set_label(label, fontsize=7, color=TEXT_SECONDARY)


def shared_colorbar(grid: BlockGrid, cmap, vmin: float, vmax: float, label: str, ticks=None,
                    ticklabels=None, width: float = 0.32) -> None:
    """One colourbar in the reserved footer, for figures where every block shares a scale."""
    cax = grid.fig.add_axes([0.5 - width / 2, 0.20 / grid.height_in, width,
                             0.09 / grid.height_in])
    cb = grid.fig.colorbar(mpl.cm.ScalarMappable(
        cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)), cax=cax,
        orientation="horizontal")
    cb.outline.set_visible(False)
    if ticks is not None:
        cb.set_ticks(ticks)
    if ticklabels is not None:
        cb.set_ticklabels(ticklabels)
    cb.ax.tick_params(labelsize=6.5, length=2, pad=1.5, color=TEXT_SECONDARY)
    cb.set_label(label, fontsize=7.5, color=TEXT_SECONDARY)


def suptitle(grid: BlockGrid, text: str, fontsize: float = 11) -> None:
    grid.fig.suptitle(text, x=0.012, ha="left", fontsize=fontsize,
                      y=1 - 0.06 / grid.height_in, va="top")


def save(grid: BlockGrid, path: Path, dpi: int = FIG_DPI) -> None:
    grid.fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(grid.fig)


# ----------------------------------------------------------------------------------
# statistics used by the two correlation scripts
# ----------------------------------------------------------------------------------
MIN_PAIRS = 10          # below this a Spearman rho is not worth reporting


def spearman(x: pd.Series, y: pd.Series, min_pairs: int = MIN_PAIRS) -> dict:
    """Spearman rho/p/n, or NaN where the cell is too thin or too degenerate to mean much.

    Spearman rather than Pearson everywhere here: n is 52 at most, and several of the EEG
    features are strongly skewed, so a rank correlation is the honest choice.
    """
    from scipy.stats import spearmanr

    ok = x.notna() & y.notna()
    if ok.sum() < min_pairs or x[ok].nunique() < 3 or y[ok].nunique() < 3:
        return {"rho": np.nan, "p": np.nan, "n": int(ok.sum())}
    r = spearmanr(x[ok], y[ok])
    return {"rho": float(r.statistic), "p": float(r.pvalue), "n": int(ok.sum())}


def signed_log10p(out: pd.DataFrame) -> pd.DataFrame:
    """sign(rho) * -log10(p): one number carrying both the direction and the strength of the
    evidence, which is what the red/white/green significance ramp is keyed on."""
    out["signed_log10p"] = np.sign(out.rho) * -np.log10(out.p)
    return out


def rho_critical(n: int, alpha: float = ALPHA) -> float:
    """Smallest |rho| that reaches `alpha` at this n, via the usual t approximation.

    Printed on the figures. At n=52 it is 0.27; at the cluster sizes (7-14) it is 0.54-0.79,
    which is why the cluster figures colour by rho and not by significance.
    """
    from scipy.stats import t as student_t

    tc = student_t.ppf(1 - alpha / 2, n - 2)
    return float(np.sqrt(tc ** 2 / (tc ** 2 + n - 2)))
