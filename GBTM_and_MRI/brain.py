"""Cortical surface rendering: Desikan-Killiany values -> an fsaverage5 2x2 brain block.

Self-contained on purpose. The parcellation is read from the two `*.aparc.annot` files in
`data/` rather than fetched, so nothing here downloads a 700 MB FreeSurfer subject; the
fsaverage5 meshes themselves ship inside nilearn, so they cost nothing either.

fsaverage5's vertices are the first 10242 of fsaverage (icosahedral nesting), which is why
the full-resolution annot can simply be truncated to address the coarse mesh.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from config import ANNOT

MESH = "fsaverage5"
HEMIS = ("lh", "rh")
VIEWS_2x2 = [("left", "lateral"), ("right", "lateral"),
             ("left", "medial"), ("right", "medial")]

# Red/blue diverging, for a between-class difference. The mid tone is off-white rather than
# pure white so a near-zero region still reads as painted rather than as missing.
CMAP_DIFF = LinearSegmentedColormap.from_list(
    "diff_blue_red", ["#2166ac", "#67a9cf", "#d1e5f0", "#f7f7f5", "#fddbc7", "#ef8a62",
                      "#b2182b"])
CMAP_SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f7fbff", "#d6e6f7", "#a8cbec", "#6aaad8", "#2f7fc1"])


@lru_cache(maxsize=1)
def _fsaverage():
    from nilearn import datasets

    return datasets.fetch_surf_fsaverage(mesh=MESH)


@lru_cache(maxsize=1)
def _n_vertices() -> int:
    from nilearn.surface import load_surf_mesh

    return int(load_surf_mesh(_fsaverage()["pial_left"]).coordinates.shape[0])


@lru_cache(maxsize=1)
def _roi_index() -> dict[str, dict[str, np.ndarray]]:
    """{hemi: {roi name: vertex indices}} -- computed once and reused.

    Done as one sort per hemisphere rather than a full string comparison per ROI; with 68
    ROIs and several maps per figure the naive version dominates the render.
    """
    nv, idx = _n_vertices(), {}
    for hemi in HEMIS:
        labels, _, names = nib.freesurfer.read_annot(str(ANNOT[hemi]))
        names = [n.decode() if isinstance(n, bytes) else n for n in names]
        per_vertex = np.array([f"ctx_{hemi}_{names[i]}" if 0 <= i < len(names) else ""
                               for i in labels])[:nv]
        order = np.argsort(per_vertex, kind="stable")
        srt = per_vertex[order]
        uniq, starts = np.unique(srt, return_index=True)
        bounds = list(starts) + [len(srt)]
        idx[hemi] = {name: order[bounds[i]:bounds[i + 1]]
                     for i, name in enumerate(uniq) if name}
    return idx


def roi_values_to_vertices(values: dict[str, float], fill: float = np.nan) -> dict:
    nv, index = _n_vertices(), _roi_index()
    out = {}
    for hemi in HEMIS:
        arr = np.full(nv, fill, dtype=float)
        for roi, v in values.items():
            ids = index[hemi].get(roi)
            if ids is not None and v is not None and np.isfinite(v):
                arr[ids] = v
        out[hemi] = arr
    return out


@dataclass(frozen=True)
class Block:
    """Figure-fraction geometry for one 2x2 brain block."""

    fig: plt.Figure
    left: float
    right: float
    top: float
    bottom: float

    @property
    def w(self) -> float:
        return self.right - self.left

    @property
    def h(self) -> float:
        return self.top - self.bottom


def draw(block: Block, values: dict[str, float], cmap, vmin: float, vmax: float,
         fill: float = np.nan, zoom: float = 1.5) -> None:
    """Render LH/RH x lateral/medial into `block`.

    A NaN `fill` is passed straight through to nilearn, which leaves those vertices as plain
    sulcal background instead of painting them. That is what the medial wall and the
    subcortical territory need: they carry no cortical value, and filling them with vmin
    (dark) or with the zero of a diverging ramp (white) both read as a measurement -- "most
    negative here", "no effect here" -- when the truth is "not measured".

    `zoom` above ~1.6 starts clipping the frontal and parietal edges off the lateral views,
    because the 3d axes clip at their bounds. Enlarge the block instead of raising it.
    """
    from nilearn import plotting as niplot

    fs = _fsaverage()
    arrays = roi_values_to_vertices(values, fill=fill)
    panel_w, panel_h = block.w * 0.48, block.h * 0.40
    for j, (hemi, view) in enumerate(VIEWS_2x2):
        r, c = divmod(j, 2)
        ax = block.fig.add_axes([block.left + block.w * 0.02 + c * panel_w,
                                 block.bottom + block.h * 0.50 - r * panel_h,
                                 panel_w, panel_h], projection="3d")
        data = arrays["lh" if hemi == "left" else "rh"]
        if np.isfinite(fill):
            data = np.nan_to_num(data, nan=fill)
        niplot.plot_surf(fs[f"pial_{hemi}"], data, hemi=hemi, view=view,
                         bg_map=fs[f"sulc_{hemi}"], bg_on_data=True, cmap=cmap,
                         vmin=vmin, vmax=vmax, colorbar=False, axes=ax, alpha=1.0)
        ax.set_box_aspect(None, zoom=zoom)


def colorbar(block: Block, cmap, vmin: float, vmax: float, label: str,
             label_size: float = 11, tick_size: float = 9.5) -> None:
    cax = block.fig.add_axes([block.left + block.w * 0.22, block.bottom + block.h * 0.11,
                              block.w * 0.56, block.h * 0.018])
    cb = block.fig.colorbar(mpl.cm.ScalarMappable(
        cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)),
        cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=tick_size, length=2, pad=1.5, color="#555555")
    cb.locator = mpl.ticker.MaxNLocator(nbins=4)
    cb.update_ticks()
    cb.set_label(label, fontsize=label_size, color="#555555")
