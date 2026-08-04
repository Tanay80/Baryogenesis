#!/usr/bin/env python
# coding: utf-8

# # Notebook to create figures in the paper "The local galaxy distribution does not violate the cosmological principle"
# 
# https://arxiv.org/abs/2607.01172
# 
# (c) Till Sawala, 2026. You are free to use any part of this notebook, but if you use it in your own work, I would appreciate a citation to the above paper and / or to this Github.
# 
# This code will download data from DESI DR1, Flamingo, and the S2 sample of Sylos Labini & Galoppo (2026). Please note that if you use any of the data (including through this notebook), it is your responsibility to comply with the creators use policies, including their the acknowledgement policies:
# 
# https://data.desi.lbl.gov/doc/acknowledgments/
# https://dataweb.cosma.dur.ac.uk:8443/flamingo/acknowledgements.html
# 
# The computation of the "ADPD" and the associated plots borrow from the following code by Marco Galoppo:
# by https://github.com/MarcoGaloppo/Code-and-Data-Detection-of-anisotropic-cosmic-structures-on-a-gigaparsec-scale associated with this paper: https://www.nature.com/articles/s41586-026-10702-5 by Francesco Sylos Labini & Marco Galoppo. 
# If you use this part of the notebook, please refer to the code by Marco Galoppo and the paper by Francesco Sylos Labini & Marco Galoppo.
# 
# For any questions, please contact me via email: till.sawala@helsinki.fi
# 
# Please note that the notebook is designed to process one snapshot at a time (snapshot 75, z=0.15, by default). To download and process other snapshots, change the FLAMINGO_SNAPSHOT variable.
# 
# The code was created in part with assistance from ChatGPT Version 5.5.

# In[3]:


# less common packages you may need to install
# %pip install hdfstream h5py astropy requests


# In[4]:


from astropy.coordinates import SkyCoord
import astropy.units as u


# In[5]:


import anisotropic_cosmology
import importlib

importlib.reload(anisotropic_cosmology)

print(dir(anisotropic_cosmology))


# In[6]:


from anisotropic_cosmology import ang_sep


# In[7]:


from anisotropic_cosmology import anisotropic_distance_array


# In[8]:


# ---------------------------------------------------------------------------
# Imports and global settings
# ---------------------------------------------------------------------------
from pathlib import Path
from urllib.request import urlretrieve
import urllib.error
import io
import math
import time
import warnings

import h5py
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter

from tqdm.auto import tqdm

try:
    import hdfstream
except ImportError:
    hdfstream = None

import astropy.units as u
from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy.table import Table, vstack

# import anisotropic_cosmology
# from astropy.coordinates import SkyCoord
# from anisotropic_cosmology import (
#     ang_sep,
#     anisotropic_distance_array,
# )

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "font.size": 10,
})

DATA_DIR = Path("data")
FIGURES_DIR = Path("figures")
DATA_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

SAVE_FIGURES = True

# ---------------------------------------------------------------------------
# FLAMINGO D3A cosmology and box convention
# ---------------------------------------------------------------------------
FLAMINGO_H = 0.681
FLAMINGO_OMEGA_M = 0.306
FLAMINGO_OMEGA_B = 0.0486
FLAMINGO_OMEGA_LAMBDA = 0.694
FLAMINGO_SIGMA8 = 0.807
FLAMINGO_NS = 0.967
FLAMINGO_SUM_MNU_EV = 0.06

COSMO = FlatLambdaCDM(
    H0=100.0 * FLAMINGO_H,
    Om0=FLAMINGO_OMEGA_M,
    Ob0=FLAMINGO_OMEGA_B,
    Tcmb0=2.7255,
)

# FLAMINGO L1 box size is fixed in comoving units.
BOX_SIZE_CMPC = 1000.0
BOX_SIZE_CMPC_OVER_H = BOX_SIZE_CMPC * FLAMINGO_H

HDFSTREAM_SERVER = "cosma"

FLAMINGO_SIMULATION = "L1_m8"

FLAMINGO_SNAPSHOT = "0073" # z=0.25
FLAMINGO_SNAPSHOT = "0075" # z=0.15          # this is the default in all plots
#FLAMINGO_SNAPSHOT = "0077" # z=0.05

# Use the L1_m9 DMO run for the DM-particle comparison. For this pair of
# simulations, the full-particle snapshot number is one less than the
# SOAP-HBT galaxy-catalogue snapshot number at the same redshift.
FLAMINGO_DM_SIMULATION = "L1_m9"
FLAMINGO_DM_RUN = "L1_m9_DMO"
FLAMINGO_DM_SNAPSHOT = f"{int(FLAMINGO_SNAPSHOT) - 1:04d}"


def make_snapshot_file_tag(snapshot):
    """Return a compact snapshot tag such as snap_077 for filenames."""
    text = str(snapshot)
    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        return f"snap_{int(digits):03d}"
    return f"snap_{text}"


FLAMINGO_SNAPSHOT_FILE_TAG = make_snapshot_file_tag(FLAMINGO_SNAPSHOT)
FLAMINGO_DM_SNAPSHOT_FILE_TAG = make_snapshot_file_tag(FLAMINGO_DM_SNAPSHOT)

FLAMINGO_REMOTE_PATH = (
    f"FLAMINGO/{FLAMINGO_SIMULATION}/{FLAMINGO_SIMULATION}/"
    f"SOAP-HBT/halo_properties_{FLAMINGO_SNAPSHOT}.hdf5"
)

FLAMINGO_DM_REMOTE_PATH = (
    f"FLAMINGO/{FLAMINGO_DM_SIMULATION}/{FLAMINGO_DM_RUN}/"
    f"snapshots/flamingo_{FLAMINGO_DM_SNAPSHOT}/flamingo_{FLAMINGO_DM_SNAPSHOT}.hdf5"
)

POSITION_DATASET = "BoundSubhalo/CentreOfMass"
VELOCITY_DATASET = "BoundSubhalo/CentreOfMassVelocity"
LUMINOSITY_DATASET = "BoundSubhalo/StellarLuminosity"
BANDS = ("u", "g", "r", "i", "z", "Y", "J", "H", "K")
FLAMINGO_LUMINOSITY_BAND = "r"
FLAMINGO_BAND_INDEX = BANDS.index(FLAMINGO_LUMINOSITY_BAND)
FLAMINGO_LUMINOSITY_OUTPUT_DATASET = f"{FLAMINGO_LUMINOSITY_BAND}_luminosity"
FLAMINGO_MAG_OUTPUT_DATASET = f"M_{FLAMINGO_LUMINOSITY_BAND}_AB"

DM_POSITION_DATASET = "PartType1/Coordinates"
N_DM_RANDOM = 5_000_000 # 5M particles is sufficient to replicate the density to M < -20.
DM_RANDOM_SEED = 0
DM_MATCH_SEED = 0
DM_STREAM_BLOCK_SIZE = 250_000

# One compact local file per simulation, snapshot, and band.
FLAMINGO_COMPACT_FILE = (
    DATA_DIR
    / f"flamingo_{FLAMINGO_SIMULATION}_{FLAMINGO_SNAPSHOT}_"
      f"galaxies_{FLAMINGO_LUMINOSITY_BAND}band_comoving_velocity.hdf5"
)

FLAMINGO_DM_COMPACT_FILE = (
    DATA_DIR
    / f"flamingo_{FLAMINGO_DM_SIMULATION}_{FLAMINGO_DM_RUN}_{FLAMINGO_DM_SNAPSHOT}_"
      f"dm_random_{N_DM_RANDOM}_positions_seed{DM_RANDOM_SEED}.hdf5"
)

OVERWRITE_FLAMINGO_COMPACT_FILE = False               # otherwise data is re-downloaded.
OVERWRITE_FLAMINGO_DM_COMPACT_FILE = False
STREAM_BLOCK_SIZE = 1_000_000
MAX_REMOTE_ROWS = None
COMPRESSION = "gzip"
COMPRESSION_OPTS = 4
VELOCITY_TO_KMS = 1.0

# ---------------------------------------------------------------------------
# DESI/S2/SDSS inputs
# ---------------------------------------------------------------------------
DESI_S2_URL = "https://zenodo.org/records/20118015/files/points_S2.dat?download=1"
DESI_S2_FILE = DATA_DIR / "points_S2.dat"

DESI_BASE_URL = "https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5"
DESI_FILES = [
    "BGS_BRIGHT_NGC_clustering.dat.fits",
    "BGS_BRIGHT_SGC_clustering.dat.fits",
]

# DESI/S2 footprint reconstruction parameters inherited from the current notebook.
RA_MIN_R25, RA_MAX_R25 = 127.0, 225.0
DEC_MIN_R25, DEC_MAX_R25 = -7.0, 3.0
D_MAX_HMPC = 680.0
N_PUBLISHED_VL3_R25 = 100_209

X_CENTER_BASE = 225.0
Y_CENTER_BASE = 28.0
Z_CENTER = 0.0
DX_PLOT = 10.0
DY_PLOT = -59.0
SCALE_TOTAL = 1.63

nanomaggy = u.def_unit("nanomaggy", 1e-9 * u.mgy) # These are in the DESI fits files
u.add_enabled_units([nanomaggy])

# Broad working cut for reconstructing the footprint.  The actual comparison
# cylinders apply their own cuts below.
MR_PROXY_CUT_WORKING = -20.0

# ---------------------------------------------------------------------------
# Requested DESI/FLAMINGO comparison cylinders
# ---------------------------------------------------------------------------
# Observer-centred coordinates use x<0 away from the observer.  The cylinder
# axis is the z direction; the circular aperture lies in the x-y plane.
COMPARISON_REGIONS = {
    "large": {
        "label": "large",
        "radius_hmpc": 290.0,
        "center_x_hmpc": -390.0,
        "center_y_hmpc": 0.0,
        "center_z_hmpc": 0.0,
        "desi_mr_limit": -21.5,
    },
    "small": {
        "label": "small",
        "radius_hmpc": 175.0,
        "center_x_hmpc": -241.0,
        "center_y_hmpc": 0.0,
        "center_z_hmpc": 0.0,
        "desi_mr_limit": -20.0,
    },
}

CYLINDER_THICKNESS_HMPC = 40.0
CYLINDER_HALF_THICKNESS_HMPC = 0.5 * CYLINDER_THICKNESS_HMPC

# Scatter and P(k) settings.
N_SCATTER_SIM_SLICES = 17
N_POWER_SIM_SLICES = None
NGRID_POWER = 512
N_K_BINS = 30
POWER_K_NYQUIST_FRACTION = 0.70
POWER_FIGSIZE = (4.2, 3.2)
POWER_LEGEND_FONTSIZE = 8

CASE_ORDER = ["real", "rsd"]
CASE_LABELS = {
    "real": "FLAMINGO, real-space",
    "rsd": "FLAMINGO + RSD",
}
CASE_FILE_TAGS = {
    "real": "real",
    "rsd": "rsd",
}
CASE_COLORS = {
    "real": "cornflowerblue",
    "rsd": "hotpink",
}

# Three-panel footprint plot settings.
FULL_FOOTPRINT_MR_LIMIT = -20.0
S2_FIDUCIAL_RADIUS_HMPC = 300.0
COMOVING_SHELL_SPACING_HMPC = 50.0
LUMINOSITY_DISTANCE_SHELL_SPACING_MPC = 100.0

ADPD_N_JOBS = 8 # APDP tasks to run in parallel.


print("Notebook configuration")
print("----------------------")
print(f"FLAMINGO galaxy simulation : {FLAMINGO_SIMULATION}")
print(f"FLAMINGO galaxy snapshot   : {FLAMINGO_SNAPSHOT}")
print(f"FLAMINGO DM simulation     : {FLAMINGO_DM_SIMULATION}/{FLAMINGO_DM_RUN}")
print(f"FLAMINGO DM snapshot       : {FLAMINGO_DM_SNAPSHOT}")
print(f"FLAMINGO galaxy remote path: {FLAMINGO_REMOTE_PATH}")
print(f"FLAMINGO DM remote path    : {FLAMINGO_DM_REMOTE_PATH}")
print(f"FLAMINGO galaxy cache      : {FLAMINGO_COMPACT_FILE}")
print(f"FLAMINGO DM cache          : {FLAMINGO_DM_COMPACT_FILE}")
print(f"DM random subset           : N={N_DM_RANDOM:,}, seed={DM_RANDOM_SEED}")
print(f"FLAMINGO h                 : {FLAMINGO_H}")
print(f"FLAMINGO box               : {BOX_SIZE_CMPC:g} cMpc = {BOX_SIZE_CMPC_OVER_H:g} h^-1 cMpc")
print(f"FLAMINGO luminosity band   : {FLAMINGO_LUMINOSITY_BAND}")
print(f"Cylinder thickness         : {CYLINDER_THICKNESS_HMPC:g} h^-1 cMpc")
for name, region in COMPARISON_REGIONS.items():
    print(
        f"{name:>5s}: R={region['radius_hmpc']:g} h^-1 Mpc, "
        f"centre=({region['center_x_hmpc']:g}, {region['center_y_hmpc']:g}, {region['center_z_hmpc']:g}) h^-1 Mpc, "
        f"DESI M_r<{region['desi_mr_limit']:g}"
    )


# ## Helper functions
# 
# The functions below handle cached downloads, coordinate utilities, circular apertures, and figure output.
# 

# In[9]:


# ---------------------------------------------------------------------------
# General, download, coordinate, and plotting helpers
# ---------------------------------------------------------------------------
def download_with_progress(url, output_path):
    """Download a URL to a local path with a tqdm progress report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")

    pbar = tqdm(unit="B", unit_scale=True, unit_divisor=1024, desc=f"Downloading {output_path.name}")

    def reporthook(block_num, block_size, total_size):
        if total_size > 0:
            pbar.total = total_size
        downloaded = block_num * block_size
        delta = downloaded - pbar.n
        if delta > 0:
            pbar.update(delta)

    try:
        urlretrieve(url, tmp_path, reporthook=reporthook)
        tmp_path.replace(output_path)
    except urllib.error.HTTPError as err:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Could not download {url}: {err}") from err
    finally:
        pbar.close()

    print(f"Downloaded {output_path}")
    return output_path


def ensure_local_file(url, destination):
    """Return a local file, downloading it if needed."""
    destination = Path(destination)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Using existing file: {destination}")
        return destination
    print(f"File not found. Downloading {url}")
    return download_with_progress(url, destination)


def get_col(table, name):
    """Return an Astropy table column by case-insensitive name."""
    lower = {n.lower(): n for n in table.colnames}
    key = name.lower()
    if key not in lower:
        raise KeyError(f"Column {name!r} not found. Available columns include: {table.colnames[:20]} ...")
    return np.asarray(table[lower[key]])


def unit_vector(ra, dec):
    """3D unit vector for spherical coordinates ra, dec in radians."""
    return np.column_stack([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec),
    ])


def add_snapshot_tag_to_path(outfile):
    """Return a PDF path whose stem includes the active FLAMINGO snapshot tag."""
    outfile = Path(outfile)

    # Enforce PDF-only plot output.  If a plotting cell accidentally passes
    # another suffix, save the figure as a PDF with the same stem.
    outfile = outfile.with_suffix(".pdf")

    tag = globals().get("FLAMINGO_SNAPSHOT_FILE_TAG", None)
    if tag is None:
        tag = make_snapshot_file_tag(globals().get("FLAMINGO_SNAPSHOT", "unknown"))

    if f"_{tag}" not in outfile.stem:
        outfile = outfile.with_name(f"{outfile.stem}_{tag}{outfile.suffix}")

    return outfile


def save_figure(fig, outfile, *, pad_inches=0.08):
    """Save a figure as a snapshot-tagged PDF if SAVE_FIGURES=True."""
    outfile = add_snapshot_tag_to_path(outfile)
    outfile.parent.mkdir(exist_ok=True, parents=True)

    if SAVE_FIGURES:
        fig.savefig(outfile, bbox_inches="tight", pad_inches=pad_inches, dpi='figure')
        print(f"Saved {outfile}")

    plt.show()

def draw_circle(
    ax,
    centre,
    radius,
    *,
    color="black",
    linestyle="-",
    lw=1.0,
    alpha=1.0,
    zorder=3,
    fill=False,
    clip_on=True,
):
    """Draw a circle clipped to the current axes."""
    circle = plt.Circle(
        centre,
        radius,
        fill=fill,
        edgecolor=color,
        facecolor="none",
        linestyle=linestyle,
        linewidth=lw,
        alpha=alpha,
        zorder=zorder,
        clip_on=clip_on,
    )

    # Explicitly clip the circle to the axes rectangle.
    circle.set_clip_path(ax.patch)

    ax.add_patch(circle)
    return circle

def select_brightest_indices(indices, magnitude, *, n_keep, context=""):
    """Return indices for the brightest n_keep objects according to an AB magnitude array."""
    indices = np.asarray(indices, dtype=np.int64)
    magnitude = np.asarray(magnitude, dtype=float)
    valid = np.isfinite(magnitude[indices])
    indices = indices[valid]

    n_available = len(indices)
    n_keep = int(n_keep)

    if n_available < n_keep:
        warnings.warn(
            f"{context}: only {n_available:,} objects available; requested {n_keep:,}. "
            "Keeping all available objects.",
            RuntimeWarning,
        )
        return indices, n_available

    order = np.argsort(magnitude[indices], kind="mergesort")
    return indices[order[:n_keep]], n_available


def hmpc_to_cmpc(distance_hmpc, h=FLAMINGO_H):
    """Convert h^-1 cMpc to cMpc."""
    return float(distance_hmpc) / float(h)


def cmpc_to_hmpc(distance_cmpc, h=FLAMINGO_H):
    """Convert cMpc to h^-1 cMpc."""
    return float(distance_cmpc) * float(h)


def finite_range(values):
    """Return a finite min/max tuple for diagnostics."""
    values = np.asarray(values, dtype=float)
    good = np.isfinite(values)
    if not np.any(good):
        return np.nan, np.nan
    return np.nanmin(values[good]), np.nanmax(values[good])


def shell_radii_hmpc(max_radius_hmpc, spacing_hmpc):
    """Radii of comoving-distance shells in h^-1 Mpc."""
    return np.arange(spacing_hmpc, max_radius_hmpc + 0.5 * spacing_hmpc, spacing_hmpc)


def luminosity_distance_shells_as_comoving_hmpc(max_radius_hmpc, spacing_mpc):
    """
    Convert luminosity-distance shell labels in Mpc to plotted comoving radii in h^-1 Mpc.

    The shell labelled D_L is drawn at chi(z) where D_L(z)=D_L, converted to h^-1 Mpc.
    """
    shells = []
    d_l = float(spacing_mpc)
    max_dl_to_try = 10_000.0
    while d_l <= max_dl_to_try:
        try:
            z = z_at_value(COSMO.luminosity_distance, d_l * u.Mpc, zmin=1e-8, zmax=5.0)
            chi_hmpc = FLAMINGO_H * COSMO.comoving_distance(z).to_value(u.Mpc)
        except Exception:
            break

        if chi_hmpc > max_radius_hmpc + spacing_mpc:
            break
        shells.append((float(d_l), float(chi_hmpc)))
        d_l += float(spacing_mpc)

    return shells


def add_distance_shells(
    ax,
    *,
    max_radius_hmpc,
    comoving_spacing_hmpc=COMOVING_SHELL_SPACING_HMPC,
    luminosity_spacing_mpc=None,
    add_labels=False,
):
    """Add constant-distance shells in the plotted x-y plane."""
    for r in shell_radii_hmpc(max_radius_hmpc, comoving_spacing_hmpc):
        draw_circle(
            ax,
            (0.0, 0.0),
            r,
            color="0.65",
            linestyle="--",
            lw=0.45,
            alpha=0.55,
            zorder=1,
        )
        if add_labels and r <= max_radius_hmpc:
            ax.text(-r / np.sqrt(2), r / np.sqrt(2), f"{r:.0f}", fontsize=5, color="0.45")

    if luminosity_spacing_mpc is not None:
        for d_l, r_chi in luminosity_distance_shells_as_comoving_hmpc(
            max_radius_hmpc,
            luminosity_spacing_mpc,
        ):
            draw_circle(
                ax,
                (0.0, 0.0),
                r_chi,
                color="0.25",
                linestyle=":",
                lw=0.65,
                alpha=0.70,
                zorder=1,
            )
            if add_labels:
                ax.text(r_chi / np.sqrt(2), r_chi / np.sqrt(2), f"$D_L={d_l:.0f}$", fontsize=5, color="0.25")


# ## FLAMINGO compact galaxy and DM-particle caches
# 
# Create compact local HDF5 files for the SOAP-HBT galaxy catalogue and a random, sorted-index subset of L1_m9 DMO DM-particle positions at the matching redshift.

# In[10]:


# ---------------------------------------------------------------------------
# hdfstream / HDF5 helper functions for a snapshot-specific compact cache
# ---------------------------------------------------------------------------
def _as_scalar_float(value):
    """Best-effort conversion of an HDF5 attribute value to a scalar float."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        cleaned = value.strip().replace(",", " ").split()[0]
        return float(cleaned)
    arr = np.asarray(value)
    if arr.shape == ():
        return float(arr)
    if arr.size == 1:
        return float(arr.reshape(-1)[0])
    raise ValueError(f"Cannot convert non-scalar value to float: {value!r}")


def _get_attrs(obj):
    """Return an HDF5/hdfstream attribute mapping if available."""
    if hasattr(obj, "attrs"):
        try:
            return obj.attrs
        except Exception:
            pass
    if hasattr(obj, "attrsIn"):
        try:
            return obj.attrsIn
        except Exception:
            pass
    return {}


def try_get_attr(obj, candidate_keys):
    """Return (key, value) for the first matching attribute, or (None, None)."""
    attrs = _get_attrs(obj)

    for key in candidate_keys:
        try:
            if key in attrs:
                return key, attrs[key]
        except Exception:
            pass

    norm_candidates = {
        key.lower().replace(" ", "").replace("_", "").replace("-", "")
        for key in candidate_keys
    }
    try:
        for key in attrs.keys():
            norm_key = str(key).lower().replace(" ", "").replace("_", "").replace("-", "")
            if norm_key in norm_candidates:
                return key, attrs[key]
    except Exception:
        pass

    return None, None


def open_hdfstream_root():
    """Open the FLAMINGO hdfstream root."""
    if hdfstream is None:
        raise ImportError(
            "Missing dependency: hdfstream. Install it with\n"
            "    python -m pip install --user hdfstream h5py numpy matplotlib tqdm astropy pandas requests\n"
            "or uncomment the pip-install cell if running this notebook on a fresh environment."
        )

    print(f"Opening hdfstream server: {HDFSTREAM_SERVER}")
    root = hdfstream.open(HDFSTREAM_SERVER, "/")
    return root, HDFSTREAM_SERVER


def read_snapshot_metadata(remote_file):
    """Read scale factor and redshift from common metadata locations."""
    scale_keys = ["Scale-factor", "ScaleFactor", "a", "ExpansionFactor", "Expansion factor"]
    redshift_keys = ["Redshift", "redshift", "z"]

    group_paths = ["", "Header", "Cosmology", "Parameters", "SWIFT", "SWIFT/Cosmology"]

    a_value = None
    a_source = None
    z_value = None
    z_source = None

    for path in group_paths:
        try:
            obj = remote_file if path == "" else remote_file[path]
        except Exception:
            continue

        if a_value is None:
            key, value = try_get_attr(obj, scale_keys)
            if key is not None:
                try:
                    candidate = _as_scalar_float(value)
                    if 0.0 < candidate <= 1.1:
                        a_value = candidate
                        a_source = f"{path + '/' if path else ''}{key}"
                except Exception:
                    pass

        if z_value is None:
            key, value = try_get_attr(obj, redshift_keys)
            if key is not None:
                try:
                    candidate = _as_scalar_float(value)
                    if candidate >= -1e-5:
                        z_value = candidate
                        z_source = f"{path + '/' if path else ''}{key}"
                except Exception:
                    pass

    if a_value is None and z_value is not None:
        a_value = 1.0 / (1.0 + z_value)
        a_source = "derived from redshift"
    if z_value is None and a_value is not None:
        z_value = 1.0 / a_value - 1.0
        z_source = "derived from scale factor"

    if a_value is None or z_value is None:
        raise RuntimeError(
            "Could not read a valid snapshot scale factor or redshift from the remote HDF5 metadata."
        )

    return float(a_value), float(z_value), a_source, z_source


def open_dataset(remote_file, dataset_name, *, kind="dataset"):
    """Open a required remote HDF5 dataset."""
    try:
        return remote_file[dataset_name], dataset_name
    except Exception as exc:
        raise RuntimeError(f"Could not open required {kind} dataset {dataset_name!r}.") from exc


def copy_attrs_best_effort(src, dst):
    """Copy HDF5 attributes where possible, converting awkward values to repr strings."""
    attrs = _get_attrs(src)
    try:
        items = list(attrs.items())
    except Exception:
        return
    for key, value in items:
        try:
            dst.attrs[key] = value
        except Exception:
            try:
                dst.attrs[key] = repr(value)
            except Exception:
                pass


def compact_file_is_usable(path):
    """Check whether the compact FLAMINGO file has the datasets needed below."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as f:
            return (
                "position_cMpc" in f
                and "velocity_kms" in f
                and FLAMINGO_LUMINOSITY_OUTPUT_DATASET in f
                and FLAMINGO_MAG_OUTPUT_DATASET in f
                and f["position_cMpc"].ndim == 2
                and f["position_cMpc"].shape[1] == 3
                and f["velocity_kms"].ndim == 2
                and f["velocity_kms"].shape[1] == 3
                and f["velocity_kms"].shape[0] == f["position_cMpc"].shape[0]
                and len(f[FLAMINGO_MAG_OUTPUT_DATASET]) == f["position_cMpc"].shape[0]
            )
    except Exception:
        return False


def ensure_flamingo_compact_catalogue():
    """Create FLAMINGO_COMPACT_FILE by streaming only the needed columns."""
    if compact_file_is_usable(FLAMINGO_COMPACT_FILE) and not OVERWRITE_FLAMINGO_COMPACT_FILE:
        print(f"Using existing compact FLAMINGO file: {FLAMINGO_COMPACT_FILE}")
        return

    if hdfstream is None:
        raise ImportError(
            "Missing dependency: hdfstream. Install it with\n"
            "    python -m pip install --user hdfstream h5py numpy matplotlib tqdm astropy pandas requests\n"
            "or uncomment the pip-install cell if running this notebook on a fresh environment."
        )

    if STREAM_BLOCK_SIZE <= 0:
        raise ValueError("STREAM_BLOCK_SIZE must be positive")

    print("Compact FLAMINGO file not found, not usable, or overwrite requested.")
    root, resolved_hdfstream_server = open_hdfstream_root()

    print(f"Opening remote file: {FLAMINGO_REMOTE_PATH}")
    remote_file = root[FLAMINGO_REMOTE_PATH]

    pos_ds, pos_name = open_dataset(remote_file, POSITION_DATASET, kind="position")
    vel_ds, vel_name = open_dataset(remote_file, VELOCITY_DATASET, kind="velocity")
    lum_ds, lum_name = open_dataset(remote_file, LUMINOSITY_DATASET, kind="luminosity")

    scale_factor, redshift, a_source, z_source = read_snapshot_metadata(remote_file)
    print(f"Snapshot scale factor = {scale_factor:.8f} ({a_source})")
    print(f"Snapshot redshift     = {redshift:.8f} ({z_source})")

    n_rows_available = int(lum_ds.shape[0])
    n_rows_to_process = n_rows_available if MAX_REMOTE_ROWS is None else min(n_rows_available, int(MAX_REMOTE_ROWS))

    print(f"Position dataset      : {pos_name}, shape={pos_ds.shape}")
    print(f"Velocity dataset      : {vel_name}, shape={vel_ds.shape}")
    print(f"Luminosity dataset    : {lum_name}, shape={lum_ds.shape}")
    print(f"Selected band         : {FLAMINGO_LUMINOSITY_BAND}, column {FLAMINGO_BAND_INDEX}")
    print(f"Rows available        : {n_rows_available:,}")
    print(f"Rows to process       : {n_rows_to_process:,}")

    if len(pos_ds.shape) != 2 or pos_ds.shape[1] != 3:
        raise ValueError(f"Expected position dataset with shape (N, 3), got {pos_ds.shape}")
    if len(vel_ds.shape) != 2 or vel_ds.shape[1] != 3:
        raise ValueError(f"Expected velocity dataset with shape (N, 3), got {vel_ds.shape}")
    if len(lum_ds.shape) != 2 or lum_ds.shape[1] <= FLAMINGO_BAND_INDEX:
        raise ValueError(
            f"Expected luminosity dataset with column {FLAMINGO_BAND_INDEX}, got shape {lum_ds.shape}"
        )

    tmp_file = FLAMINGO_COMPACT_FILE.with_suffix(FLAMINGO_COMPACT_FILE.suffix + ".tmp")
    if tmp_file.exists():
        tmp_file.unlink()

    n_blocks = math.ceil(n_rows_to_process / STREAM_BLOCK_SIZE)
    print(f"Writing compact catalogue to temporary file: {tmp_file}")
    print(f"Streaming {n_rows_to_process:,} rows in {n_blocks:,} blocks")

    start_time = time.time()
    written = 0
    invalid_or_empty = 0

    with h5py.File(tmp_file, "w") as out:
        out.attrs["source_server"] = resolved_hdfstream_server
        out.attrs["source_remote_path"] = FLAMINGO_REMOTE_PATH
        out.attrs["snapshot"] = FLAMINGO_SNAPSHOT
        out.attrs["simulation"] = FLAMINGO_SIMULATION
        out.attrs["source_position_dataset"] = pos_name
        out.attrs["source_velocity_dataset"] = vel_name
        out.attrs["source_luminosity_dataset"] = lum_name
        out.attrs["stellar_luminosity_bands"] = ",".join(BANDS)
        out.attrs["selected_luminosity_band"] = FLAMINGO_LUMINOSITY_BAND
        out.attrs["selected_band_column_index_zero_based"] = FLAMINGO_BAND_INDEX
        out.attrs["M_band_AB_definition"] = (
            f"{FLAMINGO_MAG_OUTPUT_DATASET} = "
            f"-2.5 * log10({FLAMINGO_LUMINOSITY_OUTPUT_DATASET})"
        )
        out.attrs["h"] = FLAMINGO_H
        out.attrs["Omega_m"] = FLAMINGO_OMEGA_M
        out.attrs["Omega_b"] = FLAMINGO_OMEGA_B
        out.attrs["Omega_Lambda"] = FLAMINGO_OMEGA_LAMBDA
        out.attrs["sigma8"] = FLAMINGO_SIGMA8
        out.attrs["n_s"] = FLAMINGO_NS
        out.attrs["sum_mnu_eV"] = FLAMINGO_SUM_MNU_EV
        out.attrs["scale_factor"] = scale_factor
        out.attrs["redshift"] = redshift
        out.attrs["scale_factor_source"] = a_source
        out.attrs["redshift_source"] = z_source
        out.attrs["box_size_cMpc"] = BOX_SIZE_CMPC
        out.attrs["box_size_hinv_cMpc"] = BOX_SIZE_CMPC_OVER_H
        out.attrs["velocity_to_kms"] = VELOCITY_TO_KMS
        out.attrs["velocity_units_assumed"] = "km/s peculiar velocity before velocity_to_kms factor"

        chunk_rows = min(STREAM_BLOCK_SIZE, 250_000)
        vector_kwargs = dict(maxshape=(None, 3), chunks=(chunk_rows, 3))
        one_d_kwargs = dict(maxshape=(None,), chunks=(chunk_rows,))
        if COMPRESSION is not None:
            vector_kwargs.update(compression=COMPRESSION, compression_opts=COMPRESSION_OPTS)
            one_d_kwargs.update(compression=COMPRESSION, compression_opts=COMPRESSION_OPTS)

        pos_out = out.create_dataset("position_cMpc", shape=(0, 3), dtype="f4", **vector_kwargs)
        vel_out = out.create_dataset("velocity_kms", shape=(0, 3), dtype="f4", **vector_kwargs)
        lum_out = out.create_dataset(FLAMINGO_LUMINOSITY_OUTPUT_DATASET, shape=(0,), dtype="f4", **one_d_kwargs)
        mag_out = out.create_dataset(FLAMINGO_MAG_OUTPUT_DATASET, shape=(0,), dtype="f4", **one_d_kwargs)

        copy_attrs_best_effort(pos_ds, pos_out)
        copy_attrs_best_effort(vel_ds, vel_out)
        copy_attrs_best_effort(lum_ds, lum_out)

        pos_out.attrs["description"] = "Comoving Mpc positions used for slicing"
        vel_out.attrs["description"] = "Peculiar velocities in km/s used for RSD"
        lum_out.attrs["band"] = FLAMINGO_LUMINOSITY_BAND
        mag_out.attrs["description"] = (
            f"Rest-frame, dust-free absolute AB {FLAMINGO_LUMINOSITY_BAND}-band magnitude"
        )
        mag_out.attrs["band"] = FLAMINGO_LUMINOSITY_BAND

        with tqdm(total=n_rows_to_process, desc="Streaming FLAMINGO rows", unit="row") as pbar:
            for iblock, start in enumerate(range(0, n_rows_to_process, STREAM_BLOCK_SIZE), start=1):
                stop = min(start + STREAM_BLOCK_SIZE, n_rows_to_process)
                n_this = stop - start

                positions_block = np.asarray(pos_ds[start:stop, :], dtype=float)
                velocities_block = np.asarray(vel_ds[start:stop, :]) * VELOCITY_TO_KMS
                band_lum = np.asarray(lum_ds[start:stop, FLAMINGO_BAND_INDEX])

                finite_positions = np.all(np.isfinite(positions_block), axis=1)
                finite_velocities = np.all(np.isfinite(velocities_block), axis=1)
                valid_lum = np.isfinite(band_lum) & (band_lum > 0.0)
                valid = finite_positions & finite_velocities & valid_lum

                n_valid = int(np.count_nonzero(valid))
                invalid_or_empty += n_this - n_valid

                if n_valid > 0:
                    positions_valid = positions_block[valid]
                    velocities_valid = velocities_block[valid]
                    band_lum_valid = band_lum[valid]
                    m_band = (-2.5 * np.log10(band_lum_valid)).astype(np.float32)

                    new_size = written + n_valid
                    pos_out.resize((new_size, 3))
                    vel_out.resize((new_size, 3))
                    lum_out.resize((new_size,))
                    mag_out.resize((new_size,))

                    pos_out[written:new_size, :] = positions_valid.astype(np.float32, copy=False)
                    vel_out[written:new_size, :] = velocities_valid.astype(np.float32, copy=False)
                    lum_out[written:new_size] = band_lum_valid.astype(np.float32, copy=False)
                    mag_out[written:new_size] = m_band
                    written = new_size

                pbar.update(n_this)
                elapsed = max(time.time() - start_time, 1e-6)
                pbar.set_postfix(
                    block=f"{iblock}/{n_blocks}",
                    written=f"{written:,}",
                    rate=f"{(start + n_this) / elapsed:,.0f} rows/s",
                )

        out.attrs["n_rows_available_remote"] = n_rows_available
        out.attrs["n_rows_input_processed"] = n_rows_to_process
        out.attrs["n_rows_written"] = written
        out.attrs["n_rows_invalid_or_empty_skipped"] = invalid_or_empty

    tmp_file.replace(FLAMINGO_COMPACT_FILE)
    elapsed = time.time() - start_time
    print(f"Done. Wrote {written:,} objects to {FLAMINGO_COMPACT_FILE}")
    print(
        f"Skipped {invalid_or_empty:,} rows with invalid positions, velocities, "
        f"or {FLAMINGO_LUMINOSITY_BAND}-band luminosity"
    )
    print(f"Elapsed time: {elapsed/60:.2f} minutes")


def dm_particle_file_is_usable(path):
    """Check whether the compact DM-particle file has the datasets needed below."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as f:
            if "position_cMpc" not in f:
                return False
            if f["position_cMpc"].ndim != 2 or f["position_cMpc"].shape[1] != 3:
                return False
            if f["position_cMpc"].shape[0] != int(N_DM_RANDOM):
                return False
            if int(f.attrs.get("random_seed", -1)) != int(DM_RANDOM_SEED):
                return False
            if str(f.attrs.get("source_position_dataset", "")) != str(DM_POSITION_DATASET):
                return False
            if str(f.attrs.get("snapshot", "")) != str(FLAMINGO_DM_SNAPSHOT):
                return False
            if str(f.attrs.get("simulation", "")) != str(FLAMINGO_DM_SIMULATION):
                return False
            if str(f.attrs.get("run", "")) != str(FLAMINGO_DM_RUN):
                return False
            return True
    except Exception:
        return False


def ensure_flamingo_dm_particle_cache():
    """Create a compact cache of randomly selected FLAMINGO DM-particle positions."""
    if dm_particle_file_is_usable(FLAMINGO_DM_COMPACT_FILE) and not OVERWRITE_FLAMINGO_DM_COMPACT_FILE:
        print(f"Using existing compact FLAMINGO DM-particle file: {FLAMINGO_DM_COMPACT_FILE}")
        return

    if hdfstream is None:
        raise ImportError(
            "Missing dependency: hdfstream. Install it with\n"
            "    python -m pip install --user hdfstream h5py numpy matplotlib tqdm astropy pandas requests\n"
            "or uncomment the pip-install cell if running this notebook on a fresh environment."
        )

    if N_DM_RANDOM <= 0:
        raise ValueError("N_DM_RANDOM must be positive")
    if DM_STREAM_BLOCK_SIZE <= 0:
        raise ValueError("DM_STREAM_BLOCK_SIZE must be positive")

    print("Compact FLAMINGO DM-particle file not found, not usable, or overwrite requested.")
    root, resolved_hdfstream_server = open_hdfstream_root()

    print(f"Opening remote snapshot file: {FLAMINGO_DM_REMOTE_PATH}")
    remote_file = root[FLAMINGO_DM_REMOTE_PATH]

    pos_ds, pos_name = open_dataset(remote_file, DM_POSITION_DATASET, kind="DM-particle position")
    scale_factor_dm, redshift_dm, a_source_dm, z_source_dm = read_snapshot_metadata(remote_file)

    if len(pos_ds.shape) != 2 or pos_ds.shape[1] != 3:
        raise ValueError(f"Expected DM position dataset with shape (N, 3), got {pos_ds.shape}")

    n_rows_available = int(pos_ds.shape[0])
    n_select = int(N_DM_RANDOM)
    if n_select > n_rows_available:
        raise ValueError(
            f"Requested {n_select:,} DM particles, but only {n_rows_available:,} are available."
        )

    print(f"DM position dataset : {pos_name}, shape={pos_ds.shape}")
    print(f"Rows available      : {n_rows_available:,}")
    print(f"Rows selected       : {n_select:,}")
    print(f"Random seed         : {DM_RANDOM_SEED}")
    print(f"Snapshot scale factor = {scale_factor_dm:.8f} ({a_source_dm})")
    print(f"Snapshot redshift     = {redshift_dm:.8f} ({z_source_dm})")

    rng = np.random.default_rng(DM_RANDOM_SEED)
    selected_indices = np.sort(
        rng.choice(n_rows_available, size=n_select, replace=False).astype(np.int64, copy=False)
    )

    tmp_file = FLAMINGO_DM_COMPACT_FILE.with_suffix(FLAMINGO_DM_COMPACT_FILE.suffix + ".tmp")
    if tmp_file.exists():
        tmp_file.unlink()

    n_blocks = math.ceil(n_select / DM_STREAM_BLOCK_SIZE)
    print(f"Writing compact DM-particle cache to temporary file: {tmp_file}")
    print(f"Streaming {n_select:,} sorted random rows in {n_blocks:,} blocks")

    start_time = time.time()

    chunk_rows = min(DM_STREAM_BLOCK_SIZE, 250_000)
    vector_kwargs = dict(chunks=(chunk_rows, 3))
    index_kwargs = dict(chunks=(chunk_rows,))
    if COMPRESSION is not None:
        vector_kwargs.update(compression=COMPRESSION, compression_opts=COMPRESSION_OPTS)
        index_kwargs.update(compression=COMPRESSION, compression_opts=COMPRESSION_OPTS)

    with h5py.File(tmp_file, "w") as out:
        out.attrs["source_server"] = resolved_hdfstream_server
        out.attrs["source_remote_path"] = FLAMINGO_DM_REMOTE_PATH
        out.attrs["source_position_dataset"] = pos_name
        out.attrs["snapshot"] = FLAMINGO_DM_SNAPSHOT
        out.attrs["simulation"] = FLAMINGO_DM_SIMULATION
        out.attrs["run"] = FLAMINGO_DM_RUN
        out.attrs["matched_galaxy_snapshot"] = FLAMINGO_SNAPSHOT
        out.attrs["particle_type"] = "PartType1 dark matter"
        out.attrs["random_seed"] = int(DM_RANDOM_SEED)
        out.attrs["random_selection"] = "np.random.default_rng(seed).choice without replacement"
        out.attrs["indices_sorted_before_streaming"] = True
        out.attrs["n_rows_available_remote"] = n_rows_available
        out.attrs["n_rows_selected"] = n_select
        out.attrs["h"] = FLAMINGO_H
        out.attrs["Omega_m"] = FLAMINGO_OMEGA_M
        out.attrs["Omega_b"] = FLAMINGO_OMEGA_B
        out.attrs["Omega_Lambda"] = FLAMINGO_OMEGA_LAMBDA
        out.attrs["sigma8"] = FLAMINGO_SIGMA8
        out.attrs["n_s"] = FLAMINGO_NS
        out.attrs["sum_mnu_eV"] = FLAMINGO_SUM_MNU_EV
        out.attrs["scale_factor"] = scale_factor_dm
        out.attrs["redshift"] = redshift_dm
        out.attrs["scale_factor_source"] = a_source_dm
        out.attrs["redshift_source"] = z_source_dm
        out.attrs["box_size_cMpc"] = BOX_SIZE_CMPC
        out.attrs["box_size_hinv_cMpc"] = BOX_SIZE_CMPC_OVER_H

        pos_out = out.create_dataset("position_cMpc", shape=(n_select, 3), dtype="f4", **vector_kwargs)
        idx_out = out.create_dataset("source_particle_index", data=selected_indices, dtype="i8", **index_kwargs)
        copy_attrs_best_effort(pos_ds, pos_out)
        pos_out.attrs["description"] = "Comoving Mpc positions for a sorted-index random subset of DM particles"
        idx_out.attrs["description"] = "Sorted source-row indices selected before hdfstream access"

        with tqdm(total=n_select, desc="Streaming DM particles", unit="particle") as pbar:
            for iblock, start in enumerate(range(0, n_select, DM_STREAM_BLOCK_SIZE), start=1):
                stop = min(start + DM_STREAM_BLOCK_SIZE, n_select)
                idx_block = selected_indices[start:stop]

                positions_block = np.asarray(pos_ds[idx_block, :], dtype=np.float32)

                if not np.all(np.isfinite(positions_block)):
                    raise ValueError(
                        f"Non-finite DM-particle coordinates encountered in block {iblock}/{n_blocks}"
                    )

                pos_out[start:stop, :] = positions_block
                pbar.update(stop - start)
                elapsed = max(time.time() - start_time, 1e-6)
                pbar.set_postfix(
                    block=f"{iblock}/{n_blocks}",
                    rate=f"{stop / elapsed:,.0f} particles/s",
                )

    tmp_file.replace(FLAMINGO_DM_COMPACT_FILE)
    elapsed = time.time() - start_time
    print(f"Done. Wrote {n_select:,} DM-particle positions to {FLAMINGO_DM_COMPACT_FILE}")
    print(f"Elapsed time: {elapsed/60:.2f} minutes")


# ## Load S2
# 
# The S2 point set is kept in its fiducial published coordinates.
# 

# In[11]:


# ---------------------------------------------------------------------------
# Load the published S2 projected point set
# ---------------------------------------------------------------------------
ensure_local_file(DESI_S2_URL, DESI_S2_FILE)

s2 = np.loadtxt(DESI_S2_FILE)
if s2.ndim != 2 or s2.shape[1] < 2:
    raise ValueError(f"Expected at least two columns in {DESI_S2_FILE}, got shape {s2.shape}")

s2_x = np.asarray(s2[:, 0], dtype=float)
s2_y = np.asarray(s2[:, 1], dtype=float)
s2_r = np.sqrt(s2_x**2 + s2_y**2)
mask_s2_fiducial = np.isfinite(s2_x) & np.isfinite(s2_y) & (s2_r <= S2_FIDUCIAL_RADIUS_HMPC)

print(f"S2 points: {len(s2_x):,}")
print(f"S2 fiducial points inside R={S2_FIDUCIAL_RADIUS_HMPC:g}: {np.count_nonzero(mask_s2_fiducial):,}")
print(f"S2 x range: [{s2_x.min():.3f}, {s2_x.max():.3f}] h^-1 Mpc")
print(f"S2 y range: [{s2_y.min():.3f}, {s2_y.max():.3f}] h^-1 Mpc")
print(f"S2 max projected radius: {s2_r.max():.3f} h^-1 Mpc")


# ## Load DESI and construct observer-centred coordinates
# 
# DESI redshifts are converted using the FLAMINGO D3A cosmology.  The local DESI basis and translation follow the current notebook, then the coordinates are expressed relative to the observer, with \(x<0\) pointing away from the observer.
# 

# In[12]:


# ---------------------------------------------------------------------------
# Load DESI DR1 BGS and construct observer-centred local coordinates
# ---------------------------------------------------------------------------
desi_paths = [
    ensure_local_file(f"{DESI_BASE_URL}/{filename}", DATA_DIR / filename)
    for filename in DESI_FILES
]

print("Reading DESI catalogues...")
tables = [Table.read(path, hdu=1) for path in desi_paths]
cat_bgs = vstack(tables, metadata_conflicts="silent")
print(f"Loaded cat_bgs: N = {len(cat_bgs):,}")

ra_all = get_col(cat_bgs, "RA").astype(float)
dec_all = get_col(cat_bgs, "DEC").astype(float)
z_all = get_col(cat_bgs, "Z").astype(float)
flux_g = get_col(cat_bgs, "flux_g_dered").astype(float)
flux_r = get_col(cat_bgs, "flux_r_dered").astype(float)

valid_r = (
    np.isfinite(ra_all)
    & np.isfinite(dec_all)
    & np.isfinite(z_all)
    & (z_all > 0.0)
    & np.isfinite(flux_r)
    & (flux_r > 0.0)
)
valid_g = np.isfinite(flux_g) & (flux_g > 0.0)
valid = valid_r & valid_g

#-----------------------------------------------------------------------------------------------
# ------------------------------------------------------------------
# Anisotropic cosmology parameters
# ------------------------------------------------------------------

LA_AXIS = 264.0
BA_AXIS = 48.0

H0_MODEL = 0.7
SIGMA0 = 0.001
E20 = 0.001
OM_IM0 = 0.31
OM_K0 = 1e-09
WBAR = 0.0
#-----------------------------------------------------------------------------------------------

D_hmpc_all = np.full(len(cat_bgs), np.nan)
#D_hmpc_all[valid] = FLAMINGO_H * COSMO.comoving_distance(z_all[valid]).to_value(u.Mpc)

#-----------------------------------------------------------------------------------------------
# Convert DESI coordinates to Galactic coordinates
gal = SkyCoord(
    ra=ra_all[valid] * u.deg,
    dec=dec_all[valid] * u.deg,
    frame="icrs",
).galactic

alpha = ang_sep(
    gal.l.deg,
    gal.b.deg,
    LA_AXIS,
    BA_AXIS,
)

print("alpha:", np.min(alpha), np.max(alpha), np.mean(alpha))

D_hmpc_all[valid] = anisotropic_distance_array(
    z_all[valid],
    alpha,
    H0_MODEL,
    SIGMA0,
    E20,
    OM_IM0,
    OM_K0,
    WBAR,
)
#-----------------------------------------------------------------------------------------------

m_g = np.full(len(cat_bgs), np.nan)
m_r = np.full(len(cat_bgs), np.nan)
m_g[valid_g] = 22.5 - 2.5 * np.log10(flux_g[valid_g])
m_r[valid_r] = 22.5 - 2.5 * np.log10(flux_r[valid_r])

g_minus_r = m_g - m_r

# dL_Mpc = np.full(len(cat_bgs), np.nan)
# dL_Mpc[valid] = COSMO.luminosity_distance(z_all[valid]).to_value(u.Mpc)

# DM = np.full(len(cat_bgs), np.nan)
# DM[valid] = 5.0 * np.log10(dL_Mpc[valid]) + 25.0

dL_Mpc = np.full(len(cat_bgs), np.nan)

dL_Mpc[valid] = (
    (1.0 + z_all[valid])
    * D_hmpc_all[valid]
    / FLAMINGO_H
)

DM = np.full(len(cat_bgs), np.nan)

DM[valid] = 5.0*np.log10(dL_Mpc[valid]) + 25.0

K_r_correction = np.full(len(cat_bgs), np.nan, dtype=float)
K_r_correction[valid] = z_all[valid] * (2.5 - 1.5 * g_minus_r[valid])

Mr_uncorrected = m_r - DM
Mr_proxy = Mr_uncorrected - K_r_correction

mask_r25 = (
    valid
    & (ra_all >= RA_MIN_R25)
    & (ra_all <= RA_MAX_R25)
    & (dec_all >= DEC_MIN_R25)
    & (dec_all <= DEC_MAX_R25)
)

mask_support = mask_r25 & (D_hmpc_all <= D_MAX_HMPC)
mask_desi = mask_support & np.isfinite(Mr_proxy) & (Mr_proxy < MR_PROXY_CUT_WORKING)

# Reference sample used to define the local origin.
valid_ref = mask_support & np.isfinite(Mr_proxy)
idx_ref_all = np.flatnonzero(valid_ref)
idx_ref = idx_ref_all[np.argsort(Mr_proxy[idx_ref_all])[:N_PUBLISHED_VL3_R25]]

mask_origin_reference = np.zeros(len(cat_bgs), dtype=bool)
mask_origin_reference[idx_ref] = True

# Local orthonormal basis at the centre of the R25 angular footprint.
RA0_DEG = 0.5 * (RA_MIN_R25 + RA_MAX_R25)
DEC0_DEG = 0.5 * (DEC_MIN_R25 + DEC_MAX_R25)
ra0 = np.deg2rad(RA0_DEG)
dec0 = np.deg2rad(DEC0_DEG)

e_los = unit_vector(np.array([ra0]), np.array([dec0]))[0]
e_los /= np.linalg.norm(e_los)

e_ra = np.array([-np.sin(ra0), np.cos(ra0), 0.0], dtype=float)
e_ra /= np.linalg.norm(e_ra)

e_dec = np.array([
    -np.sin(dec0) * np.cos(ra0),
    -np.sin(dec0) * np.sin(ra0),
     np.cos(dec0),
], dtype=float)
e_dec /= np.linalg.norm(e_dec)

# Origin from the brightest N_PUBLISHED_VL3_R25 objects in the R25 support region.
ra_ref = np.deg2rad(ra_all[mask_origin_reference])
dec_ref = np.deg2rad(dec_all[mask_origin_reference])
D_ref = D_hmpc_all[mask_origin_reference]

rvec_ref = D_ref[:, None] * unit_vector(ra_ref, dec_ref)
los_ref = rvec_ref @ e_los
tra_ref = rvec_ref @ e_ra
tde_ref = rvec_ref @ e_dec

los0 = np.median(los_ref)
tra0 = np.median(tra_ref)
tde0 = np.median(tde_ref)

# Coordinates for the working DESI sample.
ra = np.deg2rad(ra_all[mask_desi])
dec = np.deg2rad(dec_all[mask_desi])
D = D_hmpc_all[mask_desi]
Mr_working = Mr_proxy[mask_desi]
z_working = z_all[mask_desi]

rvec = D[:, None] * unit_vector(ra, dec)
los = rvec @ e_los
tra = rvec @ e_ra
tde = rvec @ e_dec

# Axes-aligned orientation inherited from the current notebook.
u_exact = -(los - los0)
v_exact = -(tra - tra0)
w_exact = +(tde - tde0)

U_CENTER_EXACT = X_CENTER_BASE
V_CENTER_EXACT = -Y_CENTER_BASE
W_CENTER_EFFECTIVE = Z_CENTER
U_CENTER_EFFECTIVE = U_CENTER_EXACT - DX_PLOT / SCALE_TOTAL
V_CENTER_EFFECTIVE = V_CENTER_EXACT - DY_PLOT / SCALE_TOTAL

x_true_all = u_exact - U_CENTER_EFFECTIVE
y_true_all = v_exact - V_CENTER_EFFECTIVE
z_true_all = w_exact - W_CENTER_EFFECTIVE

# Observer-centred plotting/selection coordinates.
# x<0 means increasing distance away from the observer.
D_x_all = los0 - (x_true_all + U_CENTER_EFFECTIVE)
x_obs_signed_all = -D_x_all
y_obs_all = y_true_all
z_obs_all = z_true_all

print("DESI selection summary")
print("----------------------")
print(f"Full BGS catalogue                              : {len(cat_bgs):,}")
print(f"R25 only                                        : {np.count_nonzero(mask_r25):,}")
print(f"R25 + D <= {D_MAX_HMPC:.0f} h^-1 Mpc            : {np.count_nonzero(mask_support):,}")
print(f"R25 + broad working M_r<{MR_PROXY_CUT_WORKING:.2f} : {np.count_nonzero(mask_desi):,}")
print(f"Origin reference sample                         : {np.count_nonzero(mask_origin_reference):,}")
print()
print("Observer-centred DESI coordinate ranges")
print("---------------------------------------")
for label, arr in [("x", x_obs_signed_all), ("y", y_obs_all), ("z", z_obs_all), ("D_x", D_x_all)]:
    lo, hi = finite_range(arr)
    print(f"{label:>3s}: [{lo:.2f}, {hi:.2f}] h^-1 Mpc")
print()
print("DESI magnitude convention: M_r = m_r - DM - z * (2.5 - 1.5(g-r))")


# ## Select the two DESI comparison cylinders
# 
# The two requested regions are selected in observer-centred coordinates.  The matching FLAMINGO target count for each size is the number of DESI galaxies after the corresponding \(M_r\) cut.
# 

# In[13]:


# ---------------------------------------------------------------------------
# Select the two requested DESI cylinders
# ---------------------------------------------------------------------------
def s2_density_matched_count(radius_hmpc):
    """Return the point count at the S2 fiducial surface density."""
    n_s2 = int(np.count_nonzero(mask_s2_fiducial))
    s2_area = np.pi * float(S2_FIDUCIAL_RADIUS_HMPC)**2
    target_area = np.pi * float(radius_hmpc)**2
    return int(np.rint(n_s2 * target_area / s2_area))


desi_regions = {}

for name, region in COMPARISON_REGIONS.items():
    radius = float(region["radius_hmpc"])
    cx = float(region["center_x_hmpc"])
    cy = float(region["center_y_hmpc"])
    cz = float(region["center_z_hmpc"])
    mr_limit = float(region["desi_mr_limit"])

    r2_xy = (x_obs_signed_all - cx)**2 + (y_obs_all - cy)**2
    in_cyl = (
        np.isfinite(x_obs_signed_all)
        & np.isfinite(y_obs_all)
        & np.isfinite(z_obs_all)
        & np.isfinite(Mr_working)
        & (r2_xy <= radius**2)
        & (np.abs(z_obs_all - cz) <= CYLINDER_HALF_THICKNESS_HMPC)
    )

    if name == "large":
        # Match the projected number density of the S2 fiducial catalogue.
        # Since this aperture has R=290 h^-1 Mpc rather than R=300 h^-1 Mpc,
        # this keeps area/N, and hence the Poisson shot-noise level, matched
        # to S2 rather than matching the raw number of points.
        n_target = s2_density_matched_count(radius)
        idx_all = np.flatnonzero(in_cyl)
        idx_keep, n_available = select_brightest_indices(
            idx_all,
            Mr_working,
            n_keep=n_target,
            context=(
                f"DESI {name} density match to S2, "
                f"R={radius:g} h^-1 Mpc"
            ),
        )

        x_sel = x_obs_signed_all[idx_keep] - cx
        y_sel = y_obs_all[idx_keep] - cy
        z_sel = z_obs_all[idx_keep] - cz
        mr_sel = Mr_working[idx_keep]

        selection_mode = "S2 surface-density matched"
        n_before_cut = int(n_available)
        mr_limit_effective = float(np.nanmax(mr_sel)) if len(mr_sel) else np.nan
        density_reference_count = int(np.count_nonzero(mask_s2_fiducial))
        density_reference_radius = float(S2_FIDUCIAL_RADIUS_HMPC)
        density_reference_area = np.pi * density_reference_radius**2
        density_reference_surface_density = density_reference_count / density_reference_area

        # Use the effective threshold for later labels.
        region["desi_mr_limit"] = mr_limit_effective

    else:
        # The small cylinder is not directly compared to S2, so retain the
        # original fixed magnitude selection.
        in_cyl_bright = in_cyl & (Mr_working < mr_limit)

        x_sel = x_obs_signed_all[in_cyl_bright] - cx
        y_sel = y_obs_all[in_cyl_bright] - cy
        z_sel = z_obs_all[in_cyl_bright] - cz
        mr_sel = Mr_working[in_cyl_bright]

        selection_mode = "fixed magnitude cut"
        n_before_cut = int(np.count_nonzero(in_cyl))
        mr_limit_effective = mr_limit
        density_reference_count = None
        density_reference_radius = None
        density_reference_surface_density = None

    area = np.pi * radius**2
    surface_density = len(x_sel) / area if area > 0.0 else np.nan

    desi_regions[name] = {
        **region,
        "x": x_sel,
        "y": y_sel,
        "z": z_sel,
        "Mr": mr_sel,
        "n_target": len(x_sel),
        "selection_mode": selection_mode,
        "n_available_before_magnitude_cut": n_before_cut,
        "desi_mr_limit_effective": mr_limit_effective,
        "surface_density_h2_mpc2": surface_density,
        "density_reference_count": density_reference_count,
        "density_reference_radius_hmpc": density_reference_radius,
        "density_reference_surface_density_h2_mpc2": density_reference_surface_density,
    }

    print("\n" + "-" * 72)
    print(f"DESI region: {name}")
    print(f"centre in observer coordinates = ({cx:g}, {cy:g}, {cz:g}) h^-1 Mpc")
    print(f"radius = {radius:g} h^-1 Mpc, thickness = {CYLINDER_THICKNESS_HMPC:g} h^-1 Mpc")
    print(f"selection mode = {selection_mode}")
    print(f"N before selection = {n_before_cut:,}")
    print(f"N after selection  = {len(x_sel):,}")
    print(f"effective M_r cut  = {mr_limit_effective:.4g}")
    print(f"surface density    = {surface_density:.6e} (h/Mpc)^2")

    if name == "large":
        print(
            "S2 reference density = "
            f"{density_reference_surface_density:.6e} (h/Mpc)^2 "
            f"from N={density_reference_count:,} inside "
            f"R={density_reference_radius:g} h^-1 Mpc"
        )

    print(f"Matched FLAMINGO target N per slice = {len(x_sel):,}")


# ## Show the 290 / h Mpc circle 

# In[14]:


# ---------------------------------------------------------------------------
# DESI bright-galaxy footprint with the large comparison aperture
# ---------------------------------------------------------------------------
# This uses the observer-centred coordinates constructed above.  In this
# convention, x<0 points away from the observer, so a centre 390 h^-1 Mpc from
# the observer is plotted at x=-390 h^-1 Mpc.
bright_region = COMPARISON_REGIONS["large"]

DESI_BRIGHT_MR_LIMIT = -21.5
DESI_BRIGHT_CIRCLE_RADIUS_HMPC = float(bright_region["radius_hmpc"])
DESI_BRIGHT_CIRCLE_CENTER_HMPC = (
    float(bright_region["center_x_hmpc"]),
    float(bright_region["center_y_hmpc"]),
)

# Use the same z-slab as the comparison cylinders.  Set this to False if you
# want the full projected DESI footprint without the |z| cut.
DESI_BRIGHT_USE_Z_SLICE = True
DESI_BRIGHT_Z_HALF_THICKNESS_HMPC = CYLINDER_HALF_THICKNESS_HMPC

mask_desi_bright = (
    np.isfinite(x_obs_signed_all)
    & np.isfinite(y_obs_all)
    & np.isfinite(z_obs_all)
    & np.isfinite(D_x_all)
    & np.isfinite(Mr_working)
    & (Mr_working < DESI_BRIGHT_MR_LIMIT)
    & (D_x_all >= 0.0)
    & (D_x_all <= D_MAX_HMPC)
)

if DESI_BRIGHT_USE_Z_SLICE:
    mask_desi_bright &= (
        np.abs(z_obs_all - float(bright_region["center_z_hmpc"]))
        <= DESI_BRIGHT_Z_HALF_THICKNESS_HMPC
    )

fig, ax = plt.subplots(figsize=(5.6, 5.0), constrained_layout=True)

ax.scatter(
    x_obs_signed_all[mask_desi_bright],
    y_obs_all[mask_desi_bright],
    s=0.25,
    marker=".",
    linewidths=0,
    alpha=1.0,
    color="black",
    rasterized=True,
    zorder=2,
)

draw_circle(
    ax,
    DESI_BRIGHT_CIRCLE_CENTER_HMPC,
    DESI_BRIGHT_CIRCLE_RADIUS_HMPC,
    color="black",
    linestyle="--",
    lw=1.0,
    alpha=1.0,
    zorder=4,
)

ax.set_xlim(-D_MAX_HMPC, 0.0)
ax.set_ylim(-310.0, 310.0)
ax.set_aspect("equal", adjustable="box")
ax.tick_params(direction="in", top=True, right=True, which="both")
ax.xaxis.set_major_locator(MultipleLocator(100.0))
ax.yaxis.set_major_locator(MultipleLocator(100.0))
ax.set_xlabel(r"$x_{\rm obs}\ [h^{-1}\,\mathrm{Mpc}]$")
ax.set_ylabel(r"$y\ [h^{-1}\,\mathrm{Mpc}]$")

z_text = (
    rf", $|z-z_0|<{DESI_BRIGHT_Z_HALF_THICKNESS_HMPC:g}\,h^{{-1}}\,\mathrm{{Mpc}}$"
    if DESI_BRIGHT_USE_Z_SLICE
    else ""
)
ax.set_title(
    rf"DESI BGS, $M_r<{DESI_BRIGHT_MR_LIMIT:g}${z_text}"
    + "\n"
    + rf"circle: $R={DESI_BRIGHT_CIRCLE_RADIUS_HMPC:g}\,h^{{-1}}\,\mathrm{{Mpc}}$, "
      rf"centre $x={DESI_BRIGHT_CIRCLE_CENTER_HMPC[0]:g}\,h^{{-1}}\,\mathrm{{Mpc}}$"
)

outfile = FIGURES_DIR / "DESI_bright_Mr_lt_minus21p5_large_circle.pdf"
save_figure(fig, outfile)

print("DESI bright-galaxy aperture plot")
print("--------------------------------")
print(f"M_r limit                   : {DESI_BRIGHT_MR_LIMIT:g}")
print(f"N plotted                   : {np.count_nonzero(mask_desi_bright):,}")
print(f"Circle radius               : {DESI_BRIGHT_CIRCLE_RADIUS_HMPC:g} h^-1 Mpc")
print(
    "Circle centre              : "
    f"({DESI_BRIGHT_CIRCLE_CENTER_HMPC[0]:g}, {DESI_BRIGHT_CIRCLE_CENTER_HMPC[1]:g}) h^-1 Mpc"
)
print(f"Circle centre distance      : {abs(DESI_BRIGHT_CIRCLE_CENTER_HMPC[0]):g} h^-1 Mpc")
print(f"Using comparison z-slab      : {DESI_BRIGHT_USE_Z_SLICE}")


# ## Download and project SDSS
# 
# The SDSS query is cached as CSV.  The selected SDSS galaxies are projected into the same observer-centred coordinate system as DESI.
# 

# In[15]:


# ---------------------------------------------------------------------------
# Download and project SDSS spectroscopic galaxies in the Sloan Great Wall region
# ---------------------------------------------------------------------------
SDSS_DR = 18
SDSS_SGW_CSV = DATA_DIR / "sdss_dr18_sloan_great_wall_region.csv"
FORCE_SDSS_REDOWNLOAD = False

SGW_RA_MIN_DEG = 150.0
SGW_RA_MAX_DEG = 220.0
SGW_DEC_MIN_DEG = -4.0
SGW_DEC_MAX_DEG = 8.0
SGW_Z_MIN = 0.045
SGW_Z_MAX = 0.120

SDSS_SQL_SGW = f"""
SELECT TOP 500000
    s.specObjID,
    s.bestObjID,
    s.ra,
    s.dec,
    s.z,
    s.zErr,
    s.zWarning,
    p.dered_g,
    p.dered_r,
    p.petroMag_r
FROM SpecObj AS s
JOIN PhotoObj AS p ON p.objID = s.bestObjID
WHERE
    s.class = 'GALAXY'
    AND s.sciencePrimary = 1
    AND s.zWarning = 0
    AND s.ra BETWEEN {SGW_RA_MIN_DEG} AND {SGW_RA_MAX_DEG}
    AND s.dec BETWEEN {SGW_DEC_MIN_DEG} AND {SGW_DEC_MAX_DEG}
    AND s.z BETWEEN {SGW_Z_MIN} AND {SGW_Z_MAX}
    AND p.dered_r > 0
    AND p.dered_r < 19.5
"""


def read_sdss_csv_robust(path_or_text):
    """Read SDSS SkyServer CSV output robustly."""
    is_existing_path = (
        isinstance(path_or_text, Path)
        or (
            isinstance(path_or_text, str)
            and "\n" not in path_or_text
            and len(path_or_text) < 512
            and Path(path_or_text).exists()
        )
    )

    if is_existing_path:
        df = pd.read_csv(path_or_text, comment="#")
    else:
        df = pd.read_csv(io.StringIO(str(path_or_text)), comment="#")

    df.columns = [str(c).strip() for c in df.columns]
    required = ["specObjID", "bestObjID", "ra", "dec", "z"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            "SDSS table is missing required columns "
            f"{missing}. Available columns are:\n{df.columns.tolist()}"
        )
    return df


def query_sdss_skyserver_csv(sql, *, dr=18, timeout=180):
    """Query SDSS SkyServer SQL search and return a clean pandas DataFrame."""
    endpoints = [
        f"https://skyserver.sdss.org/dr{dr}/SkyServerWS/SearchTools/SqlSearch",
        f"https://skyserver.sdss.org/dr{dr}/en/tools/search/x_results.aspx",
    ]

    last_error = None
    for url in endpoints:
        try:
            response = requests.get(url, params={"cmd": sql, "format": "csv"}, timeout=timeout)
            response.raise_for_status()
            text = response.text.strip()
            if text.lower().startswith("<html") or "error" in text[:500].lower():
                raise RuntimeError(text[:1000])
            return read_sdss_csv_robust(text)
        except Exception as err:
            last_error = err
            print(f"SDSS endpoint failed: {url}")
            print(f"  {err}")

    raise RuntimeError(f"All SDSS SkyServer endpoints failed. Last error:\n{last_error}")


need_download = FORCE_SDSS_REDOWNLOAD or not SDSS_SGW_CSV.exists()

if SDSS_SGW_CSV.exists() and not FORCE_SDSS_REDOWNLOAD:
    try:
        sdss_sgw = read_sdss_csv_robust(SDSS_SGW_CSV)
        print(f"Loaded existing SDSS SGW catalogue: {SDSS_SGW_CSV}")
    except Exception as err:
        print("Existing SDSS SGW catalogue is not usable.")
        print(err)
        SDSS_SGW_CSV.unlink(missing_ok=True)
        need_download = True

if need_download:
    sdss_sgw = query_sdss_skyserver_csv(SDSS_SQL_SGW, dr=SDSS_DR)
    sdss_sgw.to_csv(SDSS_SGW_CSV, index=False)
    print(f"Downloaded and saved clean catalogue: {SDSS_SGW_CSV}")

sdss_sgw.columns = [str(c).strip() for c in sdss_sgw.columns]
required = ["ra", "dec", "z", "dered_g", "dered_r"]
missing = [c for c in required if c not in sdss_sgw.columns]
if missing:
    raise KeyError(f"Missing SDSS columns {missing}. Available columns are:\n{sdss_sgw.columns.tolist()}")

ra_sdss = pd.to_numeric(sdss_sgw["ra"], errors="coerce").to_numpy(dtype=float)
dec_sdss = pd.to_numeric(sdss_sgw["dec"], errors="coerce").to_numpy(dtype=float)
z_sdss = pd.to_numeric(sdss_sgw["z"], errors="coerce").to_numpy(dtype=float)
dered_g_sdss = pd.to_numeric(sdss_sgw["dered_g"], errors="coerce").to_numpy(dtype=float)
dered_r_sdss = pd.to_numeric(sdss_sgw["dered_r"], errors="coerce").to_numpy(dtype=float)

good_sdss = (
    np.isfinite(ra_sdss)
    & np.isfinite(dec_sdss)
    & np.isfinite(z_sdss)
    & np.isfinite(dered_g_sdss)
    & np.isfinite(dered_r_sdss)
    & (z_sdss > 0.0)
)

sdss_sgw = sdss_sgw.loc[good_sdss].reset_index(drop=True)
ra_sdss = ra_sdss[good_sdss]
dec_sdss = dec_sdss[good_sdss]
z_sdss = z_sdss[good_sdss]
dered_g_sdss = dered_g_sdss[good_sdss]
dered_r_sdss = dered_r_sdss[good_sdss]

D_sdss_hmpc = FLAMINGO_H * COSMO.comoving_distance(z_sdss).to_value(u.Mpc)

ra_sdss_rad = np.radians(ra_sdss)
dec_sdss_rad = np.radians(dec_sdss)
rvec_sdss = D_sdss_hmpc[:, None] * unit_vector(ra_sdss_rad, dec_sdss_rad)

los_sdss = rvec_sdss @ e_los
tra_sdss = rvec_sdss @ e_ra
tde_sdss = rvec_sdss @ e_dec

u_sdss = -(los_sdss - los0)
v_sdss = -(tra_sdss - tra0)
w_sdss = +(tde_sdss - tde0)

x_true_sdss = u_sdss - U_CENTER_EFFECTIVE
y_true_sdss = v_sdss - V_CENTER_EFFECTIVE
z_true_sdss = w_sdss - W_CENTER_EFFECTIVE

D_x_sdss = los0 - (x_true_sdss + U_CENTER_EFFECTIVE)
x_obs_signed_sdss = -D_x_sdss
y_obs_sdss = y_true_sdss
z_obs_sdss = z_true_sdss

g_minus_r_sdss = dered_g_sdss - dered_r_sdss
K_r_sdss = z_sdss * (2.5 - 1.5 * g_minus_r_sdss)
DM_sdss = 5.0 * np.log10(COSMO.luminosity_distance(z_sdss).to_value(u.Mpc)) + 25.0
M_r_sdss = dered_r_sdss - DM_sdss - K_r_sdss

sdss_sgw_projected = sdss_sgw.copy()
sdss_sgw_projected["D_hmpc"] = D_sdss_hmpc
sdss_sgw_projected["x_obs_signed_hmpc"] = x_obs_signed_sdss
sdss_sgw_projected["y_obs_hmpc"] = y_obs_sdss
sdss_sgw_projected["z_obs_hmpc"] = z_obs_sdss
sdss_sgw_projected["M_r_proxy"] = M_r_sdss

sdss_sgw_projected_path = DATA_DIR / "sdss_dr18_sloan_great_wall_observer_coordinates.ecsv"
Table.from_pandas(sdss_sgw_projected).write(sdss_sgw_projected_path, overwrite=True)

mask_sdss_footprint_panel = (
    np.isfinite(x_obs_signed_sdss)
    & np.isfinite(y_obs_sdss)
    & np.isfinite(z_obs_sdss)
    & (D_sdss_hmpc >= 150.0)
    & (D_sdss_hmpc <= 300.0)
    & (np.abs(z_obs_sdss) <= CYLINDER_HALF_THICKNESS_HMPC)
)

print("SDSS Sloan Great Wall projection")
print("--------------------------------")
print(f"N raw/clean SDSS galaxies queried        = {len(sdss_sgw_projected):,}")
print(f"N in 150--300 h^-1 Mpc and same slab     = {np.count_nonzero(mask_sdss_footprint_panel):,}")
print(f"Saved {sdss_sgw_projected_path}")


# ## Load FLAMINGO galaxy and DM-particle caches
# 
# Load the compact SOAP-HBT galaxy catalogue and the random real-space L1_m9 DMO DM-particle subset.

# In[16]:


# ---------------------------------------------------------------------------
# Create/read compact FLAMINGO galaxy and DM-particle caches and load columns
# ---------------------------------------------------------------------------
ensure_flamingo_compact_catalogue()
ensure_flamingo_dm_particle_cache()

print(f"Reading compact FLAMINGO galaxy file: {FLAMINGO_COMPACT_FILE}")
with h5py.File(FLAMINGO_COMPACT_FILE, "r") as f:
    h = FLAMINGO_H  # fixed by request; do not override from cache metadata
    scale_factor = float(f.attrs["scale_factor"])
    snapshot_redshift = float(f.attrs["redshift"])
    positions_cMpc = np.asarray(f["position_cMpc"][()], dtype=np.float64)
    velocities_kms = np.asarray(f["velocity_kms"][()], dtype=np.float64)
    m_sim_ab = np.asarray(f[FLAMINGO_MAG_OUTPUT_DATASET][()], dtype=np.float64)
    flamingo_luminosity = np.asarray(f[FLAMINGO_LUMINOSITY_OUTPUT_DATASET][()], dtype=np.float64)

x_sim_cMpc = positions_cMpc[:, 0]
y_sim_cMpc = positions_cMpc[:, 1]
z_sim_cMpc = positions_cMpc[:, 2]
vx_sim_kms = velocities_kms[:, 0]
vy_sim_kms = velocities_kms[:, 1]
vz_sim_kms = velocities_kms[:, 2]

finite = (
    np.isfinite(x_sim_cMpc)
    & np.isfinite(y_sim_cMpc)
    & np.isfinite(z_sim_cMpc)
    & np.isfinite(vx_sim_kms)
    & np.isfinite(vy_sim_kms)
    & np.isfinite(vz_sim_kms)
    & np.isfinite(m_sim_ab)
)

if not np.all(finite):
    warnings.warn(f"Dropping {np.count_nonzero(~finite):,} invalid FLAMINGO objects")
    x_sim_cMpc = x_sim_cMpc[finite]
    y_sim_cMpc = y_sim_cMpc[finite]
    z_sim_cMpc = z_sim_cMpc[finite]
    vx_sim_kms = vx_sim_kms[finite]
    vy_sim_kms = vy_sim_kms[finite]
    vz_sim_kms = vz_sim_kms[finite]
    m_sim_ab = m_sim_ab[finite]
    flamingo_luminosity = flamingo_luminosity[finite]

print("FLAMINGO snapshot")
print("-----------------")
print(f"h used throughout         = {h:.6f}")
print(f"snapshot scale factor     = {scale_factor:.8f}")
print(f"snapshot redshift         = {snapshot_redshift:.8f}")
print(f"objects available         = {len(x_sim_cMpc):,}")
print(f"box size used for slicing = {BOX_SIZE_CMPC:g} cMpc = {BOX_SIZE_CMPC_OVER_H:g} h^-1 cMpc")
for label, arr in [("x", x_sim_cMpc), ("y", y_sim_cMpc), ("z", z_sim_cMpc)]:
    lo, hi = finite_range(arr)
    print(f"{label}_sim range = [{lo:.3f}, {hi:.3f}] cMpc")
lo, hi = finite_range(m_sim_ab)
print(f"M_{FLAMINGO_LUMINOSITY_BAND}_AB range = [{lo:.3f}, {hi:.3f}]")

print(f"\nReading compact FLAMINGO DM-particle file: {FLAMINGO_DM_COMPACT_FILE}")
with h5py.File(FLAMINGO_DM_COMPACT_FILE, "r") as f:
    dm_positions_cMpc = np.asarray(f["position_cMpc"][()], dtype=np.float64)
    dm_source_particle_index = np.asarray(f["source_particle_index"][()], dtype=np.int64) if "source_particle_index" in f else None

x_dm_cMpc = dm_positions_cMpc[:, 0]
y_dm_cMpc = dm_positions_cMpc[:, 1]
z_dm_cMpc = dm_positions_cMpc[:, 2]

finite_dm = (
    np.isfinite(x_dm_cMpc)
    & np.isfinite(y_dm_cMpc)
    & np.isfinite(z_dm_cMpc)
)
if not np.all(finite_dm):
    warnings.warn(f"Dropping {np.count_nonzero(~finite_dm):,} invalid FLAMINGO DM particles")
    x_dm_cMpc = x_dm_cMpc[finite_dm]
    y_dm_cMpc = y_dm_cMpc[finite_dm]
    z_dm_cMpc = z_dm_cMpc[finite_dm]
    if dm_source_particle_index is not None:
        dm_source_particle_index = dm_source_particle_index[finite_dm]

print("\nFLAMINGO DM-particle subset")
print("---------------------------")
print(f"particles available       = {len(x_dm_cMpc):,}")
print(f"random seed               = {DM_RANDOM_SEED}")
for label, arr in [("x", x_dm_cMpc), ("y", y_dm_cMpc), ("z", z_dm_cMpc)]:
    lo, hi = finite_range(arr)
    print(f"{label}_dm range = [{lo:.3f}, {hi:.3f}] cMpc")


# ## Construct FLAMINGO galaxy and DM slices
# 
# Build non-overlapping galaxy cylinders and matched random real-space DM-particle cylinders.  The large cylinders use the S2 fiducial surface density; the small cylinders keep their native DESI/FLAMINGO selection.
# 

# In[17]:


# ---------------------------------------------------------------------------
# FLAMINGO cylindrical-slice construction
# ---------------------------------------------------------------------------
def H_z_kms_per_Mpc(z):
    """Hubble rate at redshift z in km/s/Mpc for the FLAMINGO D3A cosmology."""
    return COSMO.H(z).to_value(u.km / u.s / u.Mpc)


def projected_rsd_coordinates_hinv(
    dx_cMpc,
    dy_cMpc,
    vx_kms,
    vy_kms,
    *,
    observer_x_hmpc,
    observer_y_hmpc=0.0,
    h_value=FLAMINGO_H,
    scale_factor=None,
    redshift=None,
):
    """
    Convert real projected coordinates to redshift-space projected coordinates.

    Inputs dx_cMpc, dy_cMpc are relative to the cylinder centre in comoving Mpc.
    Output coordinates are in h^-1 cMpc.  The redshift-space displacement is
    v_los / [a H(a)] in comoving Mpc.
    """
    if scale_factor is None:
        scale_factor = globals().get("scale_factor", 1.0)
    if redshift is None:
        redshift = globals().get("snapshot_redshift", 0.0)

    dx_cMpc = np.asarray(dx_cMpc, dtype=float)
    dy_cMpc = np.asarray(dy_cMpc, dtype=float)
    vx_kms = np.asarray(vx_kms, dtype=float)
    vy_kms = np.asarray(vy_kms, dtype=float)

    observer_x_cMpc = float(observer_x_hmpc) / h_value
    observer_y_cMpc = float(observer_y_hmpc) / h_value

    los_x = dx_cMpc - observer_x_cMpc
    los_y = dy_cMpc - observer_y_cMpc
    los_norm = np.sqrt(los_x**2 + los_y**2)
    safe = los_norm > 0.0

    fallback_norm = np.hypot(observer_x_cMpc, observer_y_cMpc)
    if fallback_norm > 0.0:
        fallback_n_x = -observer_x_cMpc / fallback_norm
        fallback_n_y = -observer_y_cMpc / fallback_norm
    else:
        fallback_n_x = -1.0
        fallback_n_y = 0.0

    n_x = np.full_like(los_x, fallback_n_x, dtype=float)
    n_y = np.full_like(los_y, fallback_n_y, dtype=float)
    n_x[safe] = los_x[safe] / los_norm[safe]
    n_y[safe] = los_y[safe] / los_norm[safe]

    v_los = vx_kms * n_x + vy_kms * n_y
    displacement_cMpc = v_los / (float(scale_factor) * H_z_kms_per_Mpc(redshift))

    x_rsd_hmpc = (dx_cMpc + displacement_cMpc * n_x) * h_value
    y_rsd_hmpc = (dy_cMpc + displacement_cMpc * n_y) * h_value
    return x_rsd_hmpc, y_rsd_hmpc


def construct_nonoverlapping_flamingo_slices_for_region(
    region_name,
    region,
    *,
    redshift_mode,
):
    """Construct non-overlapping FLAMINGO slabs for one requested comparison region."""
    if redshift_mode not in {"real", "rsd"}:
        raise ValueError(f"Unknown redshift_mode={redshift_mode!r}")

    radius_hmpc = float(region["radius_hmpc"])
    radius_cMpc = hmpc_to_cmpc(radius_hmpc, h)
    thickness_cMpc = hmpc_to_cmpc(CYLINDER_THICKNESS_HMPC, h)

    if 2.0 * radius_cMpc > BOX_SIZE_CMPC:
        raise ValueError(
            f"Region {region_name} with R={radius_hmpc:g} h^-1 Mpc "
            f"has diameter {2.0 * radius_cMpc:.2f} cMpc and does not fit in a "
            f"{BOX_SIZE_CMPC:g} cMpc box."
        )

    x_centre_cMpc = 0.5 * BOX_SIZE_CMPC
    y_centre_cMpc = 0.5 * BOX_SIZE_CMPC
    dx_cMpc = x_sim_cMpc - x_centre_cMpc
    dy_cMpc = y_sim_cMpc - y_centre_cMpc

    if redshift_mode == "real":
        x_proj_hmpc = dx_cMpc * h
        y_proj_hmpc = dy_cMpc * h
    else:
        observer_distance_from_centre_hmpc = abs(float(region["center_x_hmpc"]))
        x_proj_hmpc, y_proj_hmpc = projected_rsd_coordinates_hinv(
            dx_cMpc,
            dy_cMpc,
            vx_sim_kms,
            vy_sim_kms,
            observer_x_hmpc=observer_distance_from_centre_hmpc,
            observer_y_hmpc=0.0,
            h_value=h,
            scale_factor=scale_factor,
            redshift=snapshot_redshift,
        )

    n_slices_fit = int(np.floor(BOX_SIZE_CMPC / thickness_cMpc))
    if n_slices_fit < 1:
        raise ValueError("No FLAMINGO slices fit in the simulation box.")

    n_target = int(desi_regions[region_name]["n_target"])
    r_proj_hmpc = np.sqrt(x_proj_hmpc**2 + y_proj_hmpc**2)

    slices = []
    for islice in range(n_slices_fit):
        z_lo = islice * thickness_cMpc
        z_hi = z_lo + thickness_cMpc

        if islice == n_slices_fit - 1:
            in_z = (z_sim_cMpc >= z_lo) & (z_sim_cMpc <= z_hi)
        else:
            in_z = (z_sim_cMpc >= z_lo) & (z_sim_cMpc < z_hi)

        idx_all = np.flatnonzero(in_z & (r_proj_hmpc <= radius_hmpc * (1.0 + 1e-12)))

        idx_keep, n_available = select_brightest_indices(
            idx_all,
            m_sim_ab,
            n_keep=n_target,
            context=(
                f"FLAMINGO {region_name} {redshift_mode} slice {islice}, "
                f"R={radius_hmpc:g} h^-1 Mpc"
            ),
        )

        slices.append({
            "region": region_name,
            "islice": islice,
            "z_lo_cMpc": z_lo,
            "z_hi_cMpc": z_hi,
            "radius_hmpc": radius_hmpc,
            "thickness_hmpc": CYLINDER_THICKNESS_HMPC,
            "redshift_mode": redshift_mode,
            "x": x_proj_hmpc[idx_keep],
            "y": y_proj_hmpc[idx_keep],
            "m_sim_ab": m_sim_ab[idx_keep],
            "n": int(len(idx_keep)),
            "n_available_before_brightness_cut": int(n_available),
            "n_requested_brightest": int(n_target),
        })

    return slices


def select_random_indices_without_replacement(indices, *, n_keep, rng, context=""):
    """Return a sorted random subset of indices without replacement."""
    indices = np.asarray(indices, dtype=int)
    n_available = len(indices)
    n_keep = int(n_keep)

    if n_available < n_keep:
        warnings.warn(
            f"{context}: only {n_available:,} particles available; requested {n_keep:,}. "
            "Keeping all available particles.",
            RuntimeWarning,
        )
        return indices, n_available

    chosen = rng.choice(indices, size=n_keep, replace=False)
    return np.sort(chosen), n_available


def construct_nonoverlapping_flamingo_dm_slices_for_region(
    region_name,
    region,
    *,
    rng,
    target_counts_by_slice=None,
):
    """Construct matched real-space DM-particle slabs for one requested comparison region."""
    radius_hmpc = float(region["radius_hmpc"])
    radius_cMpc = hmpc_to_cmpc(radius_hmpc, h)
    thickness_cMpc = hmpc_to_cmpc(CYLINDER_THICKNESS_HMPC, h)

    if 2.0 * radius_cMpc > BOX_SIZE_CMPC:
        raise ValueError(
            f"Region {region_name} with R={radius_hmpc:g} h^-1 Mpc "
            f"has diameter {2.0 * radius_cMpc:.2f} cMpc and does not fit in a "
            f"{BOX_SIZE_CMPC:g} cMpc box."
        )

    x_centre_cMpc = 0.5 * BOX_SIZE_CMPC
    y_centre_cMpc = 0.5 * BOX_SIZE_CMPC
    dx_cMpc = x_dm_cMpc - x_centre_cMpc
    dy_cMpc = y_dm_cMpc - y_centre_cMpc

    x_proj_hmpc = dx_cMpc * h
    y_proj_hmpc = dy_cMpc * h
    r_proj_hmpc = np.sqrt(x_proj_hmpc**2 + y_proj_hmpc**2)

    n_slices_fit = int(np.floor(BOX_SIZE_CMPC / thickness_cMpc))
    if n_slices_fit < 1:
        raise ValueError("No FLAMINGO DM-particle slices fit in the simulation box.")

    if target_counts_by_slice is None:
        target_counts_by_slice = [int(desi_regions[region_name]["n_target"])] * n_slices_fit
    else:
        target_counts_by_slice = [int(n) for n in target_counts_by_slice]
        if len(target_counts_by_slice) != n_slices_fit:
            raise ValueError(
                f"Expected {n_slices_fit} target counts for {region_name}, "
                f"got {len(target_counts_by_slice)}"
            )

    slices = []
    for islice in range(n_slices_fit):
        z_lo = islice * thickness_cMpc
        z_hi = z_lo + thickness_cMpc

        if islice == n_slices_fit - 1:
            in_z = (z_dm_cMpc >= z_lo) & (z_dm_cMpc <= z_hi)
        else:
            in_z = (z_dm_cMpc >= z_lo) & (z_dm_cMpc < z_hi)

        n_target = int(target_counts_by_slice[islice])
        idx_all = np.flatnonzero(in_z & (r_proj_hmpc <= radius_hmpc * (1.0 + 1e-12)))
        idx_keep, n_available = select_random_indices_without_replacement(
            idx_all,
            n_keep=n_target,
            rng=rng,
            context=(
                f"FLAMINGO DM particles {region_name} real-space slice {islice}, "
                f"R={radius_hmpc:g} h^-1 Mpc"
            ),
        )

        slices.append({
            "region": region_name,
            "islice": islice,
            "z_lo_cMpc": z_lo,
            "z_hi_cMpc": z_hi,
            "radius_hmpc": radius_hmpc,
            "thickness_hmpc": CYLINDER_THICKNESS_HMPC,
            "redshift_mode": "dm_real",
            "x": x_proj_hmpc[idx_keep],
            "y": y_proj_hmpc[idx_keep],
            "n": int(len(idx_keep)),
            "n_available_before_random_cut": int(n_available),
            "n_requested_random": int(n_target),
            "random_seed": int(DM_MATCH_SEED),
        })

    return slices


slices_by_region_case = {name: {} for name in COMPARISON_REGIONS}
dm_slices_by_region = {}
dm_match_rng = np.random.default_rng(DM_MATCH_SEED)

for region_name, region in COMPARISON_REGIONS.items():
    for case in CASE_ORDER:
        slices = construct_nonoverlapping_flamingo_slices_for_region(
            region_name,
            region,
            redshift_mode=case,
        )
        slices_by_region_case[region_name][case] = slices

        counts = np.asarray([s["n"] for s in slices], dtype=int)
        available = np.asarray([s["n_available_before_brightness_cut"] for s in slices], dtype=int)

        print("\n" + "-" * 72)
        print(f"FLAMINGO {region_name}, {CASE_LABELS[case]}")
        print(f"R = {region['radius_hmpc']:g} h^-1 Mpc, thickness = {CYLINDER_THICKNESS_HMPC:g} h^-1 cMpc")
        print(f"Target N per slice = {desi_regions[region_name]['n_target']:,}")
        print(f"Constructed {len(slices)} non-overlapping slabs")
        print(
            "Available before top-N: "
            f"min={available.min():,}, median={int(np.median(available)):,}, max={available.max():,}"
        )
        print(
            "Retained N per slice: "
            f"min={counts.min():,}, median={int(np.median(counts)):,}, max={counts.max():,}"
        )

    real_slice_counts = [s["n"] for s in slices_by_region_case[region_name]["real"]]
    dm_slices = construct_nonoverlapping_flamingo_dm_slices_for_region(
        region_name,
        region,
        rng=dm_match_rng,
        target_counts_by_slice=real_slice_counts,
    )
    dm_slices_by_region[region_name] = dm_slices

    dm_counts = np.asarray([s["n"] for s in dm_slices], dtype=int)
    dm_available = np.asarray([s["n_available_before_random_cut"] for s in dm_slices], dtype=int)

    print("\n" + "-" * 72)
    print(f"FLAMINGO {region_name}, DM particles, real-space")
    print(f"R = {region['radius_hmpc']:g} h^-1 Mpc, thickness = {CYLINDER_THICKNESS_HMPC:g} h^-1 cMpc")
    print(f"Target N per slice = matched to FLAMINGO, real-space galaxy counts")
    print(f"Random match seed = {DM_MATCH_SEED}")
    print(f"Constructed {len(dm_slices)} non-overlapping slabs")
    print(
        "Available before random cut: "
        f"min={dm_available.min():,}, median={int(np.median(dm_available)):,}, max={dm_available.max():,}"
    )
    print(
        "Retained N per slice: "
        f"min={dm_counts.min():,}, median={int(np.median(dm_counts)):,}, max={dm_counts.max():,}"
    )


# ## Scatter Plot - large (r = 290 / h Mpc)

# In[18]:


# ---------------------------------------------------------------------------
# Scatter plot: S2 / DESI / FLAMINGO large-cylinder examples
# ---------------------------------------------------------------------------
def plot_s2_desi_flamingo_large_examples():
    """
    Make a 3x6 figure.

    Simulation columns:
      Row 1: five large FLAMINGO slices with RSD.
      Row 2: the same five large FLAMINGO slices in real-space.
      Row 3: the same five large-cylinder DM-particle slices.

    Left column:
      Row 1: DESI.
      Row 2: empty.
      Row 3: S2.
    """
    region_name = "large"
    region = COMPARISON_REGIONS[region_name]

    rsd_slices = sorted(slices_by_region_case[region_name]["rsd"], key=lambda s: s["islice"])
    real_slices = sorted(slices_by_region_case[region_name]["real"], key=lambda s: s["islice"])
    dm_slices = sorted(dm_slices_by_region[region_name], key=lambda s: s["islice"])

    # Use the same five slice examples in all three simulation rows.
    selected_islices = [s["islice"] for s in rsd_slices[:5]]

    rsd_by_islice = {s["islice"]: s for s in rsd_slices}
    real_by_islice = {s["islice"]: s for s in real_slices}
    dm_by_islice = {s["islice"]: s for s in dm_slices}

    rsd_selected = [rsd_by_islice[i] for i in selected_islices]
    real_selected = [real_by_islice[i] for i in selected_islices]
    dm_selected = [dm_by_islice[i] for i in selected_islices]

    flamingo_radius = float(region["radius_hmpc"])
    desi_radius = flamingo_radius
    s2_radius = float(S2_FIDUCIAL_RADIUS_HMPC)
    plot_radius = max(s2_radius, flamingo_radius)

    mask_s2_panel = mask_s2_fiducial & np.isfinite(s2_x) & np.isfinite(s2_y)
    desi = desi_regions[region_name]

    fig, axes = plt.subplots(
        3,
        6,
        figsize=(12, 6.0),
        sharex=True,
        sharey=True,
        constrained_layout=False,
        dpi=120,
    )
    fig.subplots_adjust(
        left=0.02,
        right=0.96,
        bottom=0.04,
        top=0.98,
        wspace=0.04,
        hspace=0.04,
    )

    def draw_panel(ax, x, y, radius, *, circle_color, circle_ls):
        ax.scatter(
            x,
            y,
            s=0.3,
            marker=".",
            linewidths=0,
            alpha=1.0,
            color="black",
            rasterized=True,
            zorder=-2,
        )

        draw_circle(
            ax,
            (0.0, 0.0),
            radius,
            color=circle_color,
            linestyle=circle_ls,
            lw=0.8,
            alpha=0.95,
            zorder=3,
            clip_on=False,
        )

        ax.set_xlim(-plot_radius, plot_radius)
        ax.set_ylim(-plot_radius, plot_radius)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("")
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Left column: DESI in row 1, empty row 2, S2 in row 3.
    ax_desi = axes[0, 0]
    ax_empty = axes[1, 0]
    ax_s2 = axes[2, 0]

    ax_empty.set_axis_off()

    draw_panel(
        ax_desi,
        desi["x"],
        desi["y"],
        desi_radius,
        circle_color="black",
        circle_ls="--",
    )

    draw_panel(
        ax_s2,
        s2_x[mask_s2_panel],
        s2_y[mask_s2_panel],
        s2_radius,
        circle_color="black",
        circle_ls="-.",
    )

    # Row 1: five RSD examples.
    for icol, s in enumerate(rsd_selected, start=1):
        draw_panel(
            axes[0, icol],
            s["x"],
            s["y"],
            flamingo_radius,
            circle_color="hotpink",
            circle_ls="-",
        )

    # Row 2: same five real-space examples.
    for icol, s in enumerate(real_selected, start=1):
        draw_panel(
            axes[1, icol],
            s["x"],
            s["y"],
            flamingo_radius,
            circle_color="cornflowerblue",
            circle_ls="-",
        )

    # Row 3: same five DM-particle examples.
    for icol, s in enumerate(dm_selected, start=1):
        draw_panel(
            axes[2, icol],
            s["x"],
            s["y"],
            flamingo_radius,
            circle_color="0.5",
            circle_ls="-",
        )

    # Labels for DESI and S2.
    ax_desi.text(
        0.5,
        -0.08,
        "DESI DR1\n" + r"$R = 290\ h^{-1}\ \mathrm{Mpc}$",
        transform=ax_desi.transAxes,
        ha="center",
        va="top",
        color="black",
        clip_on=False,
    )

    ax_s2.text(
        0.5,
        1.05,
        r"S2 (fiducial coordinates)",
        transform=ax_s2.transAxes,
        ha="center",
        va="bottom",
        color="black",
        clip_on=False,
    )

    # Vertical row labels on the right.
    right_ax_top = axes[0, 5]
    right_ax_mid = axes[1, 5]
    right_ax_bot = axes[2, 5]

    right_ax_top.text(
        1.06,
        0.5,
        "FLAMINGO + RSD",
        transform=right_ax_top.transAxes,
        rotation=270,
        ha="left",
        va="center",
        color="hotpink",
        clip_on=False,
        fontsize=11.5
    )

    right_ax_mid.text(
        1.06,
        0.5,
        "FLAMINGO, real-space",
        transform=right_ax_mid.transAxes,
        rotation=270,
        ha="left",
        va="center",
        color="cornflowerblue",
        clip_on=False,
        fontsize=11.5
    )

    right_ax_bot.text(
        1.06,
        0.5,
        "FLAMINGO particles",
        transform=right_ax_bot.transAxes,
        rotation=270,
        ha="left",
        va="center",
        color="0.5",
        clip_on=False,
        fontsize=11.5
    )

    outfile = FIGURES_DIR / (
        f"scatter_S2_DESI_FLAMINGO_large_examples_3x6_left_column_rows_1_and_3_"
        f"R{flamingo_radius:.0f}_S2density.pdf"
    )
    save_figure(fig, outfile)

    print("S2 / DESI / FLAMINGO large-example scatter plot")
    print("------------------------------------------------")
    print(f"Selected slice ids                  : {selected_islices}")
    print(f"S2 points plotted                   : {np.count_nonzero(mask_s2_panel):,}")
    print(f"DESI points plotted                 : {len(desi['x']):,}")
    print(f"RSD points per slice                : {rsd_selected[0]['n']:,}")
    print(f"Real-space points per slice         : {real_selected[0]['n']:,}")
    print(f"DM-particle points per slice        : {dm_selected[0]['n']:,}")
    if desi.get("density_reference_surface_density_h2_mpc2") is not None:
        print(
            "S2 reference surface density        : "
            f"{desi['density_reference_surface_density_h2_mpc2']:.6e} (h/Mpc)^2"
        )
        print(
            "DESI large surface density          : "
            f"{desi['surface_density_h2_mpc2']:.6e} (h/Mpc)^2"
        )
    print(f"FLAMINGO large-cylinder radius      : {flamingo_radius:.1f} h^-1 Mpc")
    print(f"S2 fiducial radius                  : {s2_radius:.1f} h^-1 Mpc")


plot_s2_desi_flamingo_large_examples()


# ## Same, but at approximately the true scale of S2.

# In[19]:


# ---------------------------------------------------------------------------
# Scatter plot: DESI / FLAMINGO small-cylinder examples
# ---------------------------------------------------------------------------
def plot_desi_flamingo_small_examples():
    """
    Make a 3x6 figure for the small 175 h^-1 Mpc cylinders.

    Simulation columns:
      Row 1: five small FLAMINGO slices with RSD.
      Row 2: the same five small FLAMINGO slices in real-space.
      Row 3: the same five small-cylinder DM-particle slices.

    Left column:
      Row 1: DESI.
      Row 2: empty.
      Row 3: empty.

    The small cylinders are not density-matched to S2. They use the native
    small-cylinder DESI/FLAMINGO selections constructed above.
    """
    region_name = "small"
    region = COMPARISON_REGIONS[region_name]

    rsd_slices = sorted(slices_by_region_case[region_name]["rsd"], key=lambda s: s["islice"])
    real_slices = sorted(slices_by_region_case[region_name]["real"], key=lambda s: s["islice"])
    dm_slices = sorted(dm_slices_by_region[region_name], key=lambda s: s["islice"])

    # Use the same five slice examples in all three simulation rows.
    selected_islices = [s["islice"] for s in rsd_slices[:5]]

    rsd_by_islice = {s["islice"]: s for s in rsd_slices}
    real_by_islice = {s["islice"]: s for s in real_slices}
    dm_by_islice = {s["islice"]: s for s in dm_slices}

    rsd_selected = [rsd_by_islice[i] for i in selected_islices]
    real_selected = [real_by_islice[i] for i in selected_islices]
    dm_selected = [dm_by_islice[i] for i in selected_islices]

    flamingo_radius = float(region["radius_hmpc"])
    desi_radius = flamingo_radius
    desi = desi_regions[region_name]

    fig, axes = plt.subplots(
        3,
        6,
        figsize=(12.0, 6.0),
        sharex=False,
        sharey=False,
        constrained_layout=False,
        dpi=120,
    )
    fig.subplots_adjust(
        left=0.02,
        right=0.96,
        bottom=0.04,
        top=0.98,
        wspace=0.04,
        hspace=0.04,
    )

    def draw_panel(ax, x, y, radius, *, circle_color, circle_ls):
        plot_radius = 1.03 * float(radius)

        ax.scatter(
            x,
            y,
            s=0.3,
            marker=".",
            linewidths=0,
            alpha=1.0,
            color="black",
            rasterized=True,
            zorder=2,
        )

        draw_circle(
            ax,
            (0.0, 0.0),
            radius,
            color=circle_color,
            linestyle=circle_ls,
            lw=0.8,
            alpha=0.95,
            zorder=3,
            clip_on=False,
        )

        ax.set_xlim(-plot_radius, plot_radius)
        ax.set_ylim(-plot_radius, plot_radius)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("")
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Left column: DESI in row 1; rows 2 and 3 empty.
    ax_desi = axes[0, 0]
    axes[1, 0].set_axis_off()
    axes[2, 0].set_axis_off()

    draw_panel(
        ax_desi,
        desi["x"],
        desi["y"],
        desi_radius,
        circle_color="black",
        circle_ls="--",
    )

    # Row 1: five RSD examples.
    for icol, s in enumerate(rsd_selected, start=1):
        draw_panel(
            axes[0, icol],
            s["x"],
            s["y"],
            flamingo_radius,
            circle_color="hotpink",
            circle_ls="-",
        )

    # Row 2: same five real-space examples.
    for icol, s in enumerate(real_selected, start=1):
        draw_panel(
            axes[1, icol],
            s["x"],
            s["y"],
            flamingo_radius,
            circle_color="cornflowerblue",
            circle_ls="-",
        )

    # Row 3: same five DM-particle examples.
    for icol, s in enumerate(dm_selected, start=1):
        draw_panel(
            axes[2, icol],
            s["x"],
            s["y"],
            flamingo_radius,
            circle_color="0.5",
            circle_ls="-",
        )

    # Label for DESI.
    ax_desi.text(
        0.5,
        -0.08,
        "DESI DR1\n" + r"$R = 175\ h^{-1}\ \mathrm{Mpc}$",
        transform=ax_desi.transAxes,
        ha="center",
        va="top",
        color="black",
        clip_on=False,
    )

    # Vertical row labels on the right.
    right_ax_top = axes[0, 5]
    right_ax_mid = axes[1, 5]
    right_ax_bot = axes[2, 5]

    right_ax_top.text(
        1.06,
        0.5,
        "FLAMINGO + RSD",
        transform=right_ax_top.transAxes,
        rotation=270,
        ha="left",
        va="center",
        color="hotpink",
        clip_on=False,
        fontsize=11.5
    )

    right_ax_mid.text(
        1.06,
        0.5,
        "FLAMINGO, real-space",
        transform=right_ax_mid.transAxes,
        rotation=270,
        ha="left",
        va="center",
        color="cornflowerblue",
        clip_on=False,
        fontsize=11.5
    )

    right_ax_bot.text(
        1.06,
        0.5,
        "FLAMINGO particles",
        transform=right_ax_bot.transAxes,
        rotation=270,
        ha="left",
        va="center",
        color="0.5",
        clip_on=False,
        fontsize=11.5
    )

    outfile = FIGURES_DIR / (
        f"scatter_DESI_FLAMINGO_small_examples_3x6_"
        f"left_column_DESI_top_"
        f"R{flamingo_radius:.0f}_Nnative.pdf"
    )
    save_figure(fig, outfile)

    print("DESI / FLAMINGO small-example scatter plot")
    print("------------------------------------------")
    print(f"Selected slice ids                  : {selected_islices}")
    print(f"DESI points plotted                 : {len(desi['x']):,}")
    print(f"RSD points per slice                : {rsd_selected[0]['n']:,}")
    print(f"Real-space points per slice         : {real_selected[0]['n']:,}")
    print(f"DM-particle points per slice        : {dm_selected[0]['n']:,}")
    print(f"DESI faintest plotted M_r           : {np.nanmax(desi['Mr']):.3f}")
    print(f"RSD faintest plotted M_AB           : {np.nanmax(rsd_selected[0]['m_sim_ab']):.3f}")
    print(f"Real-space faintest plotted M_AB    : {np.nanmax(real_selected[0]['m_sim_ab']):.3f}")
    print(f"FLAMINGO small-cylinder radius      : {flamingo_radius:.1f} h^-1 Mpc")


plot_desi_flamingo_small_examples()


# ## Three-panel footprint plot. Comparison to the Sloan Great Wall.
# 
# The three panels show the DESI footprint, SDSS, and S2 in observer-plane coordinates.

# In[20]:


# ---------------------------------------------------------------------------
# Three-panel DESI / SDSS / S2 footprint comparison
# ---------------------------------------------------------------------------

# Exact horizontal gaps between axes, in inches.
# These are measured from the right edge of one axes to the left edge of the next.
PANEL_GAP_12_IN = 0.2
PANEL_GAP_23_IN = 1.0

# Figure geometry, in inches.
FIGURE_WIDTH_IN = 9.6
FIGURE_HEIGHT_IN = 3.2

FIGURE_LEFT_MARGIN_IN = 0.55
FIGURE_RIGHT_MARGIN_IN = 0.10
FIGURE_BOTTOM_MARGIN_IN = 0.48
FIGURE_TOP_MARGIN_IN = 0.30

# S2 panel: original coordinates, no offsets or transformations.
# The x and y spans are both 600 fiducial units, so x/y data units remain proportional.
S2_XLIM = (-300.0, 300.0)
S2_YLIM = (-300.0, 300.0)


def data_span(xlim, ylim):
    """Return x-span and y-span for an axis range."""
    return (
        abs(float(xlim[1]) - float(xlim[0])),
        abs(float(ylim[1]) - float(ylim[0])),
    )


def make_three_panel_axes():
    """
    Create three axes with exact physical axis-to-axis gaps.

    All axes positions are computed in inches. The gaps are exactly the
    distances between axes boundaries:
        panel 1 right edge -> panel 2 left edge = PANEL_GAP_12_IN
        panel 2 right edge -> panel 3 left edge = PANEL_GAP_23_IN

    The panels use a common physical data scale, so equal data distances
    correspond to equal lengths on the page.
    """
    desi_xlim = (-D_MAX_HMPC, 0.0)
    desi_ylim = (-300.0, 300.0)

    sdss_xlim = (-320.0, -80.0)
    sdss_ylim = (-300.0, 300.0)

    panel_limits = [
        (desi_xlim, desi_ylim),
        (sdss_xlim, sdss_ylim),
        (S2_XLIM, S2_YLIM),
    ]

    x_spans = np.array([
        data_span(xlim, ylim)[0]
        for xlim, ylim in panel_limits
    ])
    y_spans = np.array([
        data_span(xlim, ylim)[1]
        for xlim, ylim in panel_limits
    ])

    available_width_in = (
        FIGURE_WIDTH_IN
        - FIGURE_LEFT_MARGIN_IN
        - FIGURE_RIGHT_MARGIN_IN
        - PANEL_GAP_12_IN
        - PANEL_GAP_23_IN
    )

    available_height_in = (
        FIGURE_HEIGHT_IN
        - FIGURE_BOTTOM_MARGIN_IN
        - FIGURE_TOP_MARGIN_IN
    )

    if available_width_in <= 0.0:
        raise ValueError(
            "No horizontal space left for panels. Reduce margins or gaps."
        )

    if available_height_in <= 0.0:
        raise ValueError(
            "No vertical space left for panels. Reduce top/bottom margins."
        )

    # A common data scale s means:
    #   axes_width_i  = s * x_span_i
    #   axes_height_i = s * y_span_i
    #
    # Choose the largest scale that fits both the total available width and
    # the maximum available height.
    scale_from_width = available_width_in / np.sum(x_spans)
    scale_from_height = available_height_in / np.max(y_spans)
    data_scale_in_per_hmpc = min(scale_from_width, scale_from_height)

    axes_widths_in = data_scale_in_per_hmpc * x_spans
    axes_heights_in = data_scale_in_per_hmpc * y_spans

    total_axes_width_in = np.sum(axes_widths_in)
    total_block_width_in = (
        total_axes_width_in
        + PANEL_GAP_12_IN
        + PANEL_GAP_23_IN
    )

    # Centre the whole axes+gaps block inside the left/right margins.
    block_left_in = (
        FIGURE_LEFT_MARGIN_IN
        + 0.5
        * (
            FIGURE_WIDTH_IN
            - FIGURE_LEFT_MARGIN_IN
            - FIGURE_RIGHT_MARGIN_IN
            - total_block_width_in
        )
    )

    centre_y_in = (
        FIGURE_BOTTOM_MARGIN_IN
        + 0.5 * available_height_in
    )

    left0_in = block_left_in
    left1_in = left0_in + axes_widths_in[0] + PANEL_GAP_12_IN
    left2_in = left1_in + axes_widths_in[1] + PANEL_GAP_23_IN

    bottom0_in = centre_y_in - 0.5 * axes_heights_in[0]
    bottom1_in = centre_y_in - 0.5 * axes_heights_in[1]
    bottom2_in = centre_y_in - 0.5 * axes_heights_in[2]

    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))

    def add_axis_inches(left_in, bottom_in, width_in, height_in):
        return fig.add_axes([
            left_in / FIGURE_WIDTH_IN,
            bottom_in / FIGURE_HEIGHT_IN,
            width_in / FIGURE_WIDTH_IN,
            height_in / FIGURE_HEIGHT_IN,
        ])

    ax_desi = add_axis_inches(
        left0_in,
        bottom0_in,
        axes_widths_in[0],
        axes_heights_in[0],
    )

    ax_sdss = add_axis_inches(
        left1_in,
        bottom1_in,
        axes_widths_in[1],
        axes_heights_in[1],
    )

    ax_s2 = add_axis_inches(
        left2_in,
        bottom2_in,
        axes_widths_in[2],
        axes_heights_in[2],
    )

    actual_gap_12_in = left1_in - (left0_in + axes_widths_in[0])
    actual_gap_23_in = left2_in - (left1_in + axes_widths_in[1])

    print("Panel layout")
    print("------------")
    print(
        f"Axes 1 width x height: "
        f"{axes_widths_in[0]:.3f} x {axes_heights_in[0]:.3f} inch"
    )
    print(
        f"Axes 2 width x height: "
        f"{axes_widths_in[1]:.3f} x {axes_heights_in[1]:.3f} inch"
    )
    print(
        f"Axes 3 width x height: "
        f"{axes_widths_in[2]:.3f} x {axes_heights_in[2]:.3f} inch"
    )
    print(f"Axis-to-axis gap 1-2 : {actual_gap_12_in:.3f} inch")
    print(f"Axis-to-axis gap 2-3 : {actual_gap_23_in:.3f} inch")
    print(
        f"Common data scale    : "
        f"{data_scale_in_per_hmpc:.6f} inch / h^-1 Mpc"
    )

    return fig, (ax_desi, ax_sdss, ax_s2), panel_limits


def make_desi_sdss_s2_three_panel():
    """Make the DESI/SDSS/S2 three-panel footprint comparison."""
    mask_desi_panel = (
        np.isfinite(x_obs_signed_all)
        & np.isfinite(y_obs_all)
        & np.isfinite(z_obs_all)
        & np.isfinite(Mr_working)
        & (Mr_working < FULL_FOOTPRINT_MR_LIMIT)
        & (np.abs(z_obs_all) <= CYLINDER_HALF_THICKNESS_HMPC)
        & (D_x_all >= 0.0)
        & (D_x_all <= D_MAX_HMPC)
    )

    mask_sdss_panel = (
        mask_sdss_footprint_panel
        & (D_sdss_hmpc >= 0.0)
        & (D_sdss_hmpc <= D_MAX_HMPC)
    )

    mask_s2_panel = mask_s2_fiducial

    fig, axes, panel_limits = make_three_panel_axes()
    ax_desi, ax_sdss, ax_s2 = axes

    desi_on_sdss_panel = (
        mask_desi_panel
        & np.isfinite(x_obs_signed_all)
        & np.isfinite(y_obs_all)
        & (x_obs_signed_all >= min(panel_limits[1][0]))
        & (x_obs_signed_all <= max(panel_limits[1][0]))
        & (y_obs_all >= min(panel_limits[1][1]))
        & (y_obs_all <= max(panel_limits[1][1]))
    )

    ax_desi.scatter(
        x_obs_signed_all[mask_desi_panel],
        y_obs_all[mask_desi_panel],
        s=0.2,
        marker=".",
        linewidths=0,
        alpha=1.0,
        color="black",
        rasterized=True,
        zorder=2,
    )

    ax_sdss.scatter(
        x_obs_signed_all[desi_on_sdss_panel],
        y_obs_all[desi_on_sdss_panel],
        s=0.2,
        marker=".",
        linewidths=0,
        alpha=0.3,
        color="cornflowerblue",
        rasterized=True,
        zorder=1,
    )

    ax_sdss.scatter(
        x_obs_signed_sdss[mask_sdss_panel],
        y_obs_sdss[mask_sdss_panel],
        s=0.4,
        marker=".",
        linewidths=0,
        alpha=0.5,
        color="black",
        rasterized=True,
        zorder=2,
    )

    # S2: original coordinates, no offset, no transformation, no overlays.
    ax_s2.scatter(
        s2_x[mask_s2_panel],
        s2_y[mask_s2_panel],
        s=0.3,
        marker=".",
        linewidths=0,
        alpha=1.0,
        color="black",
        rasterized=True,
        zorder=2,
    )

    # Set the fixed panel limits.
    ax_desi.set_xlim(*panel_limits[0][0])
    ax_desi.set_ylim(*panel_limits[0][1])

    ax_sdss.set_xlim(*panel_limits[1][0])
    ax_sdss.set_ylim(*panel_limits[1][1])

    ax_s2.set_xlim(*panel_limits[2][0])
    ax_s2.set_ylim(*panel_limits[2][1])

    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(
            direction="in",
            top=True,
            right=True,
            which="both",
        )

    ax_desi.set_xlabel(r"$x\ [h^{-1}\,\mathrm{Mpc}]$")
    ax_sdss.set_xlabel(r"$x\ [h^{-1}\,\mathrm{Mpc}]$")
    ax_s2.set_xlabel(r"$x\ \mathrm{(fiducial)}$")

    ax_desi.set_ylabel(r"$y\ [h^{-1}\,\mathrm{Mpc}]$")
    ax_s2.set_ylabel(r"$y\ \mathrm{(fiducial)}$")

    ax_sdss.tick_params(
        axis="y",
        labelleft=False,
        labelright=False,
    )

    ax_desi.set_title(
        rf"DESI, $M_r<{FULL_FOOTPRINT_MR_LIMIT:g}$",
        fontsize=9,
    )
    ax_sdss.set_title("SDSS", fontsize=9)
    ax_s2.set_title("S2 (fiducial)", fontsize=9)

    outfile = FIGURES_DIR / "DESI_SDSS_S2_three_panel.pdf"
    save_figure(fig, outfile)

    print("Three-panel footprint counts")
    print("----------------------------")
    print(f"DESI points: {np.count_nonzero(mask_desi_panel):,}")
    print(f"SDSS points: {np.count_nonzero(mask_sdss_panel):,}")
    print(f"S2 points  : {np.count_nonzero(mask_s2_panel):,}")
    print(
        f"Requested panel gap 1-2          : "
        f"{PANEL_GAP_12_IN:.3f} inch axis-to-axis"
    )
    print(
        f"Requested panel gap 2-3          : "
        f"{PANEL_GAP_23_IN:.3f} inch axis-to-axis"
    )
    print(
        f"S2 x range                       : "
        f"[{S2_XLIM[0]:.1f}, {S2_XLIM[1]:.1f}]"
    )
    print(
        f"S2 y range                       : "
        f"[{S2_YLIM[0]:.1f}, {S2_YLIM[1]:.1f}]"
    )


make_desi_sdss_s2_three_panel()


# ## Compare the S2 data to DESI data, and to DESI with mistaken coordiantes.

# In[21]:


# ---------------------------------------------------------------------------
# Three-panel DESI / SDSS / S2 footprint comparison
# ---------------------------------------------------------------------------

# Exact horizontal gaps between axes, in inches.
# These are measured from the right edge of one axes to the left edge of the next.
PANEL_GAP_12_IN = 0.2
PANEL_GAP_23_IN = 1.0

# Figure geometry, in inches.
FIGURE_WIDTH_IN = 9.6
FIGURE_HEIGHT_IN = 3.2

FIGURE_LEFT_MARGIN_IN = 0.55
FIGURE_RIGHT_MARGIN_IN = 0.10
FIGURE_BOTTOM_MARGIN_IN = 0.48
FIGURE_TOP_MARGIN_IN = 0.30

# S2 panel: original coordinates, no offsets or transformations.
# The x and y spans are both 600 fiducial units, so x/y data units remain proportional.
S2_XLIM = (-300.0, 300.0)
S2_YLIM = (-300.0, 300.0)


def data_span(xlim, ylim):
    """Return x-span and y-span for an axis range."""
    return (
        abs(float(xlim[1]) - float(xlim[0])),
        abs(float(ylim[1]) - float(ylim[0])),
    )


def make_three_panel_axes():
    """
    Create three axes with exact physical axis-to-axis gaps.

    All axes positions are computed in inches. The gaps are exactly the
    distances between axes boundaries:
        panel 1 right edge -> panel 2 left edge = PANEL_GAP_12_IN
        panel 2 right edge -> panel 3 left edge = PANEL_GAP_23_IN

    The panels use a common physical data scale, so equal data distances
    correspond to equal lengths on the page.
    """
    desi_xlim = (-D_MAX_HMPC, 0.0)
    desi_ylim = (-300.0, 300.0)

    sdss_xlim = (-320.0, -80.0)
    sdss_ylim = (-300.0, 300.0)

    panel_limits = [
        (desi_xlim, desi_ylim),
        (sdss_xlim, sdss_ylim),
        (S2_XLIM, S2_YLIM),
    ]

    x_spans = np.array([
        data_span(xlim, ylim)[0]
        for xlim, ylim in panel_limits
    ])
    y_spans = np.array([
        data_span(xlim, ylim)[1]
        for xlim, ylim in panel_limits
    ])

    available_width_in = (
        FIGURE_WIDTH_IN
        - FIGURE_LEFT_MARGIN_IN
        - FIGURE_RIGHT_MARGIN_IN
        - PANEL_GAP_12_IN
        - PANEL_GAP_23_IN
    )

    available_height_in = (
        FIGURE_HEIGHT_IN
        - FIGURE_BOTTOM_MARGIN_IN
        - FIGURE_TOP_MARGIN_IN
    )

    if available_width_in <= 0.0:
        raise ValueError(
            "No horizontal space left for panels. Reduce margins or gaps."
        )

    if available_height_in <= 0.0:
        raise ValueError(
            "No vertical space left for panels. Reduce top/bottom margins."
        )

    # A common data scale s means:
    #   axes_width_i  = s * x_span_i
    #   axes_height_i = s * y_span_i
    #
    # Choose the largest scale that fits both the total available width and
    # the maximum available height.
    scale_from_width = available_width_in / np.sum(x_spans)
    scale_from_height = available_height_in / np.max(y_spans)
    data_scale_in_per_hmpc = min(scale_from_width, scale_from_height)

    axes_widths_in = data_scale_in_per_hmpc * x_spans
    axes_heights_in = data_scale_in_per_hmpc * y_spans

    total_axes_width_in = np.sum(axes_widths_in)
    total_block_width_in = (
        total_axes_width_in
        + PANEL_GAP_12_IN
        + PANEL_GAP_23_IN
    )

    # Centre the whole axes+gaps block inside the left/right margins.
    block_left_in = (
        FIGURE_LEFT_MARGIN_IN
        + 0.5
        * (
            FIGURE_WIDTH_IN
            - FIGURE_LEFT_MARGIN_IN
            - FIGURE_RIGHT_MARGIN_IN
            - total_block_width_in
        )
    )

    centre_y_in = (
        FIGURE_BOTTOM_MARGIN_IN
        + 0.5 * available_height_in
    )

    left0_in = block_left_in
    left1_in = left0_in + axes_widths_in[0] + PANEL_GAP_12_IN
    left2_in = left1_in + axes_widths_in[1] + PANEL_GAP_23_IN

    bottom0_in = centre_y_in - 0.5 * axes_heights_in[0]
    bottom1_in = centre_y_in - 0.5 * axes_heights_in[1]
    bottom2_in = centre_y_in - 0.5 * axes_heights_in[2]

    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))

    def add_axis_inches(left_in, bottom_in, width_in, height_in):
        return fig.add_axes([
            left_in / FIGURE_WIDTH_IN,
            bottom_in / FIGURE_HEIGHT_IN,
            width_in / FIGURE_WIDTH_IN,
            height_in / FIGURE_HEIGHT_IN,
        ])

    ax_desi = add_axis_inches(
        left0_in,
        bottom0_in,
        axes_widths_in[0],
        axes_heights_in[0],
    )

    ax_sdss = add_axis_inches(
        left1_in,
        bottom1_in,
        axes_widths_in[1],
        axes_heights_in[1],
    )

    ax_s2 = add_axis_inches(
        left2_in,
        bottom2_in,
        axes_widths_in[2],
        axes_heights_in[2],
    )

    actual_gap_12_in = left1_in - (left0_in + axes_widths_in[0])
    actual_gap_23_in = left2_in - (left1_in + axes_widths_in[1])

    print("Panel layout")
    print("------------")
    print(
        f"Axes 1 width x height: "
        f"{axes_widths_in[0]:.3f} x {axes_heights_in[0]:.3f} inch"
    )
    print(
        f"Axes 2 width x height: "
        f"{axes_widths_in[1]:.3f} x {axes_heights_in[1]:.3f} inch"
    )
    print(
        f"Axes 3 width x height: "
        f"{axes_widths_in[2]:.3f} x {axes_heights_in[2]:.3f} inch"
    )
    print(f"Axis-to-axis gap 1-2 : {actual_gap_12_in:.3f} inch")
    print(f"Axis-to-axis gap 2-3 : {actual_gap_23_in:.3f} inch")
    print(
        f"Common data scale    : "
        f"{data_scale_in_per_hmpc:.6f} inch / h^-1 Mpc"
    )

    return fig, (ax_desi, ax_sdss, ax_s2), panel_limits


def make_desi_sdss_s2_three_panel():
    """Make the DESI/SDSS/S2 three-panel footprint comparison."""
    mask_desi_panel = (
        np.isfinite(x_obs_signed_all)
        & np.isfinite(y_obs_all)
        & np.isfinite(z_obs_all)
        & np.isfinite(Mr_working)
        & (Mr_working < FULL_FOOTPRINT_MR_LIMIT)
        & (np.abs(z_obs_all) <= CYLINDER_HALF_THICKNESS_HMPC)
        & (D_x_all >= 0.0)
        & (D_x_all <= D_MAX_HMPC)
    )

    mask_sdss_panel = (
        mask_sdss_footprint_panel
        & (D_sdss_hmpc >= 0.0)
        & (D_sdss_hmpc <= D_MAX_HMPC)
    )

    mask_s2_panel = mask_s2_fiducial

    fig, axes, panel_limits = make_three_panel_axes()
    ax_desi, ax_sdss, ax_s2 = axes

    desi_on_sdss_panel = (
        mask_desi_panel
        & np.isfinite(x_obs_signed_all)
        & np.isfinite(y_obs_all)
        & (x_obs_signed_all >= min(panel_limits[1][0]))
        & (x_obs_signed_all <= max(panel_limits[1][0]))
        & (y_obs_all >= min(panel_limits[1][1]))
        & (y_obs_all <= max(panel_limits[1][1]))
    )

    ax_desi.scatter(
        x_obs_signed_all[mask_desi_panel],
        y_obs_all[mask_desi_panel],
        s=0.2,
        marker=".",
        linewidths=0,
        alpha=1.0,
        color="black",
        rasterized=True,
        zorder=2,
    )

    ax_sdss.scatter(
        x_obs_signed_all[desi_on_sdss_panel],
        y_obs_all[desi_on_sdss_panel],
        s=0.2,
        marker=".",
        linewidths=0,
        alpha=0.3,
        color="cornflowerblue",
        rasterized=True,
        zorder=1,
    )

    ax_sdss.scatter(
        x_obs_signed_sdss[mask_sdss_panel],
        y_obs_sdss[mask_sdss_panel],
        s=0.4,
        marker=".",
        linewidths=0,
        alpha=0.5,
        color="black",
        rasterized=True,
        zorder=2,
    )

    # S2: original coordinates, no offset, no transformation, no overlays.
    ax_s2.scatter(
        s2_x[mask_s2_panel],
        s2_y[mask_s2_panel],
        s=0.3,
        marker=".",
        linewidths=0,
        alpha=1.0,
        color="black",
        rasterized=True,
        zorder=2,
    )

    # Set the fixed panel limits.
    ax_desi.set_xlim(*panel_limits[0][0])
    ax_desi.set_ylim(*panel_limits[0][1])

    ax_sdss.set_xlim(*panel_limits[1][0])
    ax_sdss.set_ylim(*panel_limits[1][1])

    ax_s2.set_xlim(*panel_limits[2][0])
    ax_s2.set_ylim(*panel_limits[2][1])

    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(
            direction="in",
            top=True,
            right=True,
            which="both",
        )

    ax_desi.set_xlabel(r"$x\ [h^{-1}\,\mathrm{Mpc}]$")
    ax_sdss.set_xlabel(r"$x\ [h^{-1}\,\mathrm{Mpc}]$")
    ax_s2.set_xlabel(r"$x\ \mathrm{(fiducial)}$")

    ax_desi.set_ylabel(r"$y\ [h^{-1}\,\mathrm{Mpc}]$")
    ax_s2.set_ylabel(r"$y\ \mathrm{(fiducial)}$")

    ax_sdss.tick_params(
        axis="y",
        labelleft=False,
        labelright=False,
    )

    ax_desi.set_title(
        rf"DESI, $M_r<{FULL_FOOTPRINT_MR_LIMIT:g}$",
        fontsize=9,
    )
    ax_sdss.set_title("SDSS", fontsize=9)
    ax_s2.set_title("S2 (fiducial)", fontsize=9)

    outfile = FIGURES_DIR / "DESI_SDSS_S2_three_panel.pdf"
    save_figure(fig, outfile)

    print("Three-panel footprint counts")
    print("----------------------------")
    print(f"DESI points: {np.count_nonzero(mask_desi_panel):,}")
    print(f"SDSS points: {np.count_nonzero(mask_sdss_panel):,}")
    print(f"S2 points  : {np.count_nonzero(mask_s2_panel):,}")
    print(
        f"Requested panel gap 1-2          : "
        f"{PANEL_GAP_12_IN:.3f} inch axis-to-axis"
    )
    print(
        f"Requested panel gap 2-3          : "
        f"{PANEL_GAP_23_IN:.3f} inch axis-to-axis"
    )
    print(
        f"S2 x range                       : "
        f"[{S2_XLIM[0]:.1f}, {S2_XLIM[1]:.1f}]"
    )
    print(
        f"S2 y range                       : "
        f"[{S2_YLIM[0]:.1f}, {S2_YLIM[1]:.1f}]"
    )


make_desi_sdss_s2_three_panel()


# In[ ]:





# ## Projected 2D power spectra
# 
# The projected spectra are computed by depositing the points on a square grid covering the circular aperture, subtracting the mean surface density inside the aperture, applying the aperture mask, and binning \(|\delta_k|^2\) in circular \(k\)-bins.  Common \(k\)-bins are used for all comparison sizes.  The large DESI/FLAMINGO/DM samples are matched to the S2 fiducial surface density; the small samples are unchanged.
# 

# In[22]:


# ---------------------------------------------------------------------------
# Projected 2D power-spectrum helpers
# ---------------------------------------------------------------------------
def common_power_k_edges(radii_hmpc, *, ngrid=NGRID_POWER):
    """Construct one common k-bin definition for all requested aperture radii."""
    radii = np.asarray(radii_hmpc, dtype=float)
    r_max = float(np.nanmax(radii))

    k_fund = 2.0 * np.pi / (2.0 * r_max)
    dx_max = (2.0 * r_max) / ngrid
    k_nyq_max_radius = np.pi / dx_max
    k_max = POWER_K_NYQUIST_FRACTION * k_nyq_max_radius

    return np.geomspace(k_fund, k_max, N_K_BINS + 1)


def make_power_spectrum_grid(radius_hmpc, *, ngrid=NGRID_POWER, k_edges=None):
    """Precompute FFT/grid/binning arrays for a circular aperture."""
    radius = float(radius_hmpc)
    box_size = 2.0 * radius
    dx = box_size / ngrid

    edges = np.linspace(-radius, radius, ngrid + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    xx, yy = np.meshgrid(centres, centres, indexing="ij")
    aperture_mask = (xx**2 + yy**2) <= radius**2

    kfreq = 2.0 * np.pi * np.fft.fftfreq(ngrid, d=dx)
    kx, ky = np.meshgrid(kfreq, kfreq, indexing="ij")
    kk = np.sqrt(kx**2 + ky**2)

    if k_edges is None:
        k_edges = np.geomspace(2.0 * np.pi / box_size, POWER_K_NYQUIST_FRACTION * np.pi / dx, N_K_BINS + 1)
    else:
        k_edges = np.asarray(k_edges, dtype=float)

    k_mid = np.sqrt(k_edges[:-1] * k_edges[1:])
    shell_index = np.digitize(kk.ravel(), k_edges) - 1
    valid_shell = (shell_index >= 0) & (shell_index < len(k_mid))
    shell_index_valid = shell_index[valid_shell]
    nmodes = np.bincount(shell_index_valid, minlength=len(k_mid)).astype(int)

    return {
        "radius": radius,
        "ngrid": ngrid,
        "box_size": box_size,
        "dx": dx,
        "edges": edges,
        "aperture_mask": aperture_mask,
        "k_edges": k_edges,
        "k_mid": k_mid,
        "valid_shell": valid_shell,
        "shell_index_valid": shell_index_valid,
        "nmodes": nmodes,
        "area_disk": np.pi * radius**2,
    }


def projected_power_spectrum_2d(x, y, grid):
    """Compute the windowed projected 2D power spectrum for points in a disk."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]

    radius = float(grid["radius"])
    inside = (x**2 + y**2) <= radius**2
    x = x[inside]
    y = y[inside]

    n_data = len(x)
    ngrid = int(grid["ngrid"])
    area = float(grid["area_disk"])

    if n_data == 0:
        return grid["k_mid"], np.full_like(grid["k_mid"], np.nan), {
            "n_data": 0,
            "shot_noise": np.nan,
            "surface_density": np.nan,
        }

    hist, _, _ = np.histogram2d(x, y, bins=[grid["edges"], grid["edges"]])
    cell_area = grid["dx"]**2
    mean_counts_per_cell_inside = n_data * cell_area / area

    delta = np.zeros((ngrid, ngrid), dtype=float)
    inside_mask = grid["aperture_mask"]
    delta[inside_mask] = hist[inside_mask] / mean_counts_per_cell_inside - 1.0
    delta[~inside_mask] = 0.0

    delta_k = np.fft.fftn(delta) * cell_area
    power_2d = (np.abs(delta_k)**2) / area

    flat_power = power_2d.ravel()
    valid_power = flat_power[grid["valid_shell"]]

    weighted_sum = np.bincount(
        grid["shell_index_valid"],
        weights=valid_power,
        minlength=len(grid["k_mid"]),
    )
    nmodes = grid["nmodes"]
    pk = np.full_like(grid["k_mid"], np.nan, dtype=float)
    ok = nmodes > 0
    pk[ok] = weighted_sum[ok] / nmodes[ok]

    info = {
        "n_data": int(n_data),
        "shot_noise": area / n_data,
        "surface_density": n_data / area,
    }
    return grid["k_mid"], pk, info


def compute_slice_power_spectra(slices, grid, *, n_slices=N_POWER_SIM_SLICES):
    """Compute P(k) for a list of FLAMINGO slices."""
    slices = sorted(slices, key=lambda s: s["islice"])
    if n_slices is not None:
        slices = slices[:int(n_slices)]

    results = []
    for s in slices:
        k, pk, info = projected_power_spectrum_2d(s["x"], s["y"], grid)
        results.append({
            "k": k,
            "pk": pk,
            "info": info,
            "slice": s,
        })
    return results


def ensemble_power_spectrum_summary(results):
    """Return k, median, p16, p84, and median shot noise for an ensemble."""
    if len(results) == 0:
        raise ValueError("No power spectra supplied")
    k_values = np.asarray(results[0]["k"], dtype=float)
    pk_stack = np.vstack([np.asarray(res["pk"], dtype=float) for res in results])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pk_median = np.nanmedian(pk_stack, axis=0)
        pk_p16 = np.nanpercentile(pk_stack, 16.0, axis=0)
        pk_p84 = np.nanpercentile(pk_stack, 84.0, axis=0)

    pshot = np.asarray([res["info"]["shot_noise"] for res in results], dtype=float)
    return k_values, pk_median, pk_p16, pk_p84, float(np.nanmedian(pshot))


def set_power_axes(ax):
    """Apply common axis styling to power-spectrum plots."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
    ax.set_ylabel(r"$P_{\rm 2D}(k)\ [(h^{-1}\,\mathrm{Mpc})^2]$")
    ax.tick_params(direction="in", top=True, right=True, which="both")


# In[23]:


# ---------------------------------------------------------------------------
# Compute DESI, S2, FLAMINGO, and DM-particle projected 2D P(k)
# ---------------------------------------------------------------------------
all_power_radii = [S2_FIDUCIAL_RADIUS_HMPC] + [
    float(region["radius_hmpc"])
    for region in COMPARISON_REGIONS.values()
]
POWER_K_EDGES_COMMON = common_power_k_edges(all_power_radii)
POWER_K_MID_COMMON = np.sqrt(POWER_K_EDGES_COMMON[:-1] * POWER_K_EDGES_COMMON[1:])

power_grids = {
    float(radius): make_power_spectrum_grid(float(radius), k_edges=POWER_K_EDGES_COMMON)
    for radius in sorted(set(all_power_radii))
}

# S2 in fiducial coordinates.
k_s2, pk_s2, info_s2 = projected_power_spectrum_2d(
    s2_x[mask_s2_fiducial],
    s2_y[mask_s2_fiducial],
    power_grids[S2_FIDUCIAL_RADIUS_HMPC],
)

pk_desi_by_region = {}
pk_flamingo_by_region_case = {name: {} for name in COMPARISON_REGIONS}
pk_dm_by_region = {}

print("Common P(k) bins")
print("----------------")
print(f"k_min edge = {POWER_K_EDGES_COMMON[0]:.5g} h Mpc^-1")
print(f"k_max edge = {POWER_K_EDGES_COMMON[-1]:.5g} h Mpc^-1")
print(f"N bins     = {len(POWER_K_MID_COMMON)}")
print()
print("S2 fiducial")
print("------------")
print(f"N used = {info_s2['n_data']:,}")
print(f"shot-noise area/N = {info_s2['shot_noise']:.3e} (h^-1 Mpc)^2")

for region_name, region in COMPARISON_REGIONS.items():
    radius = float(region["radius_hmpc"])
    grid = power_grids[radius]
    desi = desi_regions[region_name]

    k_desi, pk_desi, info_desi = projected_power_spectrum_2d(desi["x"], desi["y"], grid)
    pk_desi_by_region[region_name] = {
        "k": k_desi,
        "pk": pk_desi,
        "info": info_desi,
    }

    print("\n" + "-" * 72)
    print(f"DESI {region_name}: R={radius:g} h^-1 Mpc, M_r<{region['desi_mr_limit']:g}")
    print(f"N used = {info_desi['n_data']:,}")
    print(f"shot-noise area/N = {info_desi['shot_noise']:.3e} (h^-1 Mpc)^2")

    for case in CASE_ORDER:
        results = compute_slice_power_spectra(
            slices_by_region_case[region_name][case],
            grid,
            n_slices=N_POWER_SIM_SLICES,
        )
        pk_flamingo_by_region_case[region_name][case] = results
        counts = np.asarray([res["info"]["n_data"] for res in results], dtype=int)
        pshot = np.asarray([res["info"]["shot_noise"] for res in results], dtype=float)
        print(
            f"FLAMINGO {case}: {len(results)} slices, "
            f"N median={int(np.median(counts)):,}, "
            f"shot-noise median={np.nanmedian(pshot):.3e}"
        )

    dm_results = compute_slice_power_spectra(
        dm_slices_by_region[region_name],
        grid,
        n_slices=N_POWER_SIM_SLICES,
    )
    pk_dm_by_region[region_name] = dm_results
    dm_counts = np.asarray([res["info"]["n_data"] for res in dm_results], dtype=int)
    dm_pshot = np.asarray([res["info"]["shot_noise"] for res in dm_results], dtype=float)
    print(
        f"FLAMINGO DM particles, real-space: {len(dm_results)} slices, "
        f"N median={int(np.median(dm_counts)):,}, "
        f"shot-noise median={np.nanmedian(dm_pshot):.3e}"
    )


# ## Power-spectrum summary plots
# 
# These plots compare DESI and S2 to the median and 16--84 percentile range of the FLAMINGO (RSD, real-space, particles).

# In[24]:


# ---------------------------------------------------------------------------
# Power-spectrum summary plots
# ---------------------------------------------------------------------------
def power_summary_suffix(*, include_s2=False, include_shot_noise=False):
    """Return filename suffix matching the plotted content."""
    s2_part = "with_S2" if include_s2 else "no_S2"
    shot_part = "with_shotnoise" if include_shot_noise else "no_shotnoise"
    return f"{s2_part}_{shot_part}"


def _is_rsd_case(case):
    """Return True for the FLAMINGO redshift-space case."""
    text = CASE_LABELS[case].lower()
    return ("redshift" in text) or ("rsd" in text)


def _is_real_case(case):
    """Return True for the FLAMINGO, real-space case."""
    text = CASE_LABELS[case].lower()
    return ("real" in text)


def plot_power_summary(region_name, *, include_s2=False, include_shot_noise=False):
    """Plot DESI, FLAMINGO galaxy, and DM-particle summary spectra for one region."""
    region = COMPARISON_REGIONS[region_name]
    radius = float(region["radius_hmpc"])
    desi_power = pk_desi_by_region[region_name]

    fig, ax = plt.subplots(figsize=POWER_FIGSIZE, constrained_layout=True)

    line_dm = None
    line_s2 = None
    flamingo_handles = {}
    flamingo_labels = {}
    shot_noise_handles = []
    shot_noise_labels = []

    # Lowest z-order: S2.
    if include_s2:
        (line_s2,) = ax.loglog(
            k_s2,
            pk_s2,
            color="0.2",
            lw=1.8,
            ls="-.",
            label=rf"S2 fiducial, $R={S2_FIDUCIAL_RADIUS_HMPC:g}$",
            zorder=2,
        )

    # DM particles, real-space.
    if region_name in pk_dm_by_region:
        k_dm, pk_dm_med, pk_dm_p16, pk_dm_p84, pshot_dm_med = ensemble_power_spectrum_summary(
            pk_dm_by_region[region_name]
        )
        positive_dm_band = (
            np.isfinite(k_dm)
            & (k_dm > 0.0)
            & np.isfinite(pk_dm_p16)
            & np.isfinite(pk_dm_p84)
            & (pk_dm_p16 > 0.0)
            & (pk_dm_p84 > 0.0)
        )

        ax.fill_between(
            k_dm,
            pk_dm_p16,
            pk_dm_p84,
            where=positive_dm_band,
            color="0.55",
            alpha=0.22,
            linewidth=0,
            zorder=4,
        )

        (line_dm,) = ax.loglog(
            k_dm,
            pk_dm_med,
            color="0.40",
            lw=1.8,
            label="DM particles, real-space",
            zorder=5,
        )

        if include_shot_noise and np.isfinite(pshot_dm_med):
            shot_dm = ax.axhline(
                pshot_dm_med,
                color="0.45",
                lw=0.8,
                ls=":",
                alpha=0.8,
                label="DM particles shot noise",
                zorder=4.5,
            )
            shot_noise_handles.append(shot_dm)
            shot_noise_labels.append("DM particles shot noise")

    # FLAMINGO galaxies: real-space below redshift-space.
    real_cases = [case for case in CASE_ORDER if _is_real_case(case)]
    rsd_cases = [case for case in CASE_ORDER if _is_rsd_case(case)]

    for case in real_cases + rsd_cases:
        results = pk_flamingo_by_region_case[region_name][case]
        k, pk_med, pk_p16, pk_p84, pshot_med = ensemble_power_spectrum_summary(results)
        color = CASE_COLORS[case]

        if _is_rsd_case(case):
            z_band = 12
            z_line = 13
        else:
            z_band = 8
            z_line = 9

        positive_band = (
            np.isfinite(k)
            & (k > 0.0)
            & np.isfinite(pk_p16)
            & np.isfinite(pk_p84)
            & (pk_p16 > 0.0)
            & (pk_p84 > 0.0)
        )

        ax.fill_between(
            k,
            pk_p16,
            pk_p84,
            where=positive_band,
            color=color,
            alpha=0.20,
            linewidth=0,
            zorder=z_band,
        )

        (line_case,) = ax.loglog(
            k,
            pk_med,
            color=color,
            lw=2.1,
            label=f"{CASE_LABELS[case]}",
            zorder=z_line,
        )

        flamingo_handles[case] = line_case
        flamingo_labels[case] = f"{CASE_LABELS[case]}"

        if include_shot_noise and np.isfinite(pshot_med):
            shot_case = ax.axhline(
                pshot_med,
                color=color,
                lw=0.8,
                ls=":",
                alpha=0.8,
                label=f"{CASE_LABELS[case]} shot noise",
                zorder=z_band,
            )
            shot_noise_handles.append(shot_case)
            shot_noise_labels.append(f"{CASE_LABELS[case]} shot noise")

    # Highest z-order: DESI.
    (line_desi,) = ax.loglog(
        desi_power["k"],
        desi_power["pk"],
        color="black",
        lw=2.4,
        ls="--",
        label=rf"DESI, $M_r<{region['desi_mr_limit']:g}$",
        zorder=20,
    )

    if include_shot_noise:
        ax.axhline(
            desi_power["info"]["shot_noise"],
            color="black",
            lw=0.9,
            ls=":",
            alpha=0.9,
            label=None,
            zorder=19,
        )

        if include_s2:
            ax.axhline(
                info_s2["shot_noise"],
                color="0.2",
                lw=0.9,
                ls=":",
                alpha=0.8,
                label=None,
                zorder=1.5,
            )

    # Legend order, top to bottom:
    # DESI, FLAMINGO RSD, FLAMINGO, real-space, DM particles, S2.
    ordered_handles = [line_desi]
    ordered_labels = [rf"DESI, $M_r<{np.round(region['desi_mr_limit'],2):g}$"]

    for case in rsd_cases:
        ordered_handles.append(flamingo_handles[case])
        ordered_labels.append(flamingo_labels[case])

    for case in real_cases:
        ordered_handles.append(flamingo_handles[case])
        ordered_labels.append(flamingo_labels[case])

    if line_dm is not None:
        ordered_handles.append(line_dm)
        ordered_labels.append("DM particles, real-space")

    if line_s2 is not None:
        ordered_handles.append(line_s2)
        ordered_labels.append(rf"S2 fiducial, $R={S2_FIDUCIAL_RADIUS_HMPC:g}$")

    if include_shot_noise:
        ordered_handles.extend(shot_noise_handles)
        ordered_labels.extend(shot_noise_labels)

    set_power_axes(ax)
    ax.legend(
        ordered_handles,
        ordered_labels,
        frameon=False,
        fontsize=POWER_LEGEND_FONTSIZE,
    )




    # ax.set_title(rf"{region_name}: $R={radius:g}\,h^{{-1}}\,\mathrm{{Mpc}}$")

    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)

    suffix = power_summary_suffix(
        include_s2=include_s2,
        include_shot_noise=include_shot_noise,
    )

    density_suffix = "S2density" if region_name == "large" else "native_density"
    outfile = FIGURES_DIR / f"pk2d_summary_DESI_vs_FLAMINGO_{region_name}_{suffix}_{density_suffix}.pdf"
    save_figure(fig, outfile)


# LARGE SCALE:
XLIM = [0.025, 2.5]
YLIM = [5, 2e3]

region_name = "large"

plot_power_summary(region_name, include_s2=True, include_shot_noise=False)


# ## Power-spectrum Individual plots
# 
# These plots compare DESI and S2 to the median and 16--84 percentile range of the FLAMINGO (RSD, real-space, particles separated out). Also includes the shot-noise (optionally)

# In[25]:


# ---------------------------------------------------------------------------
# Individual-slice power-spectrum helper
# ---------------------------------------------------------------------------
def plot_power_individual(
    region_name,
    *,
    include_s2=False,
    include_shot_noise=False,
    include_rsd=True,
    include_real=True,
    include_DM=True,
):
    """
    Plot DESI, optional S2, and selected individual FLAMINGO/DM spectra.

    include_rsd, include_real, and include_DM control both the individual
    spectra and the corresponding median/legend entries.
    """
    region = COMPARISON_REGIONS[region_name]
    radius = float(region["radius_hmpc"])
    desi_power = pk_desi_by_region[region_name]

    POWER_FIGSIZE = (3.2, 2.5)

    fig, ax = plt.subplots(figsize=POWER_FIGSIZE, constrained_layout=True)

    real_cases = [case for case in CASE_ORDER if _is_real_case(case)]
    rsd_cases = [case for case in CASE_ORDER if _is_rsd_case(case)]

    cases_to_plot = []
    if include_real:
        cases_to_plot.extend(real_cases)
    if include_rsd:
        cases_to_plot.extend(rsd_cases)

    dm_line = None
    flamingo_lines = {}
    shot_noise_handles = []
    shot_noise_labels = []

    # DM particles: individual spectra and median, in grey.
    if include_DM and region_name in pk_dm_by_region:
        dm_results = pk_dm_by_region[region_name]

        for res in dm_results:
            ax.loglog(
                res["k"],
                res["pk"],
                color="0.55",
                lw=0.55,
                alpha=0.45,
                zorder=3,
            )

        k_dm, pk_dm_med, pk_dm_p16, pk_dm_p84, pshot_dm_med = ensemble_power_spectrum_summary(
            dm_results
        )

        (dm_line,) = ax.loglog(
            k_dm,
            pk_dm_med,
            color="0.40",
            lw=2.0,
            label="DM particles, real-space",
            zorder=6,
        )


    # FLAMINGO galaxies: individual spectra and medians.
    for case in cases_to_plot:
        color = CASE_COLORS[case]
        results = pk_flamingo_by_region_case[region_name][case]

        if _is_rsd_case(case):
            z_individual = 12
            z_median = 16
            alpha_individual = 0.7
        else:
            z_individual = 8
            z_median = 14
            alpha_individual = 0.5

        for res in results:
            ax.loglog(
                res["k"],
                res["pk"],
                color=color,
                lw=0.55,
                alpha=alpha_individual,
                zorder=z_individual,
            )

        k, pk_med, pk_p16, pk_p84, pshot_med = ensemble_power_spectrum_summary(results)

        (line_case,) = ax.loglog(
            k,
            pk_med,
            color=color,
            lw=2.2,
            label=f"{CASE_LABELS[case]}",
            zorder=z_median,
        )
        flamingo_lines[case] = line_case

    # DESI.
    (desi_line,) = ax.loglog(
        desi_power["k"],
        desi_power["pk"],
        color="black",
        lw=2.4,
        ls="--",
        label=rf"DESI, $M_r<{np.round(region['desi_mr_limit'], 2):g}$",
        zorder=20,
    )

    # S2.
    s2_line = None
    if include_s2:
        (s2_line,) = ax.loglog(
            k_s2,
            pk_s2,
            color="0.2",
            lw=1.8,
            ls="-.",
            label=rf"S2 fiducial, $R={S2_FIDUCIAL_RADIUS_HMPC:g}$",
            zorder=18,
        )

    if include_shot_noise: # shot noise level is the same for all.
        shot_desi = ax.axhline(
            desi_power["info"]["shot_noise"],
            color="black",
            lw=1.8,
            ls=":",
            alpha=0.9,
            label="shot noise",
            zorder=19,
        )
        shot_noise_handles.append(shot_desi)
        shot_noise_labels.append(f"shot noise")


    # Legend order: DESI, FLAMINGO RSD, FLAMINGO, real-space, DM particles, S2.
    ordered_handles = [desi_line]
    ordered_labels = [rf"DESI, $M_r<{np.round(region['desi_mr_limit'],2):g}$"]

    if include_rsd:
        for case in rsd_cases:
            ordered_handles.append(flamingo_lines[case])
            ordered_labels.append(f"{CASE_LABELS[case]}")

    if include_real:
        for case in real_cases:
            ordered_handles.append(flamingo_lines[case])
            ordered_labels.append(f"{CASE_LABELS[case]}")

    if dm_line is not None:
        ordered_handles.append(dm_line)
        ordered_labels.append("DM particles, real-space")

    if s2_line is not None:
        ordered_handles.append(s2_line)
        ordered_labels.append(rf"S2 fiducial, $R={S2_FIDUCIAL_RADIUS_HMPC:g}$")

    if include_shot_noise:
        ordered_handles.extend(shot_noise_handles)
        ordered_labels.extend(shot_noise_labels)

    set_power_axes(ax)
    ax.legend(
        ordered_handles,
        ordered_labels,
        frameon=False,
        fontsize=POWER_LEGEND_FONTSIZE,
    )

    ax.set_xlim(XLIM)
    ax.set_ylim(desi_power["info"]["shot_noise"]*.8, desi_power["info"]["shot_noise"]*500)

    component_suffix = "_".join([
        "rsd" if include_rsd else "no_rsd",
        "real" if include_real else "no_real",
        "DM" if include_DM else "no_DM",
    ])
    s2_suffix = "with_S2" if include_s2 else "no_S2"
    shot_suffix = "with_shotnoise" if include_shot_noise else "no_shotnoise"

    density_suffix = "S2density" if region_name == "large" else "native_density"
    outfile = FIGURES_DIR / (
        f"pk2d_individual_DESI_vs_FLAMINGO_{region_name}_"
        f"{component_suffix}_{s2_suffix}_{shot_suffix}_{density_suffix}.pdf"
    )
    save_figure(fig, outfile)


# LARGE SCALE:
XLIM = [0.025, 2.5]
YLIM = [5, 2e3]

region_name = "large"

plot_power_individual(
    region_name,
    include_s2=False,
    include_shot_noise=True,
    include_rsd=True,
    include_real=False,
    include_DM=False,
)

plot_power_individual(
    region_name,
    include_s2=False,
    include_shot_noise=True,
    include_rsd=False,
    include_real=True,
    include_DM=False,
)

plot_power_individual(
    region_name,
    include_s2=False,
    include_shot_noise=True,
    include_rsd=False,
    include_real=False,
    include_DM=True,
)


# SMALL SCALE:
XLIM = [0.04, 2.5]
YLIM = [0.5, 2e3]

region_name = "small"

plot_power_individual(
    region_name,
    include_s2=False,
    include_shot_noise=True,
    include_rsd=True,
    include_real=False,
    include_DM=False,
)

plot_power_individual(
    region_name,
    include_s2=False,
    include_shot_noise=True,
    include_rsd=False,
    include_real=True,
    include_DM=False,
)

plot_power_individual(
    region_name,
    include_s2=False,
    include_shot_noise=True,
    include_rsd=False,
    include_real=False,
    include_DM=True,
)


# ## Angular Distribution of Pairwise Distances (ADPD)
# 
# This section repeats the angular-dependence analysis used in Sylos Labini & Galoppo (2026) for the DESI, S2, FLAMINGO galaxy, and FLAMINGO DM-particle samples constructed above.
# 
# The public Nature code describes the ADPD estimator as follows: use projected 2D points, form all point pairs, bin pair separations in radial shells, bin pair orientations in \([0,180^\circ)\), normalize the angular counts in each radial shell, and compute
# \[
# \sigma_\theta^2(r) = \frac{1}{N_\theta}\sum_i \left[p_i(r) - \frac{1}{N_\theta}\right]^2 .
# \]
# 
# A technical issue in the public `Compute_adpd_python.py` script is that the radial-bin limits are taken from the **radii of the individual points from the origin**, not explicitly from the pair-separation range. The implementation below keeps the intended pair-separation estimator but sets the pair-separation bin edges explicitly. By default it follows the effective range of the public script, \(0 < r < R\), where \(R\) is the aperture radius. Set `ADPD_PAIR_RMAX_FACTOR = 2.0` if you want to include all possible pair separations inside a circular aperture.
# 
# The ADPD estimator itself does not apply redshift-space distortions. RSD enter only through the input coordinates. Therefore the `rsd` FLAMINGO samples below use the redshift-space projected coordinates already constructed earlier in this notebook, while `real` and `DM` are real-space projected samples.
# 

# In[26]:


# ---------------------------------------------------------------------------
# ADPD estimator and plotting helpers
# ---------------------------------------------------------------------------
ADPD_ANG_BINS = 720
ADPD_DIST_BINS = 50
ADPD_PAIR_BLOCK_SIZE = 512
ADPD_OVERWRITE_CACHE = False
ADPD_MAX_SLICES = None  # None = all available slices; set to e.g. 5 for a quick test.

# Public-code radial convention:
#   R = max sqrt(x^2 + y^2) for the supplied projected sample.
#   Pair separations are binned over 0 <= r < R.
ADPD_RADIAL_LIMIT_LABEL = "public_0_to_Rmaxpoint"

ADPD_CACHE_DIR = DATA_DIR / "adpd_cache"
ADPD_CACHE_DIR.mkdir(exist_ok=True, parents=True)

import hashlib


def adpd_array_hash(x, y):
    """Return a short hash for the projected point sample."""
    x_arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    y_arr = np.ascontiguousarray(np.asarray(y, dtype=np.float64))

    h = hashlib.blake2b(digest_size=10)
    h.update(np.asarray([x_arr.size], dtype=np.int64).tobytes())
    h.update(x_arr.tobytes())
    h.update(y_arr.tobytes())
    return h.hexdigest()


def adpd_safe_label(label):
    """Return a filesystem-safe label."""
    safe = str(label)
    for old, new in [
        (" ", "_"),
        ("/", "_"),
        ("+", "plus"),
        (",", ""),
        ("(", ""),
        (")", ""),
        ("=", ""),
        (".", "p"),
    ]:
        safe = safe.replace(old, new)
    return safe


def clean_xy_points(x, y):
    """Return finite x, y arrays."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    return x[good], y[good]


def adpd_public_Rmax_from_points(x, y):
    """Return the public-code radial upper limit R = max sqrt(x^2 + y^2)."""
    r_point = np.sqrt(np.asarray(x, dtype=np.float64)**2 + np.asarray(y, dtype=np.float64)**2)
    if r_point.size == 0 or not np.any(np.isfinite(r_point)):
        raise ValueError("Cannot determine ADPD radial upper limit from an empty/non-finite sample.")
    return float(np.nanmax(r_point))


def compute_adpd_xy(
    x,
    y,
    *,
    radius_hmpc=None,
    ang_bins=ADPD_ANG_BINS,
    dist_bins=ADPD_DIST_BINS,
    pair_block_size=ADPD_PAIR_BLOCK_SIZE,
):
    """
    Compute the Angular Distribution of Pairwise Distances for 2D points.

    This follows the public Nature-code radial convention explicitly:
      - R is computed from the input point sample as max sqrt(x^2 + y^2);
      - pair separations are binned over 0 <= r < R;
      - pair orientation angles are folded into [0, 180) deg;
      - p(theta, r) is normalized within each pair-separation bin;
      - sigma_theta^2(r) is the mean squared deviation from 1/N_theta.

    The optional radius_hmpc argument is retained only for metadata; it is not
    used to set the ADPD bin limits.
    """
    x, y = clean_xy_points(x, y)
    n = len(x)
    if n < 2:
        raise ValueError("ADPD requires at least two finite points.")

    r_min_hmpc = 0.0
    r_max_hmpc = adpd_public_Rmax_from_points(x, y)
    if not np.isfinite(r_max_hmpc) or r_max_hmpc <= 0.0:
        raise ValueError("ADPD radial upper limit must be finite and positive.")

    ang_bins = int(ang_bins)
    dist_bins = int(dist_bins)

    r_edges = np.linspace(r_min_hmpc, r_max_hmpc, dist_bins + 1)
    theta_edges = np.linspace(0.0, 180.0, ang_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])

    # Internally store counts as (distance bin, angular bin). This is the
    # transpose of the public script's (angular bin, distance bin) array, but
    # all binning, normalization, and variance operations are equivalent.
    counts = np.zeros((dist_bins, ang_bins), dtype=np.float64)
    block = int(pair_block_size)

    for i0 in range(0, n - 1, block):
        i1 = min(i0 + block, n)
        xi = x[i0:i1]
        yi = y[i0:i1]

        for j0 in range(i0, n, block):
            j1 = min(j0 + block, n)
            xj = x[j0:j1]
            yj = y[j0:j1]

            if j0 == i0:
                ii, jj = np.triu_indices(i1 - i0, k=1)
                if len(ii) == 0:
                    continue
                dx = xi[jj] - xi[ii]
                dy = yi[jj] - yi[ii]
            else:
                dx = (xj[None, :] - xi[:, None]).ravel()
                dy = (yj[None, :] - yi[:, None]).ravel()

            dist = np.sqrt(dx * dx + dy * dy)
            theta = np.mod(np.degrees(np.arctan2(dy, dx)), 180.0)

            hist, _, _ = np.histogram2d(
                dist,
                theta,
                bins=(r_edges, theta_edges),
            )
            counts += hist

    pairs_per_shell = counts.sum(axis=1)
    p_theta_r = np.zeros_like(counts, dtype=np.float64)
    p_err = np.zeros_like(counts, dtype=np.float64)

    nonzero = pairs_per_shell > 0.0
    p_theta_r[nonzero] = counts[nonzero] / pairs_per_shell[nonzero, None]
    p_err[nonzero] = np.sqrt(
        p_theta_r[nonzero] * (1.0 - p_theta_r[nonzero]) / pairs_per_shell[nonzero, None]
    )

    uniform = 1.0 / ang_bins
    sigma_theta2 = np.mean((p_theta_r - uniform)**2, axis=1)

    return {
        "r_edges": r_edges,
        "r_centers": r_centers,
        "theta_edges": theta_edges,
        "theta_centers": theta_centers,
        "p_theta_r": p_theta_r,
        "p_err": p_err,
        "pairs_per_shell": pairs_per_shell,
        "sigma_theta2": sigma_theta2,
        "n_points": n,
        "radius_hmpc": np.nan if radius_hmpc is None else float(radius_hmpc),
        "r_min_hmpc": float(r_min_hmpc),
        "r_max_hmpc": float(r_max_hmpc),
        "rmax_definition": ADPD_RADIAL_LIMIT_LABEL,
        "ang_bins": ang_bins,
        "dist_bins": dist_bins,
    }


def adpd_cache_path(label, x, y, *, radius_hmpc, ang_bins, dist_bins):
    """Return a cache path keyed by sample content and ADPD parameters."""
    x_clean, y_clean = clean_xy_points(x, y)
    sample_hash = adpd_array_hash(x_clean, y_clean)
    safe = adpd_safe_label(label)
    rmax_sample = adpd_public_Rmax_from_points(x_clean, y_clean)
    return ADPD_CACHE_DIR / (
        f"{safe}_N{len(x_clean)}_Rsample{rmax_sample:.3f}_"
        f"ang{int(ang_bins)}_dist{int(dist_bins)}_"
        f"{ADPD_RADIAL_LIMIT_LABEL}_{sample_hash}.npz"
    )


def save_adpd_npz(path, result):
    """Save an ADPD result dictionary."""
    np.savez_compressed(path, **result)


def load_adpd_npz(path):
    """Load an ADPD result dictionary."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def compute_or_load_adpd(label, x, y, *, radius_hmpc):
    """Compute ADPD for one sample, using a content-addressed cache."""
    x_clean, y_clean = clean_xy_points(x, y)
    path = adpd_cache_path(
        label,
        x_clean,
        y_clean,
        radius_hmpc=radius_hmpc,
        ang_bins=ADPD_ANG_BINS,
        dist_bins=ADPD_DIST_BINS,
    )

    if path.exists() and not ADPD_OVERWRITE_CACHE:
        result = load_adpd_npz(path)
        result["cache_file"] = str(path)
        return result

    print(
        f"Computing ADPD: {label}  "
        f"N={len(x_clean):,}  Rmax={adpd_public_Rmax_from_points(x_clean, y_clean):.3f}"
    )
    result = compute_adpd_xy(
        x_clean,
        y_clean,
        radius_hmpc=radius_hmpc,
        ang_bins=ADPD_ANG_BINS,
        dist_bins=ADPD_DIST_BINS,
    )
    save_adpd_npz(path, result)
    result["cache_file"] = str(path)
    return result


def adpd_ensemble_summary(results):
    """Return median and 16--84 percentile summary for ADPD variance curves."""
    if len(results) == 0:
        return None

    r = np.asarray(results[0]["r_centers"], dtype=float)
    arr = np.array([np.asarray(res["sigma_theta2"], dtype=float) for res in results])

    return {
        "r_centers": r,
        "median": np.nanmedian(arr, axis=0),
        "p16": np.nanpercentile(arr, 16, axis=0),
        "p84": np.nanpercentile(arr, 84, axis=0),
        "n": arr.shape[0],
    }


# In[27]:


# ---------------------------------------------------------------------------
# ADPD calculation for DESI, S2, FLAMINGO galaxies, and DM particles
# ---------------------------------------------------------------------------
from joblib import Parallel, delayed
from tqdm.auto import tqdm


def compute_adpd_region(
    region_name,
    *,
    include_s2=False,
    max_slices=ADPD_MAX_SLICES,
    n_jobs=ADPD_N_JOBS,
):
    """
    Compute ADPD results for one comparison region.

    The individual ADPD calculations are submitted as parallel tasks.
    Progress is shown with tqdm.
    The returned dictionary has the same structure as before.
    """
    region = COMPARISON_REGIONS[region_name]
    radius = float(region["radius_hmpc"])

    results = {
        "region_name": region_name,
        "radius_hmpc": radius,
        "DESI": None,
        "S2": None,
        "flamingo": {},
        "DM": [],
    }

    tasks = []

    # DESI.
    desi = desi_regions[region_name]
    tasks.append({
        "kind": "DESI",
        "case": None,
        "islice": None,
        "label": f"DESI_{region_name}",
        "x": desi["x"],
        "y": desi["y"],
        "radius_hmpc": radius,
    })

    # S2, only for the large comparison.
    if include_s2:
        mask_s2_panel = mask_s2_fiducial & np.isfinite(s2_x) & np.isfinite(s2_y)
        tasks.append({
            "kind": "S2",
            "case": None,
            "islice": None,
            "label": "S2_fiducial",
            "x": s2_x[mask_s2_panel],
            "y": s2_y[mask_s2_panel],
            "radius_hmpc": float(S2_FIDUCIAL_RADIUS_HMPC),
        })

    # FLAMINGO galaxy slices.
    for case in CASE_ORDER:
        slices = sorted(
            slices_by_region_case[region_name][case],
            key=lambda s: s["islice"],
        )
        if max_slices is not None:
            slices = slices[:int(max_slices)]

        for s in slices:
            tasks.append({
                "kind": "flamingo",
                "case": case,
                "islice": s["islice"],
                "label": f"FLAMINGO_{region_name}_{case}_slice{s['islice']:03d}",
                "x": s["x"],
                "y": s["y"],
                "radius_hmpc": radius,
            })

    # FLAMINGO DM-particle slices.
    dm_slices = sorted(dm_slices_by_region[region_name], key=lambda s: s["islice"])
    if max_slices is not None:
        dm_slices = dm_slices[:int(max_slices)]

    for s in dm_slices:
        tasks.append({
            "kind": "DM",
            "case": None,
            "islice": s["islice"],
            "label": f"FLAMINGO_DM_{region_name}_slice{s['islice']:03d}",
            "x": s["x"],
            "y": s["y"],
            "radius_hmpc": radius,
        })

    def run_one_adpd_task(task):
        adpd = compute_or_load_adpd(
            task["label"],
            task["x"],
            task["y"],
            radius_hmpc=task["radius_hmpc"],
        )
        return task, adpd

    print(f"Computing ADPD for region '{region_name}'")
    print("----------------------------------------")
    print(f"Number of ADPD tasks : {len(tasks):,}")
    print(f"Parallel jobs        : {n_jobs:,}")

    completed = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        return_as="generator",
        verbose=0,
    )(
        delayed(run_one_adpd_task)(task)
        for task in tasks
    )

    completed = list(
        tqdm(
            completed,
            total=len(tasks),
            desc=f"ADPD {region_name}",
            unit="task",
        )
    )

    # Reassemble the same output structure as the serial version.
    for case in CASE_ORDER:
        results["flamingo"][case] = []

    for task, adpd in completed:
        if task["kind"] == "DESI":
            results["DESI"] = adpd

        elif task["kind"] == "S2":
            results["S2"] = adpd

        elif task["kind"] == "flamingo":
            results["flamingo"][task["case"]].append((task["islice"], adpd))

        elif task["kind"] == "DM":
            results["DM"].append((task["islice"], adpd))

    # Restore deterministic slice ordering.
    for case in CASE_ORDER:
        results["flamingo"][case] = [
            adpd for _, adpd in sorted(results["flamingo"][case], key=lambda item: item[0])
        ]

    results["DM"] = [
        adpd for _, adpd in sorted(results["DM"], key=lambda item: item[0])
    ]

    return results


adpd_results_by_region = {}

adpd_results_by_region["large"] = compute_adpd_region(
    "large",
    include_s2=True,
    max_slices=ADPD_MAX_SLICES,
    n_jobs=ADPD_N_JOBS,
)


# In[28]:


# ---------------------------------------------------------------------------
# ADPD plotting: angular variance and heatmaps
# ---------------------------------------------------------------------------
def plot_adpd_variance_summary(
    region_name,
    *,
    include_s2=False,
    include_rsd=True,
    include_real=True,
    include_DM=True,
):
    """Plot sigma_theta^2(r) for DESI and selected FLAMINGO/DM ensembles."""
    region = COMPARISON_REGIONS[region_name]
    radius = float(region["radius_hmpc"])
    results = adpd_results_by_region[region_name]

    fig, ax = plt.subplots(figsize=(3.4, 2.7), constrained_layout=True)

    # DM first, in grey.
    dm_line = None
    if include_DM and len(results["DM"]) > 0:
        dm_summary = adpd_ensemble_summary(results["DM"])
        positive = (
            np.isfinite(dm_summary["r_centers"])
            & np.isfinite(dm_summary["p16"])
            & np.isfinite(dm_summary["p84"])
            & (dm_summary["p16"] > 0.0)
            & (dm_summary["p84"] > 0.0)
        )
        ax.fill_between(
            dm_summary["r_centers"],
            dm_summary["p16"],
            dm_summary["p84"],
            where=positive,
            color="0.55",
            alpha=0.22,
            linewidth=0,
            zorder=3,
        )
        (dm_line,) = ax.loglog(
            dm_summary["r_centers"],
            dm_summary["median"],
            color="0.40",
            lw=1.8,
            label="DM particles, real-space median",
            zorder=4,
        )

    flamingo_lines = {}
    real_cases = [case for case in CASE_ORDER if _is_real_case(case)]
    rsd_cases = [case for case in CASE_ORDER if _is_rsd_case(case)]

    cases_to_plot = []
    if include_real:
        cases_to_plot.extend(real_cases)
    if include_rsd:
        cases_to_plot.extend(rsd_cases)

    for case in cases_to_plot:
        case_results = results["flamingo"][case]
        summary = adpd_ensemble_summary(case_results)
        color = CASE_COLORS[case]

        if _is_rsd_case(case):
            z_band = 12
            z_line = 13
        else:
            z_band = 8
            z_line = 9

        positive = (
            np.isfinite(summary["r_centers"])
            & np.isfinite(summary["p16"])
            & np.isfinite(summary["p84"])
            & (summary["p16"] > 0.0)
            & (summary["p84"] > 0.0)
        )
        ax.fill_between(
            summary["r_centers"],
            summary["p16"],
            summary["p84"],
            where=positive,
            color=color,
            alpha=0.20,
            linewidth=0,
            zorder=z_band,
        )
        (line_case,) = ax.loglog(
            summary["r_centers"],
            summary["median"],
            color=color,
            lw=2.0,
            label=f"{CASE_LABELS[case]} median",
            zorder=z_line,
        )
        flamingo_lines[case] = line_case

    (desi_line,) = ax.loglog(
        results["DESI"]["r_centers"],
        results["DESI"]["sigma_theta2"],
        color="black",
        lw=2.3,
        ls="--",
        label="DESI",
        zorder=20,
    )

    s2_line = None
    if include_s2 and results.get("S2") is not None:
        (s2_line,) = ax.loglog(
            results["S2"]["r_centers"],
            results["S2"]["sigma_theta2"],
            color="0.2",
            lw=1.8,
            ls="-.",
            label=rf"S2 fiducial, $R={S2_FIDUCIAL_RADIUS_HMPC:g}$",
            zorder=18,
        )

    ordered_handles = [desi_line]
    ordered_labels = ["DESI"]

    if include_rsd:
        for case in rsd_cases:
            ordered_handles.append(flamingo_lines[case])
            ordered_labels.append(f"{CASE_LABELS[case]} median")

    if include_real:
        for case in real_cases:
            ordered_handles.append(flamingo_lines[case])
            ordered_labels.append(f"{CASE_LABELS[case]} median")

    if dm_line is not None:
        ordered_handles.append(dm_line)
        ordered_labels.append("DM particles, real-space median")

    if s2_line is not None:
        ordered_handles.append(s2_line)
        ordered_labels.append(rf"S2 fiducial, $R={S2_FIDUCIAL_RADIUS_HMPC:g}$")

    ax.set_xlabel(r"pair separation $r\ [h^{-1}\,\mathrm{Mpc}]$")
    ax.set_ylabel(r"$\sigma_\theta^2(r)$")
    ax.tick_params(direction="in", top=True, right=True, which="both")
    ax.legend(ordered_handles, ordered_labels, frameon=False, fontsize=7)

    plotted_rmax = [np.nanmax(results["DESI"]["r_centers"])]
    if s2_line is not None:
        plotted_rmax.append(np.nanmax(results["S2"]["r_centers"]))
    for line_case in flamingo_lines.values():
        xdata = np.asarray(line_case.get_xdata(), dtype=float)
        if xdata.size:
            plotted_rmax.append(np.nanmax(xdata))
    if dm_line is not None:
        xdata = np.asarray(dm_line.get_xdata(), dtype=float)
        if xdata.size:
            plotted_rmax.append(np.nanmax(xdata))

    xmax = np.nanmax(plotted_rmax)
    ax.set_xlim(0.02 * xmax, xmax)

    #-----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # Export ADPD curves for external comparison
    # ---------------------------------------------------------------------------
    import os

    os.makedirs("Comparison", exist_ok=True)

    # DESI
    r_desi = np.asarray(results["DESI"]["r_centers"], dtype=float)
    sig_desi = np.asarray(results["DESI"]["sigma_theta2"], dtype=float)

    np.savetxt(
        "Comparison/desi.dat",
        np.column_stack((r_desi, sig_desi)),
        header="r sigma_theta2",
    )

    # S2
    if include_s2 and results.get("S2") is not None:
        r_s2 = np.asarray(results["S2"]["r_centers"], dtype=float)
        sig_s2 = np.asarray(results["S2"]["sigma_theta2"], dtype=float)

        np.savetxt(
            "Comparison/s2_fiducial.dat",
            np.column_stack((r_s2, sig_s2)),
            header="r sigma_theta2",
        )

    # DM particles
    if include_DM and len(results["DM"]) > 0:
        dm_summary = adpd_ensemble_summary(results["DM"])

        r_dm = np.asarray(dm_summary["r_centers"], dtype=float)
        dm_med = np.asarray(dm_summary["median"], dtype=float)

        np.savetxt(
            "Comparison/dm_particles.dat",
            np.column_stack((r_dm, dm_med)),
            header="r sigma_theta2",
        )

    # FLAMINGO ensembles
    for case in cases_to_plot:

        summary = adpd_ensemble_summary(results["flamingo"][case])

        r_case = np.asarray(summary["r_centers"], dtype=float)
        case_med = np.asarray(summary["median"], dtype=float)

        filename = case.lower().replace(" ", "_").replace("+", "plus")

        np.savetxt(
            f"Comparison/{filename}.dat",
            np.column_stack((r_case, case_med)),
            header="r sigma_theta2",
        )
# ---------------------------------------------------------------------------

    print("Reached export point")

    print("Saving Comparison files...")
    import os
    os.makedirs("Comparison", exist_ok=True)

    np.savetxt(
        "Comparison/test.dat",
        np.array([[1.0, 2.0], [3.0, 4.0]])
    )
    print("test.dat written")

    component_suffix = "_".join([
        "rsd" if include_rsd else "no_rsd",
        "real" if include_real else "no_real",
        "DM" if include_DM else "no_DM",
    ])
    s2_suffix = "with_S2" if include_s2 else "no_S2"
    outfile = FIGURES_DIR / (
        f"adpd_variance_DESI_vs_FLAMINGO_{region_name}_{component_suffix}_{s2_suffix}.pdf"
    )
    save_figure(fig, outfile)


def plot_adpd_heatmap(ax, result, *, title, cmap="viridis"):
    """Plot one ADPD p(theta, r) heatmap on an existing axes."""
    p = np.asarray(result["p_theta_r"], dtype=float)
    r_edges = np.asarray(result["r_edges"], dtype=float)
    theta_edges = np.asarray(result["theta_edges"], dtype=float)

    finite_p = p[np.isfinite(p)]
    if finite_p.size > 0:
        vmin = np.nanpercentile(finite_p, 2)
        vmax = np.nanpercentile(finite_p, 98)
    else:
        vmin = None
        vmax = None

    im = ax.imshow(
        p,
        origin="lower",
        aspect="auto",
        extent=(theta_edges[0], theta_edges[-1], r_edges[0], r_edges[-1]),
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(r"$\theta\ [\mathrm{deg}]$")
    ax.set_ylabel(r"$r\ [h^{-1}\,\mathrm{Mpc}]$")
    ax.tick_params(direction="in", top=True, right=True, which="both", labelsize=7)
    return im


def plot_adpd_heatmap_examples(region_name, *, include_s2=False):
    """Plot example ADPD heatmaps for DESI, optional S2, and first FLAMINGO/DM slices."""
    results = adpd_results_by_region[region_name]

    panels = [("DESI", results["DESI"])]
    if include_s2 and results.get("S2") is not None:
        panels.append(("S2", results["S2"]))

    for case in CASE_ORDER:
        if len(results["flamingo"][case]) > 0:
            panels.append((CASE_LABELS[case], results["flamingo"][case][0]))

    if len(results["DM"]) > 0:
        panels.append(("DM particles", results["DM"][0]))

    ncols = len(panels)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(2.2 * ncols, 2.5),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes[0]

    last_im = None
    for ax, (title, result) in zip(axes, panels):
        last_im = plot_adpd_heatmap(ax, result, title=title)

    if last_im is not None:
        fig.colorbar(last_im, ax=axes, shrink=0.85, label=r"$p(\theta,r)$")

    outfile = FIGURES_DIR / f"adpd_heatmap_examples_{region_name}.pdf"
    save_figure(fig, outfile)


# In[29]:


# ---------------------------------------------------------------------------
# ADPD plotting: angular variance and heatmaps
# ---------------------------------------------------------------------------
ADPD_PLOT_RMIN_HMPC = 27.0
ADPD_VARIANCE_YMAX = 4e-8
ADPD_VARIANCE_XLIMS = [30,300]


def _adpd_mask_for_plot(r, y, *, rmin=ADPD_PLOT_RMIN_HMPC):
    """Return a finite plotting mask with r >= rmin and y > 0."""
    r = np.asarray(r, dtype=float)
    y = np.asarray(y, dtype=float)
    return (
        np.isfinite(r)
        & np.isfinite(y)
        & (r >= float(rmin))
        & (y > 0.0)
    )


def _adpd_band_mask_for_plot(r, ylo, yhi, *, rmin=ADPD_PLOT_RMIN_HMPC):
    """Return a finite plotting mask for filled bands with r >= rmin."""
    r = np.asarray(r, dtype=float)
    ylo = np.asarray(ylo, dtype=float)
    yhi = np.asarray(yhi, dtype=float)
    return (
        np.isfinite(r)
        & np.isfinite(ylo)
        & np.isfinite(yhi)
        & (r >= float(rmin))
        & (ylo > 0.0)
        & (yhi > 0.0)
    )


def plot_adpd_variance_summary(
    region_name,
    *,
    include_s2=False,
    include_rsd=True,
    include_real=True,
    include_DM=True,
    rmin_hmpc=ADPD_PLOT_RMIN_HMPC,
):
    print(">>> plot_adpd_variance_summary() is running")
    """Plot sigma_theta^2(r) for DESI and selected FLAMINGO/DM ensembles."""
    region = COMPARISON_REGIONS[region_name]
    radius = float(region["radius_hmpc"])
    results = adpd_results_by_region[region_name]

    fig, ax = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)

    plotted_rmax = []

    # DM first, in grey.
    dm_line = None
    if include_DM and len(results["DM"]) > 0:
        dm_summary = adpd_ensemble_summary(results["DM"])
        r_dm = np.asarray(dm_summary["r_centers"], dtype=float)
        dm_med = np.asarray(dm_summary["median"], dtype=float)
        dm_p16 = np.asarray(dm_summary["p16"], dtype=float)
        dm_p84 = np.asarray(dm_summary["p84"], dtype=float)

        band_mask = _adpd_band_mask_for_plot(r_dm, dm_p16, dm_p84, rmin=rmin_hmpc)
        line_mask = _adpd_mask_for_plot(r_dm, dm_med, rmin=rmin_hmpc)

        ax.fill_between(
            r_dm,
            dm_p16,
            dm_p84,
            where=band_mask,
            color="0.55",
            alpha=0.22,
            linewidth=0,
            zorder=3,
        )

        (dm_line,) = ax.loglog(
            r_dm[line_mask],
            dm_med[line_mask],
            color="0.40",
            lw=1.8,
            label="DM particles, real-space median",
            zorder=4,
        )

        if np.any(line_mask):
            plotted_rmax.append(np.nanmax(r_dm[line_mask]))

    flamingo_lines = {}
    real_cases = [case for case in CASE_ORDER if _is_real_case(case)]
    rsd_cases = [case for case in CASE_ORDER if _is_rsd_case(case)]

    cases_to_plot = []
    if include_real:
        cases_to_plot.extend(real_cases)
    if include_rsd:
        cases_to_plot.extend(rsd_cases)

    for case in cases_to_plot:
        case_results = results["flamingo"][case]
        summary = adpd_ensemble_summary(case_results)
        color = CASE_COLORS[case]

        r_case = np.asarray(summary["r_centers"], dtype=float)
        case_med = np.asarray(summary["median"], dtype=float)
        case_p16 = np.asarray(summary["p16"], dtype=float)
        case_p84 = np.asarray(summary["p84"], dtype=float)

        if _is_rsd_case(case):
            z_band = 12
            z_line = 13
        else:
            z_band = 8
            z_line = 9

        band_mask = _adpd_band_mask_for_plot(r_case, case_p16, case_p84, rmin=rmin_hmpc)
        line_mask = _adpd_mask_for_plot(r_case, case_med, rmin=rmin_hmpc)

        ax.fill_between(
            r_case,
            case_p16,
            case_p84,
            where=band_mask,
            color=color,
            alpha=0.20,
            linewidth=0,
            zorder=z_band,
        )

        (line_case,) = ax.loglog(
            r_case[line_mask],
            case_med[line_mask],
            color=color,
            lw=2.0,
            label=f"{CASE_LABELS[case]} median",
            zorder=z_line,
        )
        flamingo_lines[case] = line_case

        if np.any(line_mask):
            plotted_rmax.append(np.nanmax(r_case[line_mask]))

    r_desi = np.asarray(results["DESI"]["r_centers"], dtype=float)
    sig_desi = np.asarray(results["DESI"]["sigma_theta2"], dtype=float)
    desi_mask = _adpd_mask_for_plot(r_desi, sig_desi, rmin=rmin_hmpc)

    (desi_line,) = ax.loglog(
        r_desi[desi_mask],
        sig_desi[desi_mask],
        color="black",
        lw=2.3,
        ls="--",
        label="DESI",
        zorder=20,
    )

    if np.any(desi_mask):
        plotted_rmax.append(np.nanmax(r_desi[desi_mask]))

    s2_line = None
    if include_s2 and results.get("S2") is not None:
        r_s2 = np.asarray(results["S2"]["r_centers"], dtype=float)
        sig_s2 = np.asarray(results["S2"]["sigma_theta2"], dtype=float)
        s2_mask = _adpd_mask_for_plot(r_s2, sig_s2, rmin=rmin_hmpc)

        (s2_line,) = ax.loglog(
            r_s2[s2_mask],
            sig_s2[s2_mask],
            color="0.2",
            lw=1.8,
            ls="-.",
            label=rf"S2 fiducial, $R={S2_FIDUCIAL_RADIUS_HMPC:g}$",
            zorder=18,
        )

        if np.any(s2_mask):
            plotted_rmax.append(np.nanmax(r_s2[s2_mask]))

    ordered_handles = [desi_line]
    ordered_labels = ["DESI"]

    if include_rsd:
        for case in rsd_cases:
            ordered_handles.append(flamingo_lines[case])
            ordered_labels.append(f"{CASE_LABELS[case]}")

    if include_real:
        for case in real_cases:
            ordered_handles.append(flamingo_lines[case])
            ordered_labels.append(f"{CASE_LABELS[case]}")

    if dm_line is not None:
        ordered_handles.append(dm_line)
        ordered_labels.append("DM particles, real-space")

    if s2_line is not None:
        ordered_handles.append(s2_line)
        ordered_labels.append(rf"S2 fiducial, $R={S2_FIDUCIAL_RADIUS_HMPC:g}$")

    ax.set_xlabel(r"pair separation $r\ [h^{-1}\,\mathrm{Mpc}]$")
    ax.set_ylabel(r"$\sigma_\theta^2(r)$")
    ax.tick_params(direction="in", top=True, right=True, which="both")

    ax.legend(
        ordered_handles,
        ordered_labels,
        frameon=False,
        fontsize=7,
        handlelength=2.5,
        loc="upper right"
    )

    if len(plotted_rmax) > 0:
        xmax = np.nanmax(plotted_rmax)
    else:
        xmax = radius

    ax.set_xlim(ADPD_VARIANCE_XLIMS)
    ax.set_ylim(top=ADPD_VARIANCE_YMAX)


    ax.set_xlim(ADPD_VARIANCE_XLIMS)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:g}"))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda value, pos: f"{value:g}" if value in ADPD_VARIANCE_XLIMS else ""))
    ax.set_ylim(top=ADPD_VARIANCE_YMAX)

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    print("Saving Comparison files...")

    import os
    os.makedirs("Comparison", exist_ok=True)

    # DESI
    np.savetxt(
        "Comparison/desi.dat",
        np.column_stack((r_desi[desi_mask], sig_desi[desi_mask])),
        header="r sigma_theta2",
    )

    # S2
    if s2_line is not None:
        np.savetxt(
            "Comparison/s2_fiducial.dat",
            np.column_stack((r_s2[s2_mask], sig_s2[s2_mask])),
            header="r sigma_theta2",
        )

    # DM
    if dm_line is not None:
        np.savetxt(
            "Comparison/dm_particles.dat",
            np.column_stack((r_dm[line_mask], dm_med[line_mask])),
            header="r sigma_theta2",
        )

    # FLAMINGO
    for case in cases_to_plot:

        summary = adpd_ensemble_summary(results["flamingo"][case])

        r_case = np.asarray(summary["r_centers"], dtype=float)
        med_case = np.asarray(summary["median"], dtype=float)

        mask = _adpd_mask_for_plot(
            r_case,
            med_case,
            rmin=rmin_hmpc,
        )

        filename = case.lower().replace(" ", "_").replace("+", "plus")

        np.savetxt(
            f"Comparison/{filename}.dat",
            np.column_stack((r_case[mask], med_case[mask])),
            header="r sigma_theta2",
        )

    print("Finished writing Comparison files.")
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    component_suffix = "_".join([
        "rsd" if include_rsd else "no_rsd",
        "real" if include_real else "no_real",
        "DM" if include_DM else "no_DM",
    ])
    s2_suffix = "with_S2" if include_s2 else "no_S2"
    outfile = FIGURES_DIR / (
        f"adpd_variance_DESI_vs_FLAMINGO_{region_name}_"
        f"{component_suffix}_{s2_suffix}_rmin{float(rmin_hmpc):.0f}.pdf"
    )
    save_figure(fig, outfile)


def plot_adpd_heatmap(
    ax,
    result,
    *,
    title,
    cmap="viridis",
    rmin_hmpc=ADPD_PLOT_RMIN_HMPC,
    show_ylabel=True,
):
    """Plot one ADPD p(theta, r) heatmap on an existing axes."""
    p = np.asarray(result["p_theta_r"], dtype=float)

    # Display p(theta, r) in units of 10^-3.
    # A plotted value of 1.3 corresponds to p(theta, r) = 1.3e-3.
    p_plot = 1.0e3 * p

    r_edges = np.asarray(result["r_edges"], dtype=float)
    r_centers = np.asarray(result["r_centers"], dtype=float)
    theta_edges = np.asarray(result["theta_edges"], dtype=float)

    display_rows = r_centers >= float(rmin_hmpc)
    finite_p = p_plot[display_rows][np.isfinite(p_plot[display_rows])]

    if finite_p.size > 0:
        vmin = np.nanpercentile(finite_p, 2)
        vmax = np.nanpercentile(finite_p, 98)
    else:
        vmin = None
        vmax = None

    im = ax.imshow(
        p_plot,
        origin="lower",
        aspect="auto",
        extent=(theta_edges[0], theta_edges[-1], r_edges[0], r_edges[-1]),
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )

    ax.set_ylim(float(rmin_hmpc), r_edges[-1])
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(r"$\theta\ [\mathrm{deg}]$")

    if show_ylabel:
        ax.set_ylabel(r"$r\ [h^{-1}\,\mathrm{Mpc}]$")
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)

    ax.tick_params(direction="in", top=True, right=True, which="both", labelsize=7)

    return im


def plot_adpd_heatmap_examples(
    region_name,
    *,
    include_s2=False,
    rmin_hmpc=ADPD_PLOT_RMIN_HMPC,
):
    """
    Plot example ADPD heatmaps.

    Panel order:
      DESI, FLAMINGO + RSD, FLAMINGO real-space, DM particles, S2.
    """
    results = adpd_results_by_region[region_name]

    real_cases = [case for case in CASE_ORDER if _is_real_case(case)]
    rsd_cases = [case for case in CASE_ORDER if _is_rsd_case(case)]

    panels = [("DESI", results["DESI"])]

    for case in rsd_cases:
        if len(results["flamingo"][case]) > 0:
            panels.append((CASE_LABELS[case], results["flamingo"][case][0]))

    for case in real_cases:
        if len(results["flamingo"][case]) > 0:
            panels.append((CASE_LABELS[case], results["flamingo"][case][0]))

    if len(results["DM"]) > 0:
        panels.append(("DM particles", results["DM"][0]))

    if include_s2 and results.get("S2") is not None:
        panels.append(("S2 (fiducial coordinates)", results["S2"]))

    ncols = len(panels)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(1.8 * ncols, 2.),
        constrained_layout=True,
        squeeze=False,
        sharey=True,
    )
    axes = axes[0]

    last_im = None
    for ipanel, (ax, (title, result)) in enumerate(zip(axes, panels)):
        last_im = plot_adpd_heatmap(
            ax,
            result,
            title=title,
            rmin_hmpc=rmin_hmpc,
            show_ylabel=(ipanel == 0),
        )

    if last_im is not None:
        cbar = fig.colorbar(
            last_im,
            ax=axes,
            shrink=0.85,
            pad=0.01,
            label=r"$p(\theta,r)\ [\times 10^{-3}]$",
        )
        cbar.ax.tick_params(labelsize=7)

    outfile = FIGURES_DIR / (
        f"adpd_heatmap_examples_{region_name}_rmin{float(rmin_hmpc):.0f}.pdf"
    )
    save_figure(fig, outfile)


# In[30]:


# ---------------------------------------------------------------------------
# ADPD plots
# ---------------------------------------------------------------------------
plot_adpd_variance_summary(
    "large",
    include_s2=True,
    include_rsd=True,
    include_real=True,
    include_DM=True,
)



# In[31]:


# ---------------------------------------------------------------------------
# ADPD heatmaps: DESI / S2 / FLAMINGO example grid
# ---------------------------------------------------------------------------
def _adpd_heatmap_display_array(result, *, rmin_hmpc=ADPD_PLOT_RMIN_HMPC):
    """Return the heatmap array in units of 10^-3 and a finite mask above rmin."""
    p = np.asarray(result["p_theta_r"], dtype=float)
    p_plot = 1.0e3 * p

    r_centers = np.asarray(result["r_centers"], dtype=float)
    display_rows = r_centers >= float(rmin_hmpc)

    finite = p_plot[display_rows][np.isfinite(p_plot[display_rows])]
    return p_plot, finite


def _plot_adpd_heatmap_panel(
    ax,
    result,
    *,
    vmin,
    vmax,
    rmin_hmpc=ADPD_PLOT_RMIN_HMPC,
    rmax_hmpc=None,
    show_xlabel=False,
    show_ylabel=False,
):
    """Plot one ADPD heatmap panel with shared color scaling."""
    p = np.asarray(result["p_theta_r"], dtype=float)
    p_plot = 1.0e3 * p

    r_edges = np.asarray(result["r_edges"], dtype=float)
    theta_edges = np.asarray(result["theta_edges"], dtype=float)

    im = ax.imshow(
        p_plot,
        origin="lower",
        aspect="auto",
        extent=(theta_edges[0], theta_edges[-1], r_edges[0], r_edges[-1]),
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )

    ax.set_xlim(theta_edges[0], theta_edges[-1])

    if rmax_hmpc is None:
        ax.set_ylim(float(rmin_hmpc), r_edges[-1])
    else:
        ax.set_ylim(float(rmin_hmpc), float(rmax_hmpc))

    if show_xlabel:
        ax.set_xlabel(r"$\theta\ [\mathrm{deg}]$")
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)

    if show_ylabel:
        ax.set_ylabel(r"$r\ [h^{-1}\,\mathrm{Mpc}]$")
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)

    ax.tick_params(direction="in", top=True, right=True, which="both", labelsize=7)

    return im


def plot_adpd_heatmap_examples_grid(
    region_name="large",
    *,
    rmin_hmpc=ADPD_PLOT_RMIN_HMPC,
):
    """
    Plot a 3x6 ADPD heatmap grid.

    Layout:
      Row 1: DESI, then five FLAMINGO + RSD examples
      Row 2: empty, then five FLAMINGO real-space examples
      Row 3: S2, then five FLAMINGO particle examples
    """
    results = adpd_results_by_region[region_name]

    rsd_cases = [case for case in CASE_ORDER if _is_rsd_case(case)]
    real_cases = [case for case in CASE_ORDER if _is_real_case(case)]

    if len(rsd_cases) == 0:
        raise ValueError("No FLAMINGO redshift-space case found in CASE_ORDER.")
    if len(real_cases) == 0:
        raise ValueError("No FLAMINGO real-space case found in CASE_ORDER.")
    if results.get("S2") is None:
        raise ValueError("S2 result is required for this heatmap grid.")

    rsd_case = rsd_cases[0]
    real_case = real_cases[0]

    rsd_results = results["flamingo"][rsd_case][:5]
    real_results = results["flamingo"][real_case][:5]
    dm_results = results["DM"][:5]

    if len(rsd_results) < 5 or len(real_results) < 5 or len(dm_results) < 5:
        raise ValueError("Need at least five RSD, real-space, and DM results.")

    # Collect all displayed panels and compute a common color scale.
    displayed_results = [results["DESI"], results["S2"]]
    displayed_results.extend(rsd_results)
    displayed_results.extend(real_results)
    displayed_results.extend(dm_results)

    finite_all = []
    rmax_values = []
    for res in displayed_results:
        _, finite = _adpd_heatmap_display_array(res, rmin_hmpc=rmin_hmpc)
        if finite.size > 0:
            finite_all.append(finite)

        r_edges = np.asarray(res["r_edges"], dtype=float)
        rmax_values.append(r_edges[-1])

    if len(finite_all) > 0:
        finite_all = np.concatenate(finite_all)
        vmin = np.nanpercentile(finite_all, 2)
        vmax = np.nanpercentile(finite_all, 98)
    else:
        vmin = None
        vmax = None

    rmax_hmpc = np.nanmax(rmax_values)


    fig = plt.figure(
        figsize=(12, 5.5),
        constrained_layout=False,
        dpi=120,
    )

    gs = fig.add_gridspec(
        3,
        7,
        width_ratios=[1.0, 0.18, 1.0, 1.0, 1.0, 1.0, 1.0],
        wspace=0.04,
        hspace=0.04,
        left=0.06,
        right=0.91,
        bottom=0.10,
        top=0.95,
    )

    axes = np.empty((3, 6), dtype=object)

    # First displayed column: DESI/S2.
    for irow in range(3):
        axes[irow, 0] = fig.add_subplot(gs[irow, 0])

    # Remaining displayed columns: FLAMINGO RSD / real-space / particles.
    # These start after the spacer column.
    for irow in range(3):
        for icol in range(1, 6):
            axes[irow, icol] = fig.add_subplot(
                gs[irow, icol + 1],
                sharex=axes[0, 0],
                sharey=axes[0, 0],
            )


    # Left column: DESI, empty, S2.
    im = _plot_adpd_heatmap_panel(
        axes[0, 0],
        results["DESI"],
        vmin=vmin,
        vmax=vmax,
        rmin_hmpc=rmin_hmpc,
        rmax_hmpc=rmax_hmpc,
        show_xlabel=False,
        show_ylabel=True,
    )

    axes[1, 0].set_axis_off()

    _plot_adpd_heatmap_panel(
        axes[2, 0],
        results["S2"],
        vmin=vmin,
        vmax=vmax,
        rmin_hmpc=rmin_hmpc,
        rmax_hmpc=rmax_hmpc,
        show_xlabel=True,
        show_ylabel=True,
    )

    # Top row: five FLAMINGO + RSD examples.
    for icol, res in enumerate(rsd_results, start=1):
        _plot_adpd_heatmap_panel(
            axes[0, icol],
            res,
            vmin=vmin,
            vmax=vmax,
            rmin_hmpc=rmin_hmpc,
            rmax_hmpc=rmax_hmpc,
            show_xlabel=False,
            show_ylabel=False,
        )

    # Middle row: five FLAMINGO real-space examples.
    for icol, res in enumerate(real_results, start=1):
        _plot_adpd_heatmap_panel(
            axes[1, icol],
            res,
            vmin=vmin,
            vmax=vmax,
            rmin_hmpc=rmin_hmpc,
            rmax_hmpc=rmax_hmpc,
            show_xlabel=False,
            show_ylabel=False,
        )

    # Bottom row: five FLAMINGO particle examples.
    for icol, res in enumerate(dm_results, start=1):
        _plot_adpd_heatmap_panel(
            axes[2, icol],
            res,
            vmin=vmin,
            vmax=vmax,
            rmin_hmpc=rmin_hmpc,
            rmax_hmpc=rmax_hmpc,
            show_xlabel=True,
            show_ylabel=False,
        )

    axes[0, 0].text(
        0.5,
        -0.07,
        "DESI DR1\n" + r"$R = 290\ h^{-1}\ \mathrm{Mpc}$",
        transform=axes[0, 0].transAxes,
        ha="center",
        va="top",
        color="black",
        fontsize=8,
        clip_on=False,
    )

    axes[2, 0].text(
        0.5,
        1.05,
        "S2 (fiducial coordinates)",
        transform=axes[2, 0].transAxes,
        ha="center",
        va="bottom",
        color="black",
        fontsize=8,
        clip_on=False,
    )

    axes[0, 5].text(
        1.06,
        0.5,
        "FLAMINGO + RSD",
        transform=axes[0, 5].transAxes,
        rotation=270,
        ha="left",
        va="center",
        color="hotpink",
        clip_on=False,
    )

    axes[1, 5].text(
        1.06,
        0.5,
        "FLAMINGO, real-space",
        transform=axes[1, 5].transAxes,
        rotation=270,
        ha="left",
        va="center",
        color="cornflowerblue",
        clip_on=False,
    )

    axes[2, 5].text(
        1.06,
        0.5,
        "FLAMINGO particles",
        transform=axes[2, 5].transAxes,
        rotation=270,
        ha="left",
        va="center",
        color="0.5",
        clip_on=False,
    )

    cbar = fig.colorbar(
        im,
        ax=axes,
        shrink=0.4,
        pad=0.04,
        label=r"$p(\theta,r)\ [\times 10^{-3}]$",
    )

    cbar.set_ticks([1.3, 1.4, 1.5])
    cbar.set_ticklabels(["1.3", "1.4", "1.5"])
    cbar.ax.tick_params(labelsize=7)

    cbar.ax.tick_params(labelsize=7)

    outfile = FIGURES_DIR / (
        f"adpd_heatmap_grid_3x6_{region_name}_rmin{float(rmin_hmpc):.0f}.pdf"
    )
    save_figure(fig, outfile)


plot_adpd_heatmap_examples_grid(
    "large",
    rmin_hmpc=ADPD_PLOT_RMIN_HMPC,
)


# In[32]:


import numpy as np
import anisotropic_cosmology
from anisotropic_cosmology import anisotropic_distance

alpha = np.pi / 4

d = anisotropic_distance(
    z=0.2,
    alpha=alpha,
    h0=0.7,
    sigma0=-0.01,
    e20=0.0001,
    om_im0=0.31,
    om_k0=1.0e-9,
    wbar=0.0,
)

print(d)


# In[35]:


for name, value in list(globals().items()):
    if value is adpd_results_by_region:
        print(name)


# In[36]:


print(type(adpd_results_by_region))


# In[ ]:




