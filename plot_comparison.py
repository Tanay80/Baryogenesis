import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

obs_file = "../R_Baryogenesis/Nature/fig_4/DESI_Data/adpd_angular_variance.dat"

ani_file = "Comparison/desi_plot_anisotropic.dat"
frw_file = "Comparison/desi_plot_frw.dat"

obs = np.loadtxt(obs_file)
ani = np.loadtxt(ani_file)
frw = np.loadtxt(frw_file)
band = np.loadtxt("scan_curves/one_sigma_band.dat")

r_obs = obs[:,0]
y_obs = obs[:,1]

r_ani = ani[:,0]
y_ani = ani[:,1]

r_frw = frw[:,0]
y_frw = frw[:,1]

mask_ani = (r_obs >= r_ani.min()) & (r_obs <= r_ani.max())
mask_frw = (r_obs >= r_frw.min()) & (r_obs <= r_frw.max())
mask = mask_ani & mask_frw

r = r_obs[mask]
y_obs = y_obs[mask]

y_ani = np.interp(r, r_ani, y_ani)
y_frw = np.interp(r, r_frw, y_frw)

band_lower = np.interp(r, band[:,0], band[:,1])
band_upper = np.interp(r, band[:,0], band[:,2])

text = (
    "Number of data points (N) = 45\n"
    r"$\chi^2_{\rm Red}$ (FRW) = 0.298" "\n\n"
    r"$L_0$: 10.9697 Gpc" "\n"
    r"$\chi^2_{\rm Red}$ (Thurston): 0.250"
)

plt.figure(figsize=(8,6))

# --------------------------------------------------
# Load additional curves
# --------------------------------------------------

dm   = np.loadtxt("Comparison/dm_particles_paper.dat")
real = np.loadtxt("Comparison/real_paper.dat")
rsd  = np.loadtxt("Comparison/rsd_paper.dat")
s2   = np.loadtxt("Comparison/s2_fiducial_paper.dat")

# interpolate to DESI radii
y_dm   = np.interp(r, dm[:,0],   dm[:,1])
y_real = np.interp(r, real[:,0], real[:,1])
y_rsd  = np.interp(r, rsd[:,0],  rsd[:,1])
y_s2   = np.interp(r, s2[:,0],   s2[:,1])

# --------------------------------------------------
# 1 sigma region
# --------------------------------------------------

plt.fill_between(
    r,
    band_lower,
    band_upper,
    facecolor="gold",
    edgecolor="darkorange",
    alpha=0.80,
    linewidth=1.0,
    label=r"Thurston $1\sigma$ region",
    zorder=1,
)

# --------------------------------------------------
# Thurston model
# --------------------------------------------------

plt.plot(
    r,
    y_ani,
    color="firebrick",
    lw=2.8,
    label="Thurston (Nil)",
    zorder=5,
)

# --------------------------------------------------
# Additional theoretical curves
# --------------------------------------------------

plt.plot(
    r,
    y_dm,
    color="forestgreen",
    lw=1.8,
    ls="--",
    label="DM particles, REAL SPACE",
)

plt.plot(
    r,
    y_real,
    color="purple",
    lw=1.8,
    ls="-.",
    label="FLAMINGO, real space",
)

plt.plot(
    r,
    y_rsd,
    color="darkorange",
    lw=1.8,
    ls=":",
    label="FLAMINGO + RSD",
)

plt.plot(
    r,
    y_s2,
    color="teal",
    lw=1.8,
    ls=(0, (5,2)),
    label="S2 fiducial",
)

# --------------------------------------------------
# DESI observations
# --------------------------------------------------

plt.plot(
    r,
    y_obs,
    color="black",
    lw=2,
    label="DESI data",
    zorder=6,
)

plt.grid(alpha=0.3)

plt.xlabel(r"$r\ (h^{-1}\mathrm{Mpc})$", fontsize=13)
plt.ylabel(r"$\sigma_\theta^2$", fontsize=13)

ax = plt.gca()

ax.ticklabel_format(
    axis="y",
    style="sci",
    scilimits=(0,0),
    useMathText=True,
)

plt.text(
    0.985,
    0.98,
    text,
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=11,
    bbox=dict(
        boxstyle="round,pad=0.4",
        facecolor="white",
        edgecolor="black",
        linewidth=1.2,
        alpha=0.80,
    ),
    zorder=100,
)

leg = plt.legend(
    loc="upper left",
    frameon=True,
    fontsize=10,
)

frame = leg.get_frame()
frame.set_facecolor("white")
frame.set_alpha(0.75)
frame.set_edgecolor("black")
frame.set_linewidth(1.2)

plt.tight_layout()

plt.savefig(
    "comparison_anisotropic.png",
    dpi=300,
    bbox_inches="tight",
)
