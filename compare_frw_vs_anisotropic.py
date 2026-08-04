import numpy as np

# DESI observations
obs = np.loadtxt("../R_Baryogenesis/Nature/fig_4/DESI_Data/adpd_angular_variance.dat")

# Predictions
ani = np.loadtxt("Comparison/desi_anisotropic.dat")
frw = np.loadtxt("Comparison/desi_frw.dat")

# Inverse covariance
Ci = np.load("inverse_covariance.npy")

r_obs = obs[:,0]
y_obs = obs[:,1]

# Interpolate both models onto DESI radii
y_ani = np.interp(r_obs, ani[:,0], ani[:,1])
y_frw = np.interp(r_obs, frw[:,0], frw[:,1])

# Use only radii covered by both models
mask = (
    (r_obs >= max(ani[:,0].min(), frw[:,0].min())) &
    (r_obs <= min(ani[:,0].max(), frw[:,0].max()))
)

diff_ani = y_obs[mask] - y_ani[mask]
diff_frw = y_obs[mask] - y_frw[mask]

Ci_use = Ci[:len(diff_ani), :len(diff_ani)]

chi2_ani = diff_ani @ Ci_use @ diff_ani
chi2_frw = diff_frw @ Ci_use @ diff_frw

print()
print("Anisotropic chi2 =", chi2_ani)
print("FRW chi2         =", chi2_frw)
print("Delta chi2       =", chi2_frw - chi2_ani)
