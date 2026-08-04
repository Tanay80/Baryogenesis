import numpy as np

# DESI curve
desi = np.loadtxt("Comparison/desi.dat")
model = desi[:,1]

# Public-code covariance
mean = np.load("/home/tanay/Documents/Research/R_Baryogenesis/Nature/fig_4/Covariance/mean_vector.npy")

Cinverse = np.load(
"/home/tanay/Documents/Research/R_Baryogenesis/Nature/fig_4/Covariance/inverse_covariance.npy"
)

delta = model - mean

chi2 = delta @ Cinverse @ delta

print(f"Chi2 = {chi2:.8f}")
print(f"Reduced = {chi2/(len(delta)-1):.8f}")
