import numpy as np
import glob

# -------------------------------------------------------
# Load chi2 scan
# -------------------------------------------------------

scan = np.loadtxt("scan_curves/scan_results.dat")

chi2_min = scan[:,1].min()

accepted = scan[scan[:,1] <= chi2_min + 1.0]

print("Minimum chi2 =", chi2_min)
print("Accepted models =", len(accepted))

# -------------------------------------------------------
# Read accepted curves
# -------------------------------------------------------

curves = []

for omk, chi2 in accepted:

    fname = f"scan_curves/desi_{omk:.3e}.dat"

    data = np.loadtxt(fname)

    curves.append(data[:,1])

curves = np.array(curves)

r = np.loadtxt(
    f"scan_curves/desi_{accepted[0,0]:.3e}.dat"
)[:,0]

lower = curves.min(axis=0)

upper = curves.max(axis=0)

np.savetxt(
    "scan_curves/one_sigma_band.dat",
    np.column_stack((r, lower, upper)),
    header="r lower upper"
)

print("Saved scan_curves/one_sigma_band.dat")
