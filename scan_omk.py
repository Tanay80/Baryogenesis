import os
import shutil
import numpy as np
import likelihood

# -------------------------------------------------
# Fixed preferred axis
# -------------------------------------------------

L_AXIS = 67.65
B_AXIS = 50.60

# -------------------------------------------------
# Curvature values to test
# -------------------------------------------------

omk_values = [
    1.0e-2,
    1.0e-3,
    1.0e-4,
    1.0e-5,
    1.0e-6,
    1.0e-7,
    1.0e-8,
    1.0e-9,
    1.0e-10,
]
# -------------------------------------------------
# Output directory
# -------------------------------------------------

SCAN_DIR = "scan_curves"
os.makedirs(SCAN_DIR, exist_ok=True)

results = []

best_chi2 = np.inf
best_omk = None

print("\nScanning Omega_k0\n")

for omk in omk_values:

    chi2, model = likelihood.loglike(
        L_AXIS,
        B_AXIS,
        omk,
        return_model=True,
    )
    
    print(f"Omega_k0 = {omk:.3e}    chi2 = {chi2:.8f}")

    results.append([omk, chi2])

    # -------------------------------------------------
    # Save anisotropic prediction for this Omega_k
    # -------------------------------------------------

    #src = "Comparison/desi_anisotropic.dat"

    #dst = os.path.join(
        #SCAN_DIR,
        #f"desi_{omk:.3e}.dat"
    #)

    #shutil.copy(src, dst)
    
    dst = os.path.join(
        SCAN_DIR,
        f"desi_{omk:.3e}.dat"
    )

    np.savetxt(
        dst,
        model,
        header="r sigma_theta2"
    )

    print(f"Saved {dst}")

    print(f"Saved {dst}")

    if chi2 < best_chi2:
        best_chi2 = chi2
        best_omk = omk

results = np.array(results)

# -------------------------------------------------
# Save scan table
# -------------------------------------------------

np.savetxt(
    os.path.join(SCAN_DIR, "scan_results.dat"),
    results,
    header="Omega_k0 chi2"
)

print("\n==========================")
print("BEST FIT")
print("==========================")
print("Omega_k0 =", best_omk)
print("chi2 =", best_chi2)
