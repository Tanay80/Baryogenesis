import re
import subprocess
import time
import numpy as np
import sys

SOURCE = "DESI_Anisotropic.py"
#TEMP   = "DESI_run.py"

import os
TEMP = f"DESI_run_{os.getpid()}.py"

def replace_parameter(text, name, value):
    pattern = rf"^{name}\s*=.*$"
    repl = f"{name} = {value}"
    return re.sub(pattern, repl, text, flags=re.MULTILINE)

def make_model(
    sigma,
    e2,
    om_im,
    om_k=1.0e-9,
    la=264.0,
    ba=48.0,
):

    with open(SOURCE, "r") as f:
        txt = f.read()

    txt = replace_parameter(txt, "SIGMA0", sigma)
    txt = replace_parameter(txt, "E20", e2)
    txt = replace_parameter(txt, "OM_IM0", om_im)
    txt = replace_parameter(txt, "OM_K0", om_k)
    txt = replace_parameter(txt, "LA_AXIS", la)
    txt = replace_parameter(txt, "BA_AXIS", ba)

    with open(TEMP, "w") as f:
        f.write(txt)

    print("Temporary model written.")
    
# ------------------------------------------------------------------
# Choose model parameters
# ------------------------------------------------------------------

om_k0 = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0e-9
la_axis = float(sys.argv[2]) if len(sys.argv) > 2 else 264.0
ba_axis = float(sys.argv[3]) if len(sys.argv) > 3 else 48.0

make_model(
    sigma=1.0e-10,
    e2=1.0e-10,
    om_im=0.31,
    om_k=om_k0,
    la=la_axis,
    ba=ba_axis,
)

# ------------------------------------------------------------------
# Run DESI pipeline
# ------------------------------------------------------------------

t0 = time.time()

print("Running DESI model...")

subprocess.run(
    ["python", TEMP],
    check=True,
)

# ------------------------------------------------------------------
# Compute chi-square
# ------------------------------------------------------------------

# Load curves
ref = np.loadtxt("Comparison/desi_fiducial.dat")
model = np.loadtxt("Comparison/desi.dat")

# Data vectors
d_ref = ref[:,1]
d_model = model[:,1]

# Residual
diff = d_ref - d_model

# Load inverse covariance
Ci = np.load("inverse_covariance.npy")

# Covariance-weighted chi-square
chi2 = diff @ Ci @ diff

print()
print("--------------------------------")
print(f"LA_AXIS = {la_axis}")
print(f"BA_AXIS = {ba_axis}")
print(f"OM_K0   = {om_k0}")
print(f"E20     = {1.0e-10}")
print(f"SIGMA0  = {1.0e-10}")
print(f"OM_IM0  = {0.31}")
print("--------------------------------")
print(f"Chi2 = {chi2:.10f}")
print(f"Finished in {time.time()-t0:.1f} s")

if not os.path.exists("fit_log.txt"):
    with open("fit_log.txt", "w") as f:
        f.write("# LA BA SIGMA0 E20 OM_IM0 OM_K0 CHI2\n")

with open("fit_log.txt", "a") as f:
    f.write(
        f"{la_axis:10.4f} "
        f"{ba_axis:10.4f} "
        f"{1.0e-10:14.8e} "
        f"{1.0e-10:14.8e} "
        f"{0.31:10.6f} "
        f"{om_k0:14.8e} "
        f"{chi2:.12f}\n"
    )
    
try:
    os.remove(TEMP)
except FileNotFoundError:
    pass
