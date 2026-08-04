import re
import subprocess
import os
import shutil
import tempfile
import numpy as np
import sys

SOURCE = "DESI_Anisotropic.py"

# Absolute paths (independent of worker directory)
ROOT = os.path.abspath(os.getcwd())

SOURCE = os.path.join(ROOT, "DESI_Anisotropic.py")
INV_COV = os.path.join(ROOT, "inverse_covariance.npy")


def replace_parameter(text, name, value):
    pattern = rf"^{name}\s*=.*$"
    repl = f"{name} = {value}"
    return re.sub(pattern, repl, text, flags=re.MULTILINE)


def loglike(
    la_axis,
    ba_axis,
    om_k0,
    sigma0=1.0e-10,
    e20=1.0e-10,
    om_im0=0.31,
    return_model=False,
):

    # ---------------------------------------------------------
    # Create isolated working directory
    # ---------------------------------------------------------

    workdir = tempfile.mkdtemp(prefix="adpd_worker_")

    try:

        # Copy DESI script into worker directory

        temp_script = os.path.join(workdir, "DESI_run.py")

        with open(SOURCE, "r") as f:
            txt = f.read()

        txt = replace_parameter(txt, "SIGMA0", sigma0)
        txt = replace_parameter(txt, "E20", e20)
        txt = replace_parameter(txt, "OM_IM0", om_im0)
        txt = replace_parameter(txt, "OM_K0", om_k0)
        txt = replace_parameter(txt, "LA_AXIS", la_axis)
        txt = replace_parameter(txt, "BA_AXIS", ba_axis)

        with open(temp_script, "w") as f:
            f.write(txt)

        # -----------------------------------------------------
        # Run DESI code INSIDE worker directory
        # -----------------------------------------------------

        env = os.environ.copy()
        env["ADPD_OUTPUT_DIR"] = workdir

        # Make original project visible to Python
        env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")

        #print("ROOT =", ROOT)
        #print("PYTHONPATH =", env["PYTHONPATH"])
        
        #print("MARK 1")
        subprocess.run(
            [sys.executable, temp_script],
            cwd=ROOT,
            env=env,
            check=True,
        )

        # -----------------------------------------------------
        # Read worker outputs
        # -----------------------------------------------------
        #print("MARK 2")
        comparison = os.path.join(workdir, "Comparison")
        
        #print("workdir =", workdir)
        #print("comparison =", comparison)

        #print("Comparison exists:", os.path.exists(comparison))
        #raise RuntimeError("LIKELIHOOD REACHED AFTER SUBPROCESS")
        
        #print("MARK 3")
        OBSERVED_ADPD = os.path.abspath(
            os.path.join(
                ROOT,
                "..",
                "R_Baryogenesis",
                "Nature",
                "fig_4",
                "DESI_Data",
                "adpd_angular_variance.dat",
            )
        )
        
        ref = np.loadtxt(OBSERVED_ADPD)

        #print("MARK 4")
        
        #print("Worker Comparison directory:")
        #print(os.listdir(comparison))
        
        model = np.loadtxt(
            os.path.join(comparison, "desi_plot.dat")
        )
        
        r_obs = ref[:,0]
        y_obs = ref[:,1]
        r_model = model[:,0]
        y_model = model[:,1]

        #print("MARK 5")
        #diff = ref[:, 1] - model[:, 1]
        
        # Covariance was constructed after discarding only the first five bins
        r_use = r_obs[5:]
        y_obs_use = y_obs[5:]

        # Interpolate the model onto exactly the observational radii
        y_model_use = np.interp(
            r_use,
            r_model,
            y_model,
        )

        diff = y_obs_use - y_model_use

        Ci = np.load(INV_COV)
        #indices = np.where(mask)[0]
        #Ci = Ci[np.ix_(indices, indices)]

        print("Number of data points N =", len(diff))
        print("Covariance matrix shape =", Ci.shape)
        
        #print("len(r_obs)      =", len(r_obs))
        #print("len(r_model)    =", len(r_model))
        #print("len(mask) true  =", np.sum(mask))
        #print("len(diff)       =", len(diff))
        #print("Ci.shape        =", Ci.shape)
        #print("r_use first/last:", r_use[0], r_use[-1])
        #print("r_model first/last:", r_model[0], r_model[-1])
        chi2 = diff @ Ci @ diff
        #print("MARK 6", chi2)
        #print(">>> likelihood chi2 =", chi2)

    finally:

        # -----------------------------------------------------
        # Always clean up worker directory
        # -----------------------------------------------------

        shutil.rmtree(workdir, ignore_errors=True)

    if return_model:
        return chi2, model.copy()
    else:
        return chi2
