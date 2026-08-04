import subprocess
from scipy.optimize import minimize_scalar

LA = 67.65
BA = 48.60

CACHE = {}

def chi2(omk):

    omk = float(omk)

    key = round(omk, 15)
    if key in CACHE:
        return CACHE[key]

    subprocess.run(
        [
            "python",
            "fit_sigma.py",
            str(omk),
            str(LA),
            str(BA),
        ],
        check=True,
    )

    last = open("fit_log.txt").read().strip().splitlines()[-1]
    value = float(last.split()[-1])

    CACHE[key] = value

    print()
    print("----------------------------")
    print(f"OM_K0 = {omk:.3e}")
    print(f"Chi2  = {value:.10f}")
    print("----------------------------")

    return value


result = minimize_scalar(
    chi2,
    bounds=(1e-10, 1e-8),
    method="bounded",
    options={
        "maxiter": 10,
        "xatol": 5e-11,
    },
)

print(result)
