import subprocess

POINTS = [

    # Best fit
    (67.65, 48.60, 7.696030566272319e-9),

    # LA ±2°
    (65.65, 48.60, 7.696030566272319e-9),
    (69.65, 48.60, 7.696030566272319e-9),

    # BA ±2°
    (67.65, 46.60, 7.696030566272319e-9),
    (67.65, 50.60, 7.696030566272319e-9),

    # OM_K variation
    (67.65, 48.60, 5.0e-9),
    (67.65, 48.60, 1.0e-8),
]

results = []

for la, ba, omk in POINTS:

    subprocess.run(
        [
            "python",
            "fit_sigma.py",
            str(omk),
            str(la),
            str(ba),
        ],
        check=True,
    )

    last = open("fit_log.txt").read().strip().splitlines()[-1]
    chi2 = float(last.split()[-1])

    results.append((chi2, la, ba, omk))

results.sort()

print()
print("========== Ranked Results ==========")

for chi2, la, ba, omk in results:

    print(
        f"Chi2={chi2:.10f}   "
        f"LA={la:7.2f}   "
        f"BA={ba:7.2f}   "
        f"OM_K={omk:.3e}"
    )
