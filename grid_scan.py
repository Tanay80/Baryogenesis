import itertools
import numpy as np
import likelihood
from concurrent.futures import ProcessPoolExecutor, as_completed

OUTPUT = "posterior_grid.dat"

# ------------------------------------------------------------
# Worker
# ------------------------------------------------------------

def worker(job):
    la, ba, omk = job

    chi2 = likelihood.loglike(
        la_axis=float(la),
        ba_axis=float(ba),
        om_k0=float(omk),
    )

    return la, ba, omk, chi2


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":

    LA_values = np.arange(64.0, 73.0, 1.0)

    BA_values = np.arange(47.0, 54.0, 1.0)

    OMK_values = np.array([
        5.5e-9,
        6.0e-9,
        6.5e-9,
        7.0e-9,
        7.5e-9,
    ])

    grid = list(itertools.product(
        LA_values,
        BA_values,
        OMK_values,
    ))

    print(f"Total jobs : {len(grid)}")
    print("First five jobs:")

    for row in grid[:5]:
        print(row)

    NWORKERS = 8

    with open(OUTPUT, "a") as outfile:

        with ProcessPoolExecutor(max_workers=NWORKERS) as executor:

            futures = {
                executor.submit(worker, job): job
                for job in grid
            }

            completed = 0

            for future in as_completed(futures):

                completed += 1

                la, ba, omk, chi2 = future.result()

                outfile.write(
                    f"{la:.3f} "
                    f"{ba:.3f} "
                    f"{omk:.12e} "
                    f"{chi2:.12f}\n"
                )

                outfile.flush()

                print(
                    f"{completed:3d}/{len(grid)}   "
                    f"LA={la:.2f}   "
                    f"BA={ba:.2f}   "
                    f"OMK={omk:.3e}   "
                    f"Chi2={chi2:.8f}"
                )

    print("\nFinished.")
