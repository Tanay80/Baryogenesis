from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
import itertools

# -----------------------------
# Parameter ranges
# -----------------------------

OMK = [1e-9]

LA = [
    240,
    260,
]

BA = [
    35,
    50,
]

POINTS = list(itertools.product(OMK, LA, BA))


def run_one(point):
    omk, la, ba = point

    cmd = [
        "python",
        "fit_sigma.py",
        str(omk),
        str(la),
        str(ba),
    ]

    subprocess.run(
        cmd,
        check=True,
    )

    return point


def main():

    print(f"Total evaluations : {len(POINTS)}")
    print("Parallel workers  : 4")
    print()

    finished = 0

    with ProcessPoolExecutor(max_workers=4) as pool:

        futures = [pool.submit(run_one, p) for p in POINTS]

        for future in as_completed(futures):

            finished += 1
            omk, la, ba = future.result()

            print(
                f"[{finished:3d}/{len(POINTS)}] "
                f"OM_K={omk:.2e} "
                f"LA={la:6.1f} "
                f"BA={ba:6.1f}"
            )

    print("\nFinished.")


if __name__ == "__main__":
    main()
