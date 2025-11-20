import random_matrix.amplitude_matrix.isotropic_tmatrix as isotropic_tmatrix
from scipy.special import hankel1, spherical_jn, lpmv, h1vp, spherical_yn
import numpy as np
import math
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import pickle

# Creates the Convex hull for a sphere of size 600nm
wavelength1 = 400e-9
k1 = (2 * np.pi) / wavelength1
m = 1.2  # Relative refractive Index
k2 = m * k1
x = 2
Ts_zstretch = []
Ts_ystretch = []
Ts_xstretch = []
r = x / k1
n_max = int(np.max(np.floor(x + 4.05 * x**0.33333 + 2.0) + 1))
size = np.arange(1, 5.5, 0.5)

# for idx, factor in enumerate(size):
#     y = factor*x
#     n_max = int(np.max(np.floor(y + 4.05 * y**0.33333 + 2.0) + 1))
#     # Semi-axes lengths
#     lx,ly,lz= r,r,factor*r

#     # Use the same plotting function defined before
#     hull_ellipse = isotropic_tmatrix.ellipse_hull(lx, ly, lz)
#     T_ellipse = isotropic_tmatrix.get_T(hull_ellipse, k1, k2, n_max)
#     Ts_zstretch.append(T_ellipse)

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time


def compute_T_matrix(params):
    """Compute T matrix for a given configuration"""
    factor, r, x, k1, k2, stretch_type = params

    # Semi-axes based on stretch type
    if stretch_type == "y":
        lx, ly, lz = r, factor * r, r
    elif stretch_type == "x":
        lx, ly, lz = factor * r, r, r
    else:  # z
        lx, ly, lz = r, r, factor * r

    y = factor * x
    n_max = 2  # int(np.max(np.floor(y + 4.05 * y**0.33333 + 2.0) + 1))

    hull_ellipse = isotropic_tmatrix.ellipse_hull(lx, ly, lz)
    T_ellipse = isotropic_tmatrix.get_T(hull_ellipse, k1, k2, n_max)

    return factor, stretch_type, T_ellipse


# Prepare all tasks
tasks = []
for factor in size:
    tasks.append((factor, r, x, k1, k2, "y"))
    tasks.append((factor, r, x, k1, k2, "x"))

# Run in parallel with progress bar
start_time = time.time()
completed = 0

with ProcessPoolExecutor() as executor:
    futures = {executor.submit(compute_T_matrix, task): task for task in tasks}

    with tqdm(
        total=len(tasks),
        desc="Computing T matrices",
        
    ) as pbar:
        for future in as_completed(futures):
            factor, stretch_type, T_ellipse = future.result()

            if stretch_type == "y":
                Ts_ystretch.append((factor, T_ellipse))
            elif stretch_type == "x":
                Ts_xstretch.append((factor, T_ellipse))


            pbar.set_postfix(
                {
                    "AR": f"{factor:.1f}",
                    "Type": stretch_type,
                   
                }
            )
            pbar.update(1)  # Sort results by factor to maintain order
Ts_ystretch = [T for _, T in sorted(Ts_ystretch, key=lambda x: x[0])]
Ts_xstretch = [T for _, T in sorted(Ts_xstretch, key=lambda x: x[0])]

output_file = "Ts_full.pkl"
results = {
    "Ts_xstretch": Ts_xstretch,
    "Ts_ystretch": Ts_ystretch,
    "stretch_factor": size,
}

with open(output_file, "wb") as f:
    pickle.dump(results, f)
