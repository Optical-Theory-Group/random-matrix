# Random Matrix Simulation Library

![Random grid](/assets/images/random_grid.png "Random grid")

A Python library for performing random matrix simulations of the scattering of polarized light in random media. The library models how light propagates through a disordered scattering medium by statistically characterizing the random scattering (S) matrix and generating ensembles of S matrices consistent with those statistics.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Repository Structure](#repository-structure)
4. [Package Modules](#package-modules)
   - [amplitude\_matrix](#amplitude_matrix)
   - [modes](#modes)
   - [input\_statistics](#input_statistics)
   - [scattering\_matrix](#scattering_matrix)
   - [utils](#utils)
   - [config](#config)
5. [How to Use](#how-to-use)
   - [Step 1: Define the Mode Grid](#step-1-define-the-mode-grid)
   - [Step 2: Choose a Particle Scattering Model](#step-2-choose-a-particle-scattering-model)
   - [Step 3: Define Particle Statistics](#step-3-define-particle-statistics)
   - [Step 4: Set Up Medium Parameters](#step-4-set-up-medium-parameters)
   - [Step 5: Configure Integration](#step-5-configure-integration)
   - [Step 6: Run the Statistics Manager](#step-6-run-the-statistics-manager)
   - [Step 7: Sample Random S Matrices](#step-7-sample-random-s-matrices)
   - [Step 8: Analyze Results](#step-8-analyze-results)
   - [Complete Example Script](#complete-example-script)
6. [Examples](#examples)
7. [Requirements and Dependencies](#requirements-and-dependencies)

---

## Overview

The random matrix method provides a statistical approach to modeling light scattering in complex media. Instead of solving Maxwell's equations for every possible configuration of scatterers, this library:

1. **Partitions k-space** into discrete regions called **modes** (bundles of wavevectors).
2. **Characterizes single-particle scattering** via the **A matrix** (amplitude matrix), which maps an incident polarization state to a scattered polarization state for a given pair of wavevectors.
3. **Averages over particle ensembles** using probability density functions to compute the **mean** and **covariance** of the full scattering (S) matrix.
4. **Generates random samples** of the S matrix consistent with these statistics, which can then be used in further transport calculations.

The S matrix is a block matrix of the form:

```
S = [ r   t2 ]
    [ t   r2 ]
```

where `r` and `t` are reflection and transmission matrices for light incident from the left, and `r2` and `t2` are the corresponding matrices for light incident from the right. Each block contains 2×2 polarization sub-blocks (s and p polarizations) for every pair of modes.

---

## Installation

A conda environment called `random_matrix` containing all necessary packages can be installed using the environment file located at `conda/environment.yml`. From the base repository directory:

```bash
conda env create -f conda/environment.yml
conda activate random_matrix
```

---

## Repository Structure

```
random-matrix/
├── random_matrix/                    # Core library
│   ├── amplitude_matrix/             # Single-particle A matrix calculations
│   ├── modes/                        # Mode and ModeGrid definitions
│   ├── input_statistics/             # Statistical calculations
│   ├── scattering_matrix/            # S matrix operations and sampling
│   ├── utils/                        # Utility functions
│   └── config/                       # Configuration management
├── examples/                         # Example scripts
├── assets/                           # Images and other assets
├── data/                             # Simulation data directory
├── conda/                            # Conda environment files
└── README.md                         # Original README
```

---

## Package Modules

### amplitude_matrix

Computes the single-particle **A matrix** — a 2×2 complex matrix that describes how a plane wave with wavevector **k₁** and polarization state is scattered by a single particle into a plane wave with wavevector **k₂**.

Each file provides `get_A()`, `get_A_product()`, and `get_A_product_conj()` functions, which are used to compute `<A>`, `<AA>`, and `<AA*>` respectively when averaging over particle parameters.

| File | Description |
|------|-------------|
| `scattering_geometry.py` | Basis vector transformations. Computes the spherical polar basis vectors (e_theta, e_phi) for incident and scattered wavevectors, constructs rotation matrices between the wave-based and scattering-plane coordinate systems, and handles special cases (parallel/anti-parallel wavevectors). Supports both Bohren-Huffman (BH) sign conventions. |
| `isotropic_sphere.py` | Mie theory for **isotropic spheres**. Computes the A matrix using the standard Mie solution with Wiscombe's criterion for series truncation. Includes functions for 2D (2×2 in the scattering plane) and 3D (3×3 Cartesian) amplitude matrices. |
| `chiral_sphere.py` | Mie theory for **chiral spheres**. Extends the isotropic sphere case to particles with circular birefringence (different refractive indices for left- and right-circular polarizations, parameterized by `brg`). |
| `isotropic_tmatrix.py` | **T-matrix method** for arbitrary-shaped isotropic particles. Computes the T-matrix from surface integrals over the particle's convex hull using Vector Spherical Wave Functions (VSWFs), then derives the A matrix from the T-matrix. Suitable for non-spherical particles. |
| `anisotropic_tmatrix.py` | **T-matrix method** for general **anisotropic particles**. The most general particle model. Handles particles with tensorial permittivity via quasi-VSWF expansions, Wigner D-matrix rotations, and optional rotational averaging over Euler angles. Contains the `AmatrixGenerator` class with extensive functionality for computing T-matrices, rotating them, and performing orientational averaging. |
| `custom_rayleigh.py` | A **custom/synthetic** scatterer that uses a Henyey-Greenstein phase function normalized to match the Mie scattering cross-section. Useful for simplified scattering models where only the angular distribution of scattered light matters. |

**Key function signatures:**

```python
# Isotropic sphere (and similar for other particle types)
def get_A(ki_x, ki_y, ki_z, kj_x, kj_y, kj_z, x, m) -> np.ndarray
# Returns: array of shape (N, 4) with [S2, S3, S4, S1]

def get_A_product_conj(ki_x, ..., kv_x, kv_y, kv_z, x, m) -> np.ndarray
# Returns: array of shape (N, 16) for <AA*> computation

def get_A_product(ki_x, ..., kv_x, kv_y, kv_z, x, m) -> np.ndarray
# Returns: array of shape (N, 16) for <AA> computation (pseudo-covariance)
```

Each `get_A` function has a `.particle_type` attribute (e.g., `"sphere"`, `"chiral"`) used to identify the particle model.

---

### modes

Defines **modes** — non-overlapping regions in the (k_x, k_y) plane that partition k-space. Each mode represents a bundle of wavevectors with similar propagation characteristics.

| File | Description |
|------|-------------|
| `mode.py` | **Mode class.** A dataclass representing a single mode. Each mode is defined by its vertices (boundary points in k-space) and its sides (either straight lines or circular arcs). Computes properties like `wave_type` ("propagating" for \|k\|<1, "evanescent" for \|k\|>1), `weight` (area), `is_central` (centro-symmetric about origin), and `is_edge` (touches the unit circle). |
| `mode_grid.py` | **ModeGrid class.** A container for a collection of Mode objects. Manages propagating and evanescent modes separately, handles reciprocity (pairing modes with opposite k-vectors), and provides access by index or by (k_x, k_y) point. Stores the reciprocity matrix used for S-matrix symmetries. |
| `mode_grid_factory.py` | **Factory functions** for constructing ModeGrid objects. Supports multiple grid types: **polar grids** (`from_dr_dt`, `from_rt_vals`), **periodic tilings** (`from_tiling` with triangles, rectangles, or hexagons), and **random grids** (`from_random` with Delaunay triangulation or Voronoi tessellation). Handles the complex task of cutting modes across circular boundaries (k=1 and k=r_lim). |

**ModeGrid construction examples:**

```python
from random_matrix.modes import mode_grid_factory

# Hexagonal tiling grid
grid = mode_grid_factory.from_tiling("hexagons", side_length=0.07, r_lim=1.2)

# Polar grid with radial spacing dr and angular spacing dt
grid = mode_grid_factory.from_dr_dt(dr=0.05, dt=2*np.pi/40, r_lim=1.2)

# Random Delaunay grid
grid = mode_grid_factory.from_random(num_points=500, r_lim=1.2, random_type="delaunay")
```

---

### input_statistics

The computational core of the library. Computes the **mean** and **covariance** statistics of the scattering matrix by averaging single-particle A matrices over particle parameter distributions and integrating over mode volumes in k-space.

| File | Description |
|------|-------------|
| `density_function.py` | **Probability density functions.** Provides `DensityFunction`, `DensityFunctionTerm`, `DeltaDensityFactor`, and `RegularDensityFactor` classes for defining complex PDFs over particle parameters (e.g., size parameter x, refractive index m). Supports mixtures of Dirac delta distributions and continuous functions. |
| `density_integrals.py` | **Integration against PDFs.** Contains `integrate_by_density()` which integrates a function with respect to a `DensityFunction`, producing a new function with the integrated variables removed. |
| `medium_parameters.py` | **Background medium properties.** A dataclass storing the `wavelength`, `number_density` (n), and `slab_thickness` (L) of the scattering medium. Computes derived quantities like k = 2π/λ and constant prefactors for the mean and covariance. |
| `medium_statistics.py` | **Particle statistics.** `ParticleStatistics` extends `DensityFunction` and associates an A-matrix function with a particle parameter distribution and mixing ratio. `MediumStatistics` combines multiple `ParticleStatistics` (for different particle types) and computes the ensemble-averaged `<A>`, `<AA*>`, and `<AA>` functions. |
| `index_finder.py` | **Non-zero statistics finder.** The `IndexFinder` class determines which elements of the scattering matrix have non-zero statistics based on mode grid symmetries (e.g., reciprocity, lattice symmetry). This dramatically reduces computational cost. |
| `integration_task.py` | **Integration engine.** `IntegrationTaskPreparer` manages the numerical integration over mode volumes in k-space. Supports multiple integration methods ("cubature", "midpoint", "lattice") and handles both mean and covariance integrals over quadruples of modes (i,j,u,v). |
| `shape_classifier.py` | **Mode shape classification.** Groups modes into equivalence classes based on their geometric shapes to avoid redundant integrations. Uses side lengths and exterior angles to identify congruent mode polygons. |
| `input_statistics_manager.py` | **Main orchestrator.** `InputStatisticsManager` ties everything together: it coordinates the calculation of volumes, indices, mean S matrix, A-matrix values, Cholesky decompositions of covariance matrices, and provides the `get_matrix_pool_manager()` method as the primary entry point. Supports checkpointing — partial results are saved to disk and computations resume from where they left off. |
| `matrix_pool_manager.py` | **Random matrix pool.** `MatrixPoolManager` generates ensembles of random scattering matrices from the pre-computed mean and Cholesky decomposition. Provides methods for sampling single or multiple pools of S matrices and computing the corresponding transfer (M) matrices. |
| `paths.py` | **File path management.** Defines `InputStatisticsPaths` and `MatrixPoolsPaths` dataclasses that organize all file paths for a simulation, ensuring consistent organization of saved data. |
| `input_statistics_logger.py` | **Logging and timing.** Provides logging classes that track computation progress and timing for the statistics pipeline. |

---

### scattering_matrix

Operations on scattering matrices and field vectors, including random sampling and visualization.

| File | Description |
|------|-------------|
| `scattering_matrix.py` | **ScatteringMatrix class.** Provides methods for extracting sub-blocks from the full S matrix: `get_ss()`, `get_sp()`, `get_ps()`, `get_pp()` (polarization components), `get_block()` (r, t, t2, r2 blocks), and `get_column_intensity()` (intensity scattered into a specific output mode). |
| `sampler.py` | **S matrix sampler.** `S_sampler()` generates random scattering matrices from a mean matrix and its Cholesky decomposition. Handles the complex reordering of elements to map the block-structured S matrix onto the flat covariance representation, and optionally symmetrizes the result to ensure unitarity. |
| `field_vector.py` | **Field vector representation.** `FieldVector` class for working with the electric field vector in the mode basis. Provides methods to extract s and p polarization components, create Jones vector dictionaries keyed by mode index, and visualize field distributions using polygon-fill or imshow plots. |

---

### utils

A collection of general-purpose utility modules used throughout the library.

| File | Description |
|------|-------------|
| `array_utils.py` | Array manipulation utilities: duplicate removal, equality checking, point ordering, pair generation, list splitting, sorting by reference, and module detection (NumPy vs CuPy). |
| `function_utils.py` | Mathematical function manipulation: extracting function variables/signatures, multiplying functions by constants, adding functions, and composing functions. |
| `geometry_utils.py` | Geometric operations: convex polygon areas, polygon-circle intersection, polar/Cartesian coordinate conversion, point rotation/translation, and edge area calculations for modes bounded by circular arcs. |
| `integration_utils.py` | Numerical integration helpers: domain extraction from dictionaries, product integration over hypercubes, and scheme selection for multi-dimensional quadrature. |
| `matrix_utils.py` | Scattering matrix operations: block extraction, sub-block indexing, reciprocity transformations (`r_sym`), Cholesky decomposition, and closest-unitary approximation via SVD. |
| `memoize.py` | Function memoization decorator for caching expensive function evaluations. |
| `plotting_utils.py` | Matplotlib plotting helpers: drawing circles, lines, rays, and polygons. |
| `special_functions.py` | Special mathematical function definitions used in scattering calculations. |
| `types.py` | Reusable custom Python type aliases (`Numeric`, `MathematicalFunction`, etc.). |
| `system_utils.py` | System-level utilities (e.g., memory usage tracking). |

---

### config

Configuration management for the library.

| File | Description |
|------|-------------|
| `config.py` | Functions for managing persistent configuration via a JSON file: `load_config()`, `save_config()`, `get_data_directory()`, `set_data_directory()`. |
| `tools.py` | (Standalone script) CLI utility for viewing and setting the data directory. |

---

## How to Use

This section walks through a typical simulation workflow, from defining the mode grid to sampling random scattering matrices.

### Step 1: Define the Mode Grid

First, partition k-space into modes. The choice of grid affects both accuracy and computational cost.

```python
from random_matrix.modes import mode_grid_factory

# Option A: Periodic tiling (good for lattice symmetries)
mode_grid = mode_grid_factory.from_tiling(
    tiling_type="hexagons",   # or "triangles", "rectangles"
    side_length=0.07,
    r_lim=1.2,                # radial extent of the grid
)

# Option B: Polar grid (good for rotationally symmetric problems)
mode_grid = mode_grid_factory.from_dr_dt(
    dr=0.05,
    dt=2 * np.pi / 40,
    r_lim=1.2,
    include_central_mode=True,
)

# Option C: Random grid (for Monte Carlo style approaches)
mode_grid = mode_grid_factory.from_random(
    num_points=500,
    r_lim=1.2,
    random_type="delaunay",
)
```

> **Key concepts:**
> - Modes with \|k\| < 1 are **propagating** (inside the unit circle).
> - Modes with \|k\| > 1 are **evanescent** (outside the unit circle, up to `r_lim`).
> - A **reciprocal** grid has every mode paired with its opposite (−k_x, −k_y).

### Step 2: Choose a Particle Scattering Model

Select the A-matrix function for your scatterers. The library provides several options:

```python
# Isotropic sphere (Mie theory) — most common
from random_matrix.amplitude_matrix import isotropic_sphere
a_matrix_func = isotropic_sphere.get_A
a_product_func = isotropic_sphere.get_A_product
a_product_conj_func = isotropic_sphere.get_A_product_conj

# For chiral spheres:
# from random_matrix.amplitude_matrix import chiral_sphere
# For T-matrix (non-spherical particles):
# from random_matrix.amplitude_matrix import isotropic_tmatrix
# For anisotropic T-matrix particles:
# from random_matrix.amplitude_matrix.anisotropic_tmatrix import AmatrixGenerator
```

The A matrix function takes wavevector components and particle parameters. For an isotropic sphere, the function signature is:

```python
get_A(ki_x, ki_y, ki_z, kj_x, kj_y, kj_z, x, m) -> np.ndarray
```
where `x` is the size parameter (k × radius) and `m` is the relative refractive index.

### Step 3: Define Particle Statistics

Define the probability distribution over particle parameters using the density function framework:

```python
from random_matrix.input_statistics.density_function import (
    DensityFunction, DensityFunctionTerm, DeltaDensityFactor, RegularDensityFactor
)
from random_matrix.input_statistics.medium_statistics import (
    ParticleStatistics, MediumStatistics
)

# Example: A mixture of two particle populations

# Population 1 (70%): fixed-size spheres (delta distribution)
term1 = DensityFunctionTerm.from_delta({"x": 2.0, "m": 1.2 + 0.01j}, factor=0.7)

# Population 2 (30%): polydisperse spheres with a size distribution
def size_pdf(x):
    return 2 * x * np.exp(-x**2)  # example distribution

term2 = DensityFunctionTerm.from_regular(size_pdf, {"x": [0.5, 5.0]}, factor=0.3)

# Build particle statistics
particle_stats = ParticleStatistics(
    terms=[term1, term2],
    a_matrix=a_matrix_func,
    a_product=a_product_func,
    a_product_conj=a_product_conj_func,
    mixing_ratio=1.0,
)

# Combine into medium statistics
medium_stats = MediumStatistics(particle_terms=[particle_stats])
```

### Step 4: Set Up Medium Parameters

Define the properties of the background medium:

```python
from random_matrix.input_statistics.medium_parameters import MediumParameters

medium_params = MediumParameters(
    wavelength=550e-9,        # 550 nm (green light)
    number_density=1e18,      # particles per m³
    slab_thickness=1e-6,      # 1 µm slab
)
```

### Step 5: Configure Integration

Set up the integration method and precision:

```python
from random_matrix.input_statistics.integration_task import IntegrationTaskConfig

integration_config = IntegrationTaskConfig(
    integration_method="cubature",  # or "midpoint", "lattice"
    integration_accuracy=3,        # higher = more accurate but slower
)
```

> **Integration methods:**
> - `"cubature"`: High-accuracy adaptive quadrature. Best for general use.
> - `"midpoint"`: Simple midpoint rule. Faster but less accurate.
> - `"lattice"`: Exploits lattice symmetry for periodic grids. Fastest for tiling grids.

### Step 6: Run the Statistics Manager

This is the main computation step. The manager orchestrates all calculations, saves intermediate results, and supports resumption if interrupted:

```python
from random_matrix.input_statistics.input_statistics_manager import InputStatisticsManager

simulation_name = "my_simulation"
base_path = "./data"  # where results will be saved

# Create the manager (or load an existing simulation)
try:
    ism = InputStatisticsManager.from_name(simulation_name, base_path)
    print("Loaded existing simulation.")
except FileNotFoundError:
    ism = InputStatisticsManager(
        simulation_name=simulation_name,
        medium_parameters=medium_params,
        medium_statistics=medium_stats,
        mode_grid=mode_grid,
        integration_task_config=integration_config,
        base_path=base_path,
    )
    print("Created new simulation.")

# This triggers the full calculation pipeline:
#   1. Compute mode volumes
#   2. Find non-zero statistics indices
#   3. Compute the mean S matrix
#   4. Pre-compute A matrix values (for lattice grids)
#   5. Compute Cholesky decompositions for each S-matrix block
matrix_pool_manager = ism.get_matrix_pool_manager()
```

> **Note:** The first run may take significant time depending on grid size and integration accuracy. Results are saved to disk, so subsequent runs with the same simulation name will load cached data.

### Step 7: Sample Random S Matrices

Once the statistics are computed, generate random scattering matrices:

```python
# Sample a single pool of N random S matrices
S_matrices = matrix_pool_manager.sample_single_pool(num_matrices=1000)

# Or sample multiple independent pools
S_pools = matrix_pool_manager.sample_multi_pool(num_pools=10, pool_size=100)
```

Each S matrix is a complex 2D array of shape `(4N_prop, 4N_prop)` where N_prop is the number of propagating modes. The matrix has the block structure:

```
S = [ r (2N×2N)   t2 (2N×2N) ]
    [ t (2N×2N)   r2 (2N×2N) ]
```

### Step 8: Analyze Results

Use the `ScatteringMatrix` and `FieldVector` classes to work with the results:

```python
from random_matrix.scattering_matrix.scattering_matrix import ScatteringMatrix
from random_matrix.scattering_matrix.field_vector import FieldVector

sm = ScatteringMatrix(mode_grid)
fv = FieldVector(mode_grid)

# Extract the transmission block for a single S matrix
S = S_matrices[0]
t_block = sm.get_block(S, "t")

# Get the intensity scattered into each output mode for input mode 0 with s-polarization
intensity_per_mode = sm.get_column_intensity(S, incident_index=0, incident_polarization="s")

# Visualize the field distribution
fv.plot_polygon(intensity_per_mode)
```

---

### Complete Example Script

Below is a minimal end-to-end example:

```python
import numpy as np
from random_matrix.modes import mode_grid_factory
from random_matrix.amplitude_matrix import isotropic_sphere
from random_matrix.input_statistics.density_function import DensityFunctionTerm
from random_matrix.input_statistics.medium_parameters import MediumParameters
from random_matrix.input_statistics.medium_statistics import (
    ParticleStatistics, MediumStatistics
)
from random_matrix.input_statistics.integration_task import IntegrationTaskConfig
from random_matrix.input_statistics.input_statistics_manager import InputStatisticsManager

# 1. Mode grid
mode_grid = mode_grid_factory.from_tiling("hexagons", side_length=0.07, r_lim=1.2)

# 2. Particle model
a_matrix_func = isotropic_sphere.get_A

# 3. Particle distribution (monodisperse spheres)
term = DensityFunctionTerm.from_delta({"x": 2.0, "m": 1.2 + 0.01j})
particle_stats = ParticleStatistics(
    terms=[term],
    a_matrix=a_matrix_func,
    a_product=isotropic_sphere.get_A_product,
    a_product_conj=isotropic_sphere.get_A_product_conj,
)
medium_stats = MediumStatistics(particle_terms=[particle_stats])

# 4. Medium parameters
medium_params = MediumParameters(
    wavelength=550e-9,
    number_density=1e18,
    slab_thickness=1e-6,
)

# 5. Integration config
integration_config = IntegrationTaskConfig(
    integration_method="cubature",
    integration_accuracy=3,
)

# 6. Compute statistics
ism = InputStatisticsManager(
    simulation_name="my_first_simulation",
    medium_parameters=medium_params,
    medium_statistics=medium_stats,
    mode_grid=mode_grid,
    integration_task_config=integration_config,
    base_path="./data",
)

# 7. Get matrix pool and sample
mpm = ism.get_matrix_pool_manager()
S_matrices = mpm.sample_single_pool(num_matrices=100)

print(f"Generated {len(S_matrices)} random S matrices")
print(f"Each S matrix has shape: {S_matrices[0].shape}")
```

---

## Examples

Example scripts are provided in the `examples/` directory:

| Script | Description |
|--------|-------------|
| `example_mode_grids.py` | Demonstrates all mode grid types: polar grids, hexagonal/rectangular/triangular tilings, and random Delaunay/Voronoi grids. |
| `example_mode_grids_simple.py` | Simplified version of the grid generation examples. |
| `example_distributions.py` | Shows how to define probability density functions with combinations of Dirac delta and regular distributions, and how to integrate functions against them. |
| `logging_demo.py` | Demonstrates the logging and timing infrastructure. |

---

## Requirements and Dependencies

Key dependencies include:

- **NumPy** / **CuPy** — Array operations (CuPy for optional GPU acceleration)
- **SciPy** — Special functions (Bessel, Hankel, Legendre), sparse matrices, spatial algorithms
- **Numba** — JIT compilation (optional, for performance)
- **quadpy** — Numerical integration schemes
- **Shapely** — Computational geometry (point-in-polygon tests)
- **scikit-spatial** — Geometric operations (circle-polygon intersection)
- **scikit-sparse** — Cholesky decomposition for sparse matrices (CHOLMOD)
- **h5py** — HDF5 file I/O for large datasets
- **tqdm** — Progress bars
- **pathos** — Parallel processing
- **Matplotlib** — Plotting and visualization
