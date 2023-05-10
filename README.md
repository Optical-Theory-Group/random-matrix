# Random Matrix Simulation Library
![Random grid](/assets/images/random_grid.png "Random grid")

A Python library for performing random matrix simulations of the scattering of polarized light in random media.

## Repository Structure

The core library is contained in the `random_matrix/` directory, which has the following structure:

```
random_matrix/
│
├── amplitude_matrix/ --------------- Calculation of the single particle A matrix
│   ├── _anisotropic_tmatrix.py ----- Anisotropic particle, T matrix method  
│   ├── _chiral_sphere.py ----------- Chiral sphere, Mie theory
│   ├── _isotropic_sphere.py -------- Isotropic sphere, Mie theory
|   └── functions.py ---------------- Summary of functions and parameters for use elsewhere
│
├── modes/ -------------------------- Definitions and properties of modes
│   ├── mode_grid_generator.py ------ Functions for generating ModeGrid objects
│   ├── mode_grid.py ---------------- ModeGrid class, container for Mode objects
│   └── mode.py --------------------- Mode class
│ 
├── statistics/ --------------------- Statistical calculations
│   ├── density_function.py --------- Classes for defining generalized density functions 
│   └── density_integrals.py -------- Integrating functions with respect to pdfs
│
├── utils/ -------------------------- Miscellaneous utility functions
│   ├── array_utils.py -------------- Array manipulation
│   ├── function_utils.py ----------- Mathematical function manipulation
│   ├── geometry_utils.py ----------- Geometric operations
│   ├── integration_utils.py -------- Numerical integration
│   ├── memoize.py ------------------ Function memoization
│   └── plotting_utils.py ----------- Plotting figures
│
└── main.py ------------------------- Main simulation script
```

Some examples can be seen by running the scripts within the root level `examples/` directory.

## Installation

A conda environment called `random_matrix` containing all necessary packages can be installed using the environments file located at `conda/environments.yml`. From the base repository directory, this can be achieved with the following commands:

```
conda env create -f conda/environment.yml
conda activate random_matrix
```

## Repo to-do list

- Consider adding requirements.txt for installing the project dependencies with pip.
- Add project description (more physics and how to use).
- Add license.