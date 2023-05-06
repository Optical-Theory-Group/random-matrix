# Random Matrix Simulation Library
![Random grid](/assets/images/random_grid.png "Random grid")

A Python library for performing random matrix simulations of the scattering of polarized light in random media.

## Repository Structure

The core library is contained in the `random_matrix/` directory, which has the following structure:

```
random_matrix/
├── amplitude_matrix/ --------------- Calculating the amplitude matrix
|   └── anisotropic_tmatrix.py ------ T matrix method  
│
├── utils/ -------------------------- Utility functions used throughout the library
│   ├── array_utils.py -------------- Array manipulation
│   ├── geometry_utils.py ----------- Geometric operations
|   ├── memoize.py ------------------ Function memoization with caching
│   └── plotting_utils.py ----------- Plotting figures
│
├── integrator.py ------------------- Integration of functions over Mode objects
├── mode_grid_generator.py ---------- Functions for generating ModeGrid objects
├── mode_grid.py -------------------- ModeGrid class, container for Mode objects
└── mode.py ------------------------- Mode class
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