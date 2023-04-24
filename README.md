### Installation

A conda environment containing all packages used in development can be installed with the following commands from the base repository folder:

```
$ conda env create -f conda/environment.yml
$ conda activate random_matrix
```

### Testing
Latest functionality is, for now, usually tested in 
```
$ tests/mode_tests.py
```

### Developer notes


### Code to-do list
- Extend polar grid generation to incorporate evanescent modes.
- Add functionality for checking grid reciprocity (this is automatically enforced by polar or cartesian generation).
- Add random grid generation.
- Add integrators.

### Repo to-do list

- Consider adding requirements.txt for installing the project dependencies with pip.
- Add project description (physics and how to use).
- Add license.