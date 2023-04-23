### Installation

A conda environment containing all necessary packages can be installed with the following commands from the base repository folder:

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

- Cartesian grid generation doesn't work for rare cases where small rectangles close to the circle intersect the circle at more than two points. Only seems to happen when dx or dy are very small, so might not be worth fixing.
- "boundary" is used in several sense, including modes close to the circular boundary as well as a mode's individual boundary, irrespective of whether it is close to the circle or not. Consider renaming the former, e.g. "boundary_points" to "circle_points".
- Linter is not happy with test files. Clean these up at some point. Test also more rigorously.

### Code to-do list
- Add indexing for Cartesian grid generation.
- Add functionality for checking grid reciprocity (this is automatically enforced by polar or cartesian generation).
- Add random grid generation.
- Add integrators.

### Repo to-do list

- Consider adding requirements.txt for installing the project dependencies with pip.
- Add project description (physics and how to use).
- Add license.