# Graph Report - .  (2026-06-04)

## Corpus Check
- 50 files · ~74,351 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1174 nodes · 2512 edges · 61 communities (55 shown, 6 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 310 edges (avg confidence: 0.54)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Mode Grid Construction|Mode Grid Construction]]
- [[_COMMUNITY_Geometry Utilities|Geometry Utilities]]
- [[_COMMUNITY_Matrix Pool Manager|Matrix Pool Manager]]
- [[_COMMUNITY_Anisotropic T-Matrix|Anisotropic T-Matrix]]
- [[_COMMUNITY_Shape Classifier & Logging|Shape Classifier & Logging]]
- [[_COMMUNITY_Input Statistics Manager|Input Statistics Manager]]
- [[_COMMUNITY_Array & Plotting Utilities|Array & Plotting Utilities]]
- [[_COMMUNITY_Mie Scattering (SphereChiral)|Mie Scattering (Sphere/Chiral)]]
- [[_COMMUNITY_T-Matrix Method (Isotropic)|T-Matrix Method (Isotropic)]]
- [[_COMMUNITY_Function Utilities|Function Utilities]]
- [[_COMMUNITY_Integration Task Engine|Integration Task Engine]]
- [[_COMMUNITY_Integration Task Preparer|Integration Task Preparer]]
- [[_COMMUNITY_Index Finder|Index Finder]]
- [[_COMMUNITY_S-Matrix Operations|S-Matrix Operations]]
- [[_COMMUNITY_Medium Statistics & Averages|Medium Statistics & Averages]]
- [[_COMMUNITY_Field Vector|Field Vector]]
- [[_COMMUNITY_Integration Utilities|Integration Utilities]]
- [[_COMMUNITY_Custom Rayleigh Scatterer|Custom Rayleigh Scatterer]]
- [[_COMMUNITY_Density Function Framework|Density Function Framework]]
- [[_COMMUNITY_Statistics Manager Setup|Statistics Manager Setup]]
- [[_COMMUNITY_Integration Rationale & Logging|Integration Rationale & Logging]]
- [[_COMMUNITY_ModeGrid Properties & Access|ModeGrid Properties & Access]]
- [[_COMMUNITY_Sparse Matrix Operations|Sparse Matrix Operations]]
- [[_COMMUNITY_Density Integrals|Density Integrals]]
- [[_COMMUNITY_ModeGrid Internals|ModeGrid Internals]]
- [[_COMMUNITY_ScatteringMatrix Class|ScatteringMatrix Class]]
- [[_COMMUNITY_Density Function Core|Density Function Core]]
- [[_COMMUNITY_Memoization|Memoization]]
- [[_COMMUNITY_Integration Task Config|Integration Task Config]]
- [[_COMMUNITY_File Path Management|File Path Management]]
- [[_COMMUNITY_Matrix Block Operations|Matrix Block Operations]]
- [[_COMMUNITY_Integration Task Variants|Integration Task Variants]]
- [[_COMMUNITY_Shape Quadruple Classification|Shape Quadruple Classification]]
- [[_COMMUNITY_Shape Data & Rationale|Shape Data & Rationale]]
- [[_COMMUNITY_Special Functions|Special Functions]]
- [[_COMMUNITY_Class Quadruple Management|Class Quadruple Management]]
- [[_COMMUNITY_Shape Congruence Detection|Shape Congruence Detection]]
- [[_COMMUNITY_Domain Calculation|Domain Calculation]]
- [[_COMMUNITY_Shape Domain Templates|Shape Domain Templates]]
- [[_COMMUNITY_S-Matrix Sampling|S-Matrix Sampling]]
- [[_COMMUNITY_Covariance Sub-Block Indexing|Covariance Sub-Block Indexing]]
- [[_COMMUNITY_ModeGrid Plotting|ModeGrid Plotting]]
- [[_COMMUNITY_Index Finder Internals|Index Finder Internals]]
- [[_COMMUNITY_Configuration Tools|Configuration Tools]]
- [[_COMMUNITY_Volume Calculation|Volume Calculation]]
- [[_COMMUNITY_Integration Result Types|Integration Result Types]]
- [[_COMMUNITY_Integration Result Lists|Integration Result Lists]]
- [[_COMMUNITY_Reciprocity Matrix|Reciprocity Matrix]]
- [[_COMMUNITY_Shape Classifier Init|Shape Classifier Init]]
- [[_COMMUNITY_System Utilities|System Utilities]]
- [[_COMMUNITY_Data Directory Config|Data Directory Config]]
- [[_COMMUNITY_Covariance Block Extraction|Covariance Block Extraction]]
- [[_COMMUNITY_Conda Environment|Conda Environment]]
- [[_COMMUNITY_Shape Data|Shape Data]]
- [[_COMMUNITY_Shape Quadruple|Shape Quadruple]]
- [[_COMMUNITY_Shape Single|Shape Single]]

## God Nodes (most connected - your core abstractions)
1. `Numeric` - 109 edges
2. `ModeGrid` - 82 edges
3. `MathematicalFunction` - 69 edges
4. `MatrixPoolManager` - 58 edges
5. `AmatrixGenerator` - 48 edges
6. `InputStatisticsManager` - 45 edges
7. `Mode` - 39 edges
8. `bool` - 26 edges
9. `NDArray` - 25 edges
10. `IntegrationTaskPreparer` - 24 edges

## Surprising Connections (you probably didn't know these)
- `Single-Particle Amplitude (A) Matrix` --cites--> `AmatrixGenerator`  [INFERRED]
  README2.md → random_matrix/amplitude_matrix/anisotropic_tmatrix.py
- `Single-Particle Amplitude (A) Matrix` --cites--> `get_A()`  [INFERRED]
  README2.md → random_matrix/amplitude_matrix/custom_rayleigh.py
- `Single-Particle Amplitude (A) Matrix` --cites--> `get_A()`  [INFERRED]
  README2.md → random_matrix/amplitude_matrix/chiral_sphere.py
- `Single-Particle Amplitude (A) Matrix` --cites--> `get_A()`  [INFERRED]
  README2.md → random_matrix/amplitude_matrix/isotropic_sphere.py
- `Single-Particle Amplitude (A) Matrix` --cites--> `get_T()`  [INFERRED]
  README2.md → random_matrix/amplitude_matrix/isotropic_tmatrix.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Particle Scattering Models (A-matrix interface)** — amplitude_matrix_anisotropic_tmatrix_amatrixgenerator, amplitude_matrix_chiral_sphere_get_a, amplitude_matrix_isotropic_sphere_get_a, amplitude_matrix_isotropic_sphere_get_a_3d, amplitude_matrix_custom_rayleigh_get_a, amplitude_matrix_isotropic_tmatrix_get_t [INFERRED 0.95]
- **Modules dependent on scattering_geometry** — amplitude_matrix_scattering_geometry_get_transformation_matrices, amplitude_matrix_chiral_sphere_get_a, amplitude_matrix_isotropic_sphere_get_a, amplitude_matrix_isotropic_sphere_get_a_3d, amplitude_matrix_custom_rayleigh_get_a [INFERRED 0.95]
- **** — modes_mode_grid_factory, modes_mode_mode, modes_mode_side, modes_mode_grid_mode_grid [EXTRACTED 1.00]
- **** — modes_mode_grid_mode_grid, scattering_matrix_field_vector_field_vector, scattering_matrix_scattering_matrix_scattering_matrix [INFERRED 0.85]
- **** — modes_mode_mode, modes_mode_grid_mode_grid, utils_integration_utils, utils_function_utils, utils_geometry_utils [INFERRED 0.85]

## Communities (61 total, 6 thin omitted)

### Community 0 - "Mode Grid Construction"
Cohesion: 0.06
Nodes (66): _cut_and_filter(), _cut_by_circle(), _filter_by_radius(), from_dr_dt(), from_random(), from_rt_vals(), from_tiling(), _generate_base_lattice() (+58 more)

### Community 1 - "Geometry Utilities"
Cohesion: 0.05
Nodes (74): Circle, float128, Circle-cutting algorithm for mode splitting, Mode weight computation, bool, ConvexHull, float, float64 (+66 more)

### Community 2 - "Matrix Pool Manager"
Cohesion: 0.06
Nodes (28): MatrixPoolManager, Save a given pool to memory for reuse in the future, Method for more intense data runs. This particular vesrion does each         mat, Load the given pool from memory, Alias for the single pool S matrices, Alias for the single pool M matrices, Get the array module used for the matrices in single_pool_S, Get the size of the t,r etc. matrices involved (+20 more)

### Community 3 - "Anisotropic T-Matrix"
Cohesion: 0.07
Nodes (9): AmatrixGenerator, Parameters         ----------         Tmat : Tmatrix in reference frame, Parameters         ----------         k1 : array             Wave vector of i, Find the angle between v1 and v2, oriented in accordance with the conventions us, Parameters         ----------         k1 : array             Wave vector of i, Prints a progress bar to console          Parameters         ----------, Parameters         ----------         omega : Optical frequency         eps :, Ported from Archontis Politis' MATLAB wignerD implementation by M.R.Foreman (+1 more)

### Community 4 - "Shape Classifier & Logging"
Cohesion: 0.07
Nodes (25): ShapeClassifier (old), ShapeQuadruple (old), ShapeSingle (old), IndexFinderLogger, InputStatisticsLogger, InputStatisticsManagerLogger, IntegrationTaskPreparerLogger, NullLogger (+17 more)

### Community 5 - "Input Statistics Manager"
Cohesion: 0.07
Nodes (23): InputStatisticsManager, Get a matrix whose elements are like 1/sqrt(w)          Used to distribute weigh, Get a matrix whose elements are like 1/sqrt(wi wj)          Used to distribute w, Quick load of statistics manager when the data directory already         exists, Main method for computing the mean, covariance and pseudo-covariance         for, Calculate scattering matrix indices for which statistics exist and         save, Calculate the mean scattering matrix and save the result to         memory, Pre-compute volumes for integration. (+15 more)

### Community 6 - "Array & Plotting Utilities"
Cohesion: 0.07
Nodes (44): Any, bool, float, float64, int, ndarray, Axes, float (+36 more)

### Community 7 - "Mie Scattering (Sphere/Chiral)"
Cohesion: 0.08
Nodes (36): get_A(), get_A_product(), get_A_product_conj(), get_A_scattering_plane(), get_A(), get_A_3d(), get_A_product(), get_A_product_3d() (+28 more)

### Community 8 - "T-Matrix Method (Isotropic)"
Cohesion: 0.10
Nodes (37): B_mn(), C_mn(), compute_Js(), const_mn(), create_hull_GC(), create_hull_inv_transform(), create_hull_random(), create_hull_uniform() (+29 more)

### Community 9 - "Function Utilities"
Cohesion: 0.10
Nodes (27): Any, int, MathematicalFunction, Numeric, str, Signature, add_functions(), equate_arguments() (+19 more)

### Community 10 - "Integration Task Engine"
Cohesion: 0.10
Nodes (16): ABC, build_integration_task(), CubatureIntegrationTask, IntegrationResult, IntegrationResultList, IntegrationTask, MidpointIntegrationTask, General integration task container.      Consists of a function that is to be in (+8 more)

### Community 11 - "Integration Task Preparer"
Cohesion: 0.17
Nodes (13): Append results from another result list object to itself, Get a statistic from the list based on desired properties, Get an integration task list consisting of all necessary tasks.          This is, Main method for preparing mean integral tasks, Choose the correct integration method based upon the config, Get an integration result list consisting of all required         statistics., Main method for computing mean results, Get all the covariance results for lattice based mode_grids         where memory (+5 more)

### Community 12 - "Index Finder"
Cohesion: 0.15
Nodes (15): IndexFinder, Get the independent elements of the propagating-propagating block         of the, Get indices of the scattering matrix for which the mean needs to         be calc, Get indices of 2x2 blocks in the pp section of the scattering         matrix for, Class for determining which elements and combinations of elements of     the sca, Get quadruples of indices of the scattering matrix for which the         covaria, Get indices of 2x2 blocks in the pp section of the scattering         matrix for, Get indices of the scattering matrix for which the pseudo-covariance         nee (+7 more)

### Community 13 - "S-Matrix Operations"
Cohesion: 0.13
Nodes (26): bool, ndarray, block_cholesky(), get_cov_sub_block_from_indices(), get_exchange_matrix(), get_M_energy_matrix(), get_M_reciprocity_matrix(), get_pauli_x() (+18 more)

### Community 14 - "Medium Statistics & Averages"
Cohesion: 0.12
Nodes (11): density_integrals module, MediumStatistics, ParticleStatistics, Computes <A> over the contained particle terms and A functions, Subclass of DensityFunction for keeping track of components of the     total par, Computes <AA*> over the contained particle terms and A functions, Computes <AA> over the contained particle terms and A functions, Check that density function variables are a proper subset of         A matrix va (+3 more)

### Community 15 - "Field Vector"
Cohesion: 0.13
Nodes (11): array, int, ModeGrid, ndarray, str, FieldVector, Plot the values contained in array on top of the mode grid using         a polyg, Plot the values contained in array on top of the mode grid using         a polyg (+3 more)

### Community 16 - "Integration Utilities"
Cohesion: 0.15
Nodes (22): Any, bool, ConvexHull, int, MathematicalFunction, ndarray, Numeric, str (+14 more)

### Community 17 - "Custom Rayleigh Scatterer"
Cohesion: 0.17
Nodes (19): get_A(), get_A_HG_scatterer(), get_A_product(), get_A_product_conj(), get_scattering_cs(), Synthetic scattering A-matrix using Henyey-Greenstein phase function,     normal, complex, IntegrationTask (old) (+11 more)

### Community 18 - "Density Function Framework"
Cohesion: 0.13
Nodes (10): Return sets containing all of the variables         from the regular and delta d, Compute the integral of the probability density function term, Compute the total integral of the probability density function, Create an instance with only a single density function and no         deltas, Create an instance with only delta functions, float, MathematicalFunction, Numeric (+2 more)

### Community 19 - "Statistics Manager Setup"
Cohesion: 0.12
Nodes (13): Create a JSON file containing metadata for the simulation., Initialize paths for saving simulation data and create directories         if th, Initialize loggers for this and helper classes., Initialize helper classes for calculation of statistics., Save medium parameters, medium statistics, and mode grid to disk., Save mode grid plots with and without indices., IntegrationTaskConfig, bool (+5 more)

### Community 20 - "Integration Rationale & Logging"
Cohesion: 0.18
Nodes (11): Container class for holding many IntegrationResult objects., Perofrm the integral associated with the task., Container class for many IntegrationTask objects., Perofrm the integral associated with the task., Container class for many IntegrationTask objects., Main method for preparing covariance integral tasks, Container class for many IntegrationResult objects., Main method for preparing covariance integral tasks (+3 more)

### Community 21 - "ModeGrid Properties & Access"
Cohesion: 0.16
Nodes (5): ModeGrid, A class used to represent a grid of modes.      To construct a grid, please do n, Get a mode from "modes" dictionary          Parameters         ----------, int, ndarray

### Community 22 - "Sparse Matrix Operations"
Cohesion: 0.12
Nodes (16): csc_matrix, spmatrix, block_cholesky_sparse(), Covariance matrix block indices, get_cholesky_decomposition(), get_closest_unitary_approximation(), get_real_covariance_matrix(), r_sym() (+8 more)

### Community 23 - "Density Integrals"
Cohesion: 0.20
Nodes (17): DeltaDensityFactor, DensityFunction, integrate_by_delta_density_factor(), integrate_by_delta_density_function(), integrate_by_density(), integrate_by_regular_density_factor(), integrate_by_regular_density_function(), Module for computing statistical quantities associated with functions. (+9 more)

### Community 24 - "ModeGrid Internals"
Cohesion: 0.16
Nodes (7): Determine whether or not the list of modes satisfies reciprocity.          Param, Get a list of indices for saving the modes in the modes dictionary.          If, Get a dictionary of modes with correct indices and wave_types          Parameter, Separate a list of modes into lists of propagating and evanescent         compon, Validates input data and sets up the mode grid, Find the index of a mode's reciprocal partner in mode_list          Parameters, Mode

### Community 25 - "ScatteringMatrix Class"
Cohesion: 0.21
Nodes (8): int, ModeGrid, ndarray, str, It is assumed that the matrix has already been reduced to one of         the fou, It is assumed that the matrix has already been reduced to one of         the fou, It is assumed the matrix given is one of the blocks, i.e. r,t,t2,r2, ScatteringMatrix

### Community 26 - "Density Function Core"
Cohesion: 0.18
Nodes (8): DensityFunction, DensityFunctionTerm, Classes for describing probability density functions containing Dirac delta dist, A single term in the probability density function, potentially     containing bo, Check that the variables are not repeated in the delta         functions and reg, A probability density function as a collection of terms.      Attributes     ---, Check that the variables within each term object are         consistent., Check that the given probability density function is normalized         i.e. its

### Community 27 - "Memoization"
Cohesion: 0.24
Nodes (9): Any, bool, float, Numeric, Path, str, Memoize, Function memoization module.  Memoization can be used either by crating a "Memoi (+1 more)

### Community 28 - "Integration Task Config"
Cohesion: 0.18
Nodes (12): Configuration metadata for controlling the degree of parallelisation of the, Configuration metadata for controlling the degree of parallelisation of the, Protocol, IntegrationTaskConfig, InputStatisticsLogger, MediumParameters, MediumStatistics, ModeGrid (+4 more)

### Community 29 - "File Path Management"
Cohesion: 0.16
Nodes (7): InputStatisticsPaths, MatrixPoolsPaths, Path names for all stored data relevant to the incident statistics that     feed, Path names for all stored data relevant to the matrix pools and     subsequent d, int, Path, str

### Community 30 - "Matrix Block Operations"
Cohesion: 0.16
Nodes (15): str, get_block(), get_block_indices(), get_M_from_S(), get_reciprocal_sub_block_indices(), get_S_from_M(), get_sub_block(), get_sub_block_indices() (+7 more)

### Community 31 - "Integration Task Variants"
Cohesion: 0.29
Nodes (13): _get_covariance_integrand(), _get_covariance_integration_results(), _get_covariance_integration_results_parallelized(), _get_covariance_integration_results_partial(), Get the integrand for covariance calculations. Note that this does         not i, Main method for computing covariance results, Get the integrand for covariance calculations. Note that this does     not inclu, Any (+5 more)

### Community 32 - "Shape Quadruple Classification"
Cohesion: 0.16
Nodes (7): Get a list of singles_indices for the class's members, Get the total number of quadruples in all of the classes, List of all singles_indices for everything in all the classes, Get the class containing the quadruple with given indices, Return the quadruple with the given indices, Get the template for a given quadruple, int

### Community 33 - "Shape Data & Rationale"
Cohesion: 0.16
Nodes (9): Mode that has been classified according to its geometric properties.      Attrib, self.shape_data, but with all quantities going backwards, Mode that has been classified according to its geometric properties.      Attrib, Array containing the lenghts and angles, self.shape_data, but with all quantities going backwards, Array containing the lenghts and angles, ShapeSingle, ndarray (+1 more)

### Community 34 - "Special Functions"
Cohesion: 0.24
Nodes (13): MathematicalFunction, Numeric, get_sinc_mean_kappa(), identity(), inverse_kz(), kz(), Note this is NORMALISED by k, Factor that appears in lots of integrals.      Often appears as sec(theta) (+5 more)

### Community 35 - "Class Quadruple Management"
Cohesion: 0.18
Nodes (6): ClassQuadruple, Append given quadruple to the members list, Number of members in the class (note that this doesn't include         the templ, Get the singles_indices for the class's template, Get a list of singles_indices for the class's members and         template, Append new class to classes

### Community 36 - "Shape Congruence Detection"
Cohesion: 0.21
Nodes (11): get_angle(), get_congruent_shape_rotation_angle(), Method for generating ShapeData objects from arrays of vertices, Method for generating ShapeData objects from arrays of vertices, Gets the angle that the reference_shape_data would need to be rotated by     to, Gets the angle that the reference_shape_data would need to be rotated by     to, float64, get_shape_data() (+3 more)

### Community 37 - "Domain Calculation"
Cohesion: 0.24
Nodes (4): ClassQuadrupleList, Get domain templates, paralelllised over multiple cores, Get quadruples with domains for quadruples that are not templates, ShapeClassifier

### Community 38 - "Shape Domain Templates"
Cohesion: 0.24
Nodes (5): Given points in the 4D base domain, calculate their extensions in the         fi, List of all of the quadruples, including the template, Gets the integration domain associated with a template quadruple, Get template domains as internal attribute, ShapeQuadruple

### Community 39 - "S-Matrix Sampling"
Cohesion: 0.24
Nodes (10): ModeGrid, ModeGrid reciprocity system, bool, int, ndarray, FieldVector, reorder_block(), S_sampler() (+2 more)

### Community 40 - "Covariance Sub-Block Indexing"
Cohesion: 0.24
Nodes (11): int, block_cholesky_sparse_recursive(), get_cov_block_indices(), get_cov_starting_index(), get_cov_sub_block(), get_cov_sub_block_indices(), Get the starting index within the covariance matrix from information     about t, Get the cov matrix indices as slice objects from information about the     block (+3 more)

### Community 41 - "ModeGrid Plotting"
Cohesion: 0.20
Nodes (6): Returns the index of the mode whose polygon contains or touches         the give, Draws the grid of modes.          Parameters         ----------         show_ind, bool, float, Path, str

### Community 42 - "Index Finder Internals"
Cohesion: 0.22
Nodes (6): Get indices of 2x2 blocks in the pp section of the scattering         matrix for, Update covariance_indices dictionary with the correlation (i,j,u,v)., update_covariance_indices(), Any, InputStatisticsLogger, ModeGrid

### Community 43 - "Configuration Tools"
Cohesion: 0.43
Nodes (7): get_data_directory(), load_config(), print_data_directory(), save_config(), set_data_directory(), Path, str

### Community 44 - "Volume Calculation"
Cohesion: 0.29
Nodes (4): Pre-compute A matrix values for faster computation of statistics         later o, Pre-compute volumes for faster computation of statistics         later on, Pre-compute volumes for faster computation of statistics         later on, Path

### Community 45 - "Integration Result Types"
Cohesion: 0.40
Nodes (3): General integration result container.      Contains results from executing an In, General integration result container.      Contains results from executing an In, IntegrationResult

### Community 46 - "Integration Result Lists"
Cohesion: 0.40
Nodes (3): Append tasks from another task list object to itself, Append tasks from another task list object to itself, Self

### Community 47 - "Reciprocity Matrix"
Cohesion: 0.40
Nodes (3): ModeGrid class for use in scattering calculations.  ModeGrid serves as a contain, The scattering matrix satisfies          S = rec_mat @ S^T @ rec_mat, Numeric

### Community 48 - "Shape Classifier Init"
Cohesion: 0.50
Nodes (3): InputStatisticsLogger, ModeGrid, str

### Community 49 - "System Utilities"
Cohesion: 0.50
Nodes (3): bool, float, get_current_ram_usage()

## Knowledge Gaps
- **31 isolated node(s):** `Any`, `bool`, `str`, `data_directory`, `str` (+26 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Numeric` connect `Statistics Manager Setup` to `Mode Grid Construction`, `Geometry Utilities`, `Input Statistics Manager`, `Function Utilities`, `Integration Task Engine`, `Integration Task Preparer`, `Medium Statistics & Averages`, `Integration Utilities`, `Custom Rayleigh Scatterer`, `Density Function Framework`, `Integration Rationale & Logging`, `ModeGrid Properties & Access`, `Density Integrals`, `ModeGrid Internals`, `Density Function Core`, `Memoization`, `Integration Task Config`, `Special Functions`, `ModeGrid Plotting`, `Volume Calculation`, `Integration Result Types`, `Integration Result Lists`, `Reciprocity Matrix`?**
  _High betweenness centrality (0.458) - this node is a cross-community bridge._
- **Why does `ModeGrid` connect `ModeGrid Properties & Access` to `Mode Grid Construction`, `Shape Data & Rationale`, `Shape Quadruple Classification`, `Class Quadruple Management`, `Shape Congruence Detection`, `Domain Calculation`, `Shape Domain Templates`, `ModeGrid Plotting`, `Index Finder Internals`, `Index Finder`, `Field Vector`, `Reciprocity Matrix`, `Shape Classifier Init`, `Statistics Manager Setup`, `ModeGrid Internals`, `ScatteringMatrix Class`?**
  _High betweenness centrality (0.280) - this node is a cross-community bridge._
- **Why does `get_A()` connect `Custom Rayleigh Scatterer` to `Mie Scattering (Sphere/Chiral)`?**
  _High betweenness centrality (0.184) - this node is a cross-community bridge._
- **Are the 107 inferred relationships involving `Numeric` (e.g. with `Circle` and `DeltaDensityFactor`) actually correct?**
  _`Numeric` has 107 INFERRED edges - model-reasoned connections that need verification._
- **Are the 43 inferred relationships involving `ModeGrid` (e.g. with `array` and `IndexFinder`) actually correct?**
  _`ModeGrid` has 43 INFERRED edges - model-reasoned connections that need verification._
- **Are the 58 inferred relationships involving `MathematicalFunction` (e.g. with `DeltaDensityFactor` and `DensityFunction`) actually correct?**
  _`MathematicalFunction` has 58 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `MatrixPoolManager` (e.g. with `complex` and `float`) actually correct?**
  _`MatrixPoolManager` has 3 INFERRED edges - model-reasoned connections that need verification._