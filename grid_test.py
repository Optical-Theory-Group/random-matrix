import pickle
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import shapely
import tqdm
from scipy.spatial import Delaunay

from random_matrix.amplitude_matrix import isotropic_sphere
from random_matrix.input_statistics import density_function, density_integrals
from random_matrix.input_statistics.density_function import (
    DeltaDensityFactor,
    DensityFunction,
    DensityFunctionTerm,
    RegularDensityFactor,
)
from random_matrix.input_statistics.index_finder import IndexFinder
from random_matrix.input_statistics.input_statistics_manager import (
    InputStatisticsManager,
)
from random_matrix.input_statistics.integration_task import (
    IntegrationTaskPreparer,
)
from random_matrix.input_statistics.medium_parameters import MediumParameters
from random_matrix.input_statistics.medium_statistics import (
    MediumStatistics,
    ParticleStatistics,
)
from random_matrix.modes import mode_grid, mode_grid_factory
from random_matrix.scattering_matrix import sampler
from random_matrix.utils import (
    array_utils,
    function_utils,
    geometry_utils,
    integration_utils,
    matrix_utils,
    special_functions,
)

#seed = 0
#np.random.seed(seed)
side_length = 0.4

warnings.filterwarnings("ignore")
my_grid = mode_grid_factory.from_tiling(
    tiling_type="rectangles",
    side_length=(side_length, side_length),
    r_lim=1.2,
    grid_wave_type="propagating",
    rotation_angle=0,
    translation_vector=np.array([0.0, 0.0]),
)
my_grid.plot(show_indices=True)


def Right_Es(kx, ky):
    return ky / (np.sqrt(kx**2 + ky**2)*np.sqrt(1 - kx**2 - ky**2))


def Right_Ep(kx, ky):
    return kx  / np.sqrt(kx**2 + ky**2)


def dipole_field(my_grid):
    vertices_list = [mode.vertices for mode in my_grid.propagating_modes_list]
    E_s = []
    E_p = []
    for unit_cell in vertices_list:
        # Perform Delaunay triangulation for the current square
        tri = Delaunay(unit_cell)
        triangle_indices = tri.simplices  # Indices of the triangle vertices
        triangle_vertices = unit_cell[
            triangle_indices
        ]  # Get the vertices of the triangles
        E_s.append(
            np.sum(
                integration_utils.basic_triangle_integral(
                    Right_Es, triangle_vertices, scheme=None
                )
            )
        )
        E_p.append(
            np.sum(
                integration_utils.basic_triangle_integral(
                    Right_Ep, triangle_vertices, scheme=None
                )
            )
        )

    return np.array(E_s), np.array(E_p)  # Return the list as a numpy array


def plot_field_on_grid(my_grid, E):
    # Extract the vertices and indices of the grid
    vertices_list = [mode.vertices for mode in my_grid.propagating_modes_list]
    indices = [mode.index for mode in my_grid.propagating_modes_list]
    plt.ion()
    # Create the plot
    plt.figure(figsize=(5, 5))
    for i, vertices in enumerate(vertices_list):
        # Get the corresponding field value for the current box
        field_value = E[i]

        # Create a polygon for the current box
        polygon = plt.Polygon(
            vertices, color=plt.cm.viridis(field_value / np.max(E)), alpha=0.8
        )

        # Add the polygon to the plot
        plt.gca().add_patch(polygon)

    # Set plot limits and aspect ratio
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.gca().set_aspect("equal", adjustable="box")

    # Add a colorbar to the side
    sm = plt.cm.ScalarMappable(
        cmap="viridis", norm=plt.Normalize(vmin=np.min(E), vmax=np.max(E))
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), orientation="vertical", pad=0.02)
    cbar.set_label("Field Intensity")

    plt.title("Field Visualization on Grid")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


def centres_of_cells(vertices_list):
    """
    Calculate the center of each square given its vertices.

    Parameters:
        vertices_list (list): A list of arrays, where each array contains the vertices of a square.

    Returns:
        list: A list of arrays, where each array represents the center of a square.
    """
    centres = []
    for vertices in vertices_list:
        # Calculate the mean of the vertices to find the center
        center = np.mean(vertices, axis=0)
        centres.append(center)
    return np.array(centres)


# Example usage
vertices_list = [mode.vertices for mode in my_grid.propagating_modes_list]
centres = centres_of_cells(vertices_list)


def xy_to_rt(x,y):
    """
    Convert Cartesian coordinates (x, y) to polar coordinates (r, theta).

    Parameters:
        x (float): x-coordinate.
        y (float): y-coordinate.

    Returns:
        tuple: A tuple containing the polar coordinates (r, theta).
    """

    # Extract x and y coordinates from the centres
    #x = 1/centres[:, 0]
    #y = 1/centres[:, 1]

    r = np.sqrt(x**2 + y**2)
    theta = -(np.pi / 2 - np.arctan2(y, x))  # Angle in radians
    return r, theta


def anal_dipole_field(r, theta):
    Er = 2 * np.cos(theta) / (r**3)
    Et = np.sin(theta) / (r**3)
    return Er**2 + Et**2
