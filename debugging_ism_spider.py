import numpy as np
from random_matrix.amplitude_matrix import (
    isotropic_sphere,
)
from random_matrix.input_statistics.density_function import (
    DensityFunctionTerm,
)
import multiprocess as mp
from random_matrix.input_statistics.input_statistics_manager import (
    InputStatisticsManager,
)
from random_matrix.input_statistics.integration_task import (
    IntegrationTaskConfig,
)
from random_matrix.input_statistics.medium_parameters import MediumParameters
from random_matrix.input_statistics.medium_statistics import (
    MediumStatistics,
    ParticleStatistics,
)
from random_matrix.utils import matrix_utils
from random_matrix.modes import mode_grid_factory
import traceback
import os
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Simulation parameters
# -----------------------------------------------------------------------------

wavelength = 550e-9
slab_thickness = 1.8992695221776513e-06
number_density = 5.921762640653617e17

medium_parameters = MediumParameters(
    wavelength=wavelength,
    number_density=number_density,
    slab_thickness=slab_thickness,
)
term = DensityFunctionTerm.from_delta({"x": 2.0, "m": 1.2})
particle_statistics = ParticleStatistics(
    term,
    isotropic_sphere.get_A,
    isotropic_sphere.get_A_product,
    isotropic_sphere.get_A_product_conj,
)
medium_statistics = MediumStatistics([particle_statistics])
integration_task_config = IntegrationTaskConfig()


def run_one(dr, dt_divisor):
    try:
        my_grid = mode_grid_factory.from_dr_dt(
            dr=dr,
            dt=2 * np.pi / dt_divisor,
            r_lim=1.0,
            include_central_mode=False,
            rotation_angle=0.0,
            is_spiderweb=True,
            include_edge_modes=False,
        )
        print(f"Number of modes: {my_grid.num_propagating}")

        simulation_name = f"spiderweb_dr={dr:.3f}_dt_divisor={dt_divisor}"
        ism = InputStatisticsManager(
            simulation_name,
            medium_parameters,
            medium_statistics,
            my_grid,
            integration_task_config,
            base_path="/mnt/raid/rmt/data/",
        )
        pm = ism.get_matrix_pool_manager()
        print("Finishing...")
    except Exception as e:
        traceback.print_exc()
    return


drs = [0.1, 0.05]
dt_divisors = [10, 20, 30, 40, 50]
for dr in drs:
    for dt_divisor in dt_divisors:
        print(f"Starting dr={dr:.3f}, dt_divisor={dt_divisor}")
        p = mp.Process(target=run_one, args=(dr, dt_divisor))
        p.start()
        p.join()
