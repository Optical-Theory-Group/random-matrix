import numpy as np

from random_matrix.utils.types import FloatLike


class Matrix:
    matrix: FloatLike
    matrix_type: str


points = np.random.randn(5,2)


y_coordinates = points[:, 1]
x_coordinates = points[:, 0]
min_y_index = np.argmin(y_coordinates)
min_y_values = np.where(np.isclose(y_coordinates, y_coordinates[min_y_index]))[0]
min_x_index = min_y_values[np.argmin(x_coordinates[min_y_values])]