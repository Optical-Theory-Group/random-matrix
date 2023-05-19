from random_matrix.utils.types import FloatLike


def get_A(
    k_inc: FloatLike,
    k_sca: FloatLike,
    x: FloatLike,
    m: FloatLike,
) -> FloatLike:
    return (k_inc + k_sca) * x * m


get_A.particle_type = "sphere"
