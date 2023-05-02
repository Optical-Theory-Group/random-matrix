import numpy as np

from random_matrix.amplitude_matrix import anisotropic_tmatrix

AMG = anisotropic_tmatrix.AmatrixGenerator()

r = 50e-9
n_diel = 1.4
n_gold = 0.24873 + 1j * 3.0740
wavelength = 600e-9
k = 2 * np.pi / wavelength
x = k * r

cs_sca_diel, cs_ext_diel = AMG.mie_cross_section(x, n_diel, k, N_lim=30)
cs_sca_gold, cs_ext_gold = AMG.mie_cross_section(x, n_gold, k, N_lim=30)

print("Dielectric particle:")
print(f"C_sca = {cs_sca_diel}")
print(f"C_ext = {cs_ext_diel}\n")
print("Gold particle:")
print(f"C_sca = {cs_sca_gold}")
print(f"C_ext = {cs_ext_gold}")
