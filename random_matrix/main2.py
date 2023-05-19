import miepython
import numpy as np

angles = np.linspace(0, np.pi, 1000)
mus = np.cos(angles)

S1, S2 = miepython.mie_S1_S2(1.2, 1.0, 0.0)
