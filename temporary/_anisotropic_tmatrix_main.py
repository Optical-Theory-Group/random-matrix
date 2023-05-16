#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 17:46:35 2022

@author: mforeman
"""

import itertools
import sys
# %% --------------- testing/debugging ------------
import warnings
from math import factorial

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.constants
from scipy.special import hankel1, jv, lpmv
from scipy.special.orthogonal import p_roots

from random_matrix.amplitude_matrix.anisotropic_tmatrix import AmatrixGenerator

plt.close("all")
warnings.filterwarnings("ignore", message="divide by zero encountered in *")
warnings.filterwarnings("ignore", message="invalid value encountered in *")

c_const = scipy.constants.c
mu = scipy.constants.mu_0
eps0 = scipy.constants.epsilon_0

wavelength = 600e-9
omega = 2 * np.pi * c_const / wavelength
k0 = 2 * np.pi / wavelength

# particle/host properties
n_s = 1.0  # surroundings
n_p = 1.8  # particle
a = wavelength / (2 * np.pi * n_p)
eps_p = n_p**2 * eps0 * np.eye(3)
eps_p[0, 0] = (1.001 * n_p) ** 2 * eps0
eps_s = n_s**2 * eps0

ks = n_s * k0
kp = n_p * k0

# mode indices
nmax = AmatrixGenerator.nWiscombe(k0 * n_p * a)
nmax = 4

# coordinates
Nt = 51  # angular spectrum integration points
Np = 50

Nts = 54  # particle surface points
Nps = 55
psurf_coords, psurf_weights, psurf_norms = AmatrixGenerator.define_sphere(
    a, Nts, Nps
)

# #%% calculate T matrix for homogeneous sphere from T matrix theory
# Q1,Q3 = calc_Q1Q3(nmax,omega,eps_s,eps_p,mu,psurf_coords,psurf_weights,psurf_norms,Nt,Np)
# Tmat1 = calc_Tmatrix_from_Q(Q1,Q3)

# ## calculate T matrix for homogeneous sphere from Mie theory results
# Tmat2 = calc_Tmatrix_sphere(nmax,ks,kp,a)

# plt.close('all');
# fig = plt.figure()
# fig.clear()
# ax = fig.add_subplot(2, 3, 1)
# surf=plt.imshow(np.abs(np.abs(Tmat1)))
# fig.colorbar(surf,ax=ax)

# ax = fig.add_subplot(2, 3, 2)
# surf=ax.imshow(np.abs(Tmat2))
# fig.colorbar(surf,ax=ax)

# ax = fig.add_subplot(2, 3, 3)
# surf=ax.imshow(np.abs(Tmat2)/np.abs(Tmat1))
# fig.colorbar(surf,ax=ax)

# ax = fig.add_subplot(2, 3, 4)
# surf=ax.imshow(np.angle(Tmat1))
# fig.colorbar(surf,ax=ax)

# ax = fig.add_subplot(2, 3, 5)
# surf=ax.imshow(np.angle(Tmat2))
# fig.colorbar(surf,ax=ax)

# ax = fig.add_subplot(2, 3, 6)
# surf=ax.imshow(np.abs(Tmat1-Tmat2))
# fig.colorbar(surf,ax=ax)


# %% calculate some A matrices and compare

tsca = np.linspace(0, np.pi, 20)
psca = np.linspace(0, np.pi, 21)

A_Niall = np.zeros((2, 2, len(tsca), len(psca)), dtype=complex)
A_Matt = np.zeros((2, 2, len(tsca), len(psca)), dtype=complex)

np.random.seed(420)
n1v = np.random.randn(3)

params = {"x": n_s * k0 * a, "m": n_p / n_s, "dm": 0, "particle_type": "mie"}

params2 = {
    "nmax": nmax,
    "omega": omega,
    "eps_s": eps_s,
    "eps_p": eps_p,
    "mu": mu,
    "psurf_coords": psurf_coords,
    "psurf_weights": psurf_weights,
    "psurf_norms": psurf_norms,
    "Nt": Nt,
    "Np": Np,
    "k_s": k0 * n_s,
    "euler_alpha": 0.2,
    "euler_beta": 0.3,
    "euler_gamma": np.pi / 3,
    "rotational_average": False,
    "pdf": None,
}

AMG = AmatrixGenerator()


for jj in range(0, len(tsca)):
    for kk in range(0, len(psca)):
        AMG.update_progress(
            (jj * len(tsca) + kk) / (len(tsca) * len(psca)),
            "Calculating angular distribution...",
        )
        n2v = np.array(
            [
                np.sin(tsca[jj]) * np.cos(psca[kk]),
                np.sin(tsca[jj]) * np.sin(psca[kk]),
                np.cos(tsca[jj]),
            ]
        )

        A = AMG.get_A(n1v, n2v, params)

        A2 = AMG.get_A(n1v, n2v, params2)
        A_Niall[:, :, jj, kk] = A
        A_Matt[:, :, jj, kk] = A2

print(f"k1 = {n1v}")
print(f"k2 = {n2v}")
print("A matrix - Niall :")
print(np.array2string(A, precision=4))
print("A matrix - Tmatrix :")
print(np.array2string(A2, precision=4))
print("--------")

# %%
fig = plt.figure("Niall legacy")
for rr in range(2):
    for ss in range(2):
        ax = fig.add_subplot(2, 2, 2 * rr + ss + 1)
        surf = ax.imshow(
            np.abs(np.squeeze(A_Niall[rr, ss, :, :])), cmap="RdBu"
        )
        fig.colorbar(surf, ax=ax)

fig = plt.figure("Matt Tmatrix")
for rr in range(2):
    for ss in range(2):
        ax = fig.add_subplot(2, 2, 2 * rr + ss + 1)
        surf = ax.imshow(np.abs(np.squeeze(A_Matt[rr, ss, :, :])), cmap="RdBu")
        fig.colorbar(surf, ax=ax)
