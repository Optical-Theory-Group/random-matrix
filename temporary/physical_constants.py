# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 16:05:19 2022

@author: Matthew Foreman
"""

import numpy as np


class physical_constants:
    
    def __init__(self):
        # Speed of light in vacuum
        self.c = 2.99792458e8

        # Peremability of vacuum
        self.mu0 = 4 * np.pi * 1e-7

        # Permittivity of vacuum
        self.eps0 = 1 / (self.mu0 * self.c**2)

        # Gravitational constant
        self.G = 6.67259e-11

        # Planck constant
        self.h = 6.6260755e-34
        self.hbar = self.h / (2 * np.pi)

        # electron charge
        self.e = 1.60217733e-19

        # magnetic fluxx quantum
        self.Phi = self.h / (2 * self.e)

        # electron mass
        self.me = 9.1093897e-31

        # proton mass
        self.mp = 1.672623e-27

        # atomic mass unit
        self.amu = 1.6605402e-27

        # fine structure constant
        self.alpha = self.mu0 * self.c * (self.e) ** 2 / (2 * self.h)

        # Rydberg constant
        self.Ryd = 1.0973731534e7

        # Avogadro number
        NA = 6.0221367e23

        # Molar gas constanr
        self.R = 8.314510

        # Boltzmann constant
        self.kb = 1.380658e-23

        # Stefan Boltzmann constnat
        self.sigma = (
            np.pi**2
            * (self.kb) ** 4
            / (60 * (self.hbar) ** 3 * (self.c) ** 2)
        )

        # Bohr magneton
        self.mub = 9.2740154e-24

        # Bohr radius
        self.a0 = 5.291772083e-11

        # Impedance of free space
        self.Z0 = self.mu0 * self.c
    