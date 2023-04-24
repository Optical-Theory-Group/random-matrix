# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 15:45:00 2022

@author: Matthew Foreman

Functions to calculate the T-matrix of a general anisotropic particle.
See https://doi.org/10.1016/j.jqsrt.2014.11.009 for details.

"""

import numpy as np
import scipy as sp
from scipy.special import lpmv, jv, hankel1
from scipy.special.orthogonal import p_roots
from math import factorial
import itertools
import matplotlib.pyplot as plt
from PhysicalConstants import physical_constants
import sys

class AmatrixGenerator:
    def __init__(self,):
        pass

    #%% --------------- utility functions  ------------------------------------
    @staticmethod
    def update_progress(progress,status='',barLength=20):
        """
        Prints a progress bar to console

        Parameters
        ----------
        progress : float
            Variable ranging from 0 to 1 indicating fractional progress.
        status : TYPE, optional
            Status text to suffix progress bar. The default is ''.
        barLength : str, optional
            Controls width of progress bar in console. The default is 20.

        """
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
            status = 'error: progress var must be float\r\n'
        if progress < 0:
            progress = 0
            status = 'Halt...\r\n'
        if progress >= 1:
            progress = 1
            status = 'Done.\r\n'
        block = int(round(barLength*progress))
        text = '\rPercent: [{0}] {1:.2f}% {2}'.format( '#'*block + '-'*(barLength-block), progress*100, status)
        sys.stdout.write(text)
        sys.stdout.flush()


    #  Matrix to convert the spherical vector components (Ur,Ut,Up) of a 3D field specified at
    #  positions defined by theta and phi into the equivalent Cartesian components (Ux,Uy,Uz)
    @staticmethod
    def sph2cart_comp_mat(theta,phi):
        R = np.array([ [np.cos(phi)*np.sin(theta) , np.cos(phi)*np.cos(theta), - np.sin(phi) ] ,
                       [np.sin(phi)*np.sin(theta) , np.sin(phi)*np.cos(theta),   np.cos(phi) ] ,
                       [   np.cos(theta)          ,    - np.sin(theta)       ,   0*theta] ])

        return R

    #  Matrix to convert the Cartesian vector components (Ux,Uy,Uz) of a 3D field specified at
    #  positions defined by theta and phi into the equivalent spherical points (Ur,Ut,Up)
    @staticmethod
    def cart2sph_comp_mat(theta,phi):
        R = np.array([ [np.cos(phi)*np.sin(theta) , np.sin(phi)*np.sin(theta),   np.cos(theta) ] ,
                          [np.cos(phi)*np.cos(theta) , np.sin(phi)*np.cos(theta), - np.sin(theta) ] ,
                          [ - np.sin(phi)          ,      np.cos(phi)       ,   0*theta ] ])

        return R

    # Convert the Cartesian vector components (Ux,Uy,Uz) of a 3D field specified at
    # positions defined by theta and phi into the equivalent spherical points (Ur,Ut,Up)
    @staticmethod
    def cart2sph_comp_vec(U,theta,phi):
        R = AmatrixGenerator.cart2sph_comp_mat(theta,phi)
        Uout = np.einsum('mn...,n...->m...', R,U)
        return Uout

    # Convert the spherical vector components (Ur,Ut,Up) of a 3D field specified at
    # positions defined by theta and phi into the equivalent Cartesian components (Ux,Uy,Uz)
    @staticmethod
    def sph2cart_comp_vec(U,theta,phi):
        R = AmatrixGenerator.sph2cart_comp_mat(theta,phi)
        Uout = np.einsum('mn...,n...->m...', R,U)
        return Uout

    # convert a matrix relating cartesian field components to a spherical polar basis matrix
    @staticmethod
    def cart2sph_convert_matrix(M,theta,phi):
        R  = AmatrixGenerator.sph2cart_comp_mat(theta,phi)
        iR = AmatrixGenerator.cart2sph_comp_mat(theta,phi)

        # R.T @ M @ R
        iRmR = np.einsum('im...,mn,nj...->ij...', iR, M, R)
        return iRmR

    # convert a matrix relating cartesian field components to a spherical polar basis matrix
    @staticmethod
    def sph2cart_convert_matrix(M,theta,phi):
        iR  = AmatrixGenerator.sph2cart_comp_mat(theta,phi)
        R = AmatrixGenerator.cart2sph_comp_mat(theta,phi)

        # R.T @ M @ R
        iRmR = np.einsum('im...,mn,nj...->ij...', iR, M, R)
        return iRmR

    # converts spherical polar coordinates to cartesian coordinates
    @staticmethod
    def sph2cart(p,t,r):
        z = r*np.cos(t)
        rsin = r*np.sin(t)
        x = rsin * np.cos(p)
        y = rsin * np.sin(p)

        return (x,y,z)

    # converts cartesian coordinates to spherical polar coordinates
    @staticmethod
    def cart2sph(x,y,z): # NB convention on t differs from MATLAB
        xy = x**2 + y**2
        r = np.sqrt(xy + z**2)
        t = np.pi/2 - np.arctan2(z,np.sqrt(xy)) # for elevation angle defined from Z-axis down
        p = np.arctan2(y, x)

        return (p,t,r)

    @staticmethod
    def lm_2_n_all(l,m):
        if np.any(np.abs(m) > l) :
            raise ValueError ('abs(m) > n')
        else:
            n = int(l**2 + m + l )
        return n

    @staticmethod
    def n_2_lm_all(n):
        n += 1 # due to difference between matlab and python indexing
        l = int(np.floor(np.sqrt(n-1)))
        m = int(n - l**2 - l - 1)
        return l,m

    @staticmethod
    def nWiscombe(z):
        return np.ceil(z + 4*z**(1/3) + 2);

    @staticmethod
    def plot_fields(ord1,ord2,E):
        ord1=np.squeeze(ord1)
        ord2=np.squeeze(ord2)
        fig = plt.figure()
        for jj in range(3):
            ax = fig.add_subplot(2, 2, jj+1)
            surf = ax.imshow(np.real(np.squeeze(E[jj])),cmap='RdBu')
            fig.colorbar(surf,ax=ax)
            # ax.axis([ord1.min(), ord1.max(), ord2.min(), ord2.max()])

        ax = fig.add_subplot(2, 2, 4)
        surf = ax.imshow(np.squeeze(np.abs(E[0])**2 + np.abs(E[1])**2 + np.abs(E[2])**2),cmap='RdBu')
        fig.colorbar(surf,ax=ax)
        # ax.axis([ord1.min(), ord1.max(), ord2.min(), ord2.max()])

    #%% --------------- vector spherical wave functions ------------
    @staticmethod
    def mnp_mn(m,n,T,P):
        pi_mn,tau_mn,Lmn = AmatrixGenerator.pi_tau_mn(m,n,T)

        exp_fac = np.exp(1j*m*P)
        pi_exp = exp_fac * 1j * m * pi_mn
        tau_exp = exp_fac * tau_mn
        m_mn = np.array([ 0*T ,  pi_exp,  - tau_exp])
        n_mn = np.array([ 0*T ,  tau_exp,    pi_exp])
        p_mn = np.array([ n*(n+1) * Lmn * exp_fac ,  0*T , 0*T  ])

        return (m_mn, n_mn, p_mn)

    @staticmethod
    def pi_tau_mn(m,n,theta):
        theta[theta==0] = np.sqrt(2*np.finfo(np.float64).eps) # remove issues with divide by zero
        theta[theta==np.pi] = np.pi+np.sqrt(2*np.finfo(np.float64).eps) # remove issues with divide by zero

        Ln   = lpmv(np.abs(m),n,np.cos(theta))
        Lnp1 = lpmv(np.abs(m),n+1,np.cos(theta))

        norm = 1
        pi_out  = Ln / (np.sin(theta));
        tau_out = norm * ( - (1 + n) * (np.cos(theta) / np.sin(theta)) * Ln
                           + (1 - np.abs(m) + n) * Lnp1 / (np.sin(theta)) )

        Ln = norm * Ln;
        # pi_out[~np.isfinite(pi_out)] = 0
        # tau_out[~np.isfinite(tau_out)] = 0
        # Ln[~np.isfinite(Ln)] = 0
        return (pi_out, tau_out, Ln)

    @staticmethod
    def MN_lmn(k,l,m,n,R,T,P):
        # k = wavenumber
        # l = index 3 for regular bessel j functions or 1 for hankel h1 function
        # m,n = azimuthal and polar mode index
        # R,T,P = spherical polar coordinates at which to return M, N modes
        m_mn,n_mn,p_mn = AmatrixGenerator.mnp_mn(m,n,T,P);

        if l == 3: # bessel function - non -diverging multipoles
            zl = np.sqrt(np.pi / (2*k*R)) * jv(n + 1/2,k*R)
            zldot = (1 + n) * zl / (k*R) -  np.sqrt(np.pi / (2*k*R)) * jv(n + 1 + 1/2,k*R)
        elif l == 1: # hankel function - outward propagating
            zl = np.sqrt(np.pi / (2*k*R)) * hankel1(n + 1/2,k*R)
            zldot = (1 + n) * zl / (k*R) -  np.sqrt(np.pi / (2*k*R)) * hankel1(n + 1 + 1/2,k*R)

        M_lmn_sp = np.array([zl * m_mn[kk] for kk in range(3) ])
        N_lmn_sp = np.array([(zl/(k*R)) * p_mn[kk] + zldot * n_mn[kk]  for kk in range(3)    ])
        M_lmn_sp[~np.isfinite(M_lmn_sp)] = 0
        N_lmn_sp[~np.isfinite(N_lmn_sp)] = 0

        M_lmn = AmatrixGenerator.sph2cart_comp_vec(M_lmn_sp,T,P)
        N_lmn = AmatrixGenerator.sph2cart_comp_vec(N_lmn_sp,T,P)

        M_lmn[~np.isfinite(M_lmn)] = 0
        N_lmn[~np.isfinite(N_lmn)] = 0
        return (M_lmn,N_lmn)

    @staticmethod
    def MN_orthogonality_matrix(k,nmax,R,Nt,Np):
        # define angular coordinates for angular spectrum caclulation of X and Y modes
        psurf_coords,psurf_weights,psurf_norms = AmatrixGenerator.define_sphere(R, Nt,Np)
        P,T,_    = AmatrixGenerator.cart2sph(psurf_coords[0],psurf_coords[1],psurf_coords[2])

        # precalculate necessary modes
        nummodes = AmatrixGenerator.lm_2_n_all(nmax,nmax) + 1
        M1_mn = np.squeeze(np.zeros( (nummodes,3) + T.shape, dtype=complex))
        N1_mn = np.squeeze(np.zeros( (nummodes,3) + T.shape, dtype=complex))
        M3_mn = np.squeeze(np.zeros( (nummodes,3) + T.shape, dtype=complex))
        N3_mn = np.squeeze(np.zeros( (nummodes,3) + T.shape, dtype=complex))
        m1m1 = np.zeros( (nummodes,nummodes) , dtype=complex)
        n1m1 = np.zeros( (nummodes,nummodes) , dtype=complex)
        m1n1 = np.zeros( (nummodes,nummodes) , dtype=complex)
        n1n1 = np.zeros( (nummodes,nummodes) , dtype=complex)
        m3m3 = np.zeros( (nummodes,nummodes) , dtype=complex)
        n3m3 = np.zeros( (nummodes,nummodes) , dtype=complex)
        m3n3 = np.zeros( (nummodes,nummodes) , dtype=complex)
        n3n3 = np.zeros( (nummodes,nummodes) , dtype=complex)

        m1m1_anal = np.zeros( (nummodes,nummodes) , dtype=complex)
        n1m1_anal = np.zeros( (nummodes,nummodes) , dtype=complex)
        m1n1_anal = np.zeros( (nummodes,nummodes) , dtype=complex)
        n1n1_anal = np.zeros( (nummodes,nummodes) , dtype=complex)
        m3m3_anal = np.zeros( (nummodes,nummodes) , dtype=complex)
        n3m3_anal = np.zeros( (nummodes,nummodes) , dtype=complex)
        m3n3_anal = np.zeros( (nummodes,nummodes) , dtype=complex)
        n3n3_anal = np.zeros( (nummodes,nummodes) , dtype=complex)

        for jj in range(1,nummodes): # don't include multipole
            n,m = AmatrixGenerator.n_2_lm_all(jj)
            AmatrixGenerator.update_progress(jj/nummodes, "Pre-calculating modes: (n,m) = ({},{})".format(n,m))
            M3_mn[jj],N3_mn[jj] =  AmatrixGenerator.MN_lmn(k,3,m,n,R,T,P)
            M1_mn[jj],N1_mn[jj] =  AmatrixGenerator.MN_lmn(k,1,m,n,R,T,P)

        for jj1 in range(1,nummodes):
            n1,m1 = AmatrixGenerator.n_2_lm_all(jj1)
            jj1minusm = AmatrixGenerator.lm_2_n_all(n1,-m1) # since projections required -m modes

            zn3 = np.sqrt(np.pi / (2*k*R)) * jv(n1 + 1/2,k*R)
            zn1 = np.sqrt(np.pi / (2*k*R)) * hankel1(n1 + 1/2,k*R)
            zn3dot = (1 + n1) * zn3 / (k*R) -  np.sqrt(np.pi / (2*k*R)) * jv(n1 + 1 + 1/2,k*R)
            zn1dot = (1 + n1) * zn1 / (k*R) -  np.sqrt(np.pi / (2*k*R)) * hankel1(n1 + 1 + 1/2,k*R)
            m1m1_anal[jj1,jj1] = zn1**2 * 4 * np.pi * n1*(n1+1)*factorial(n1+np.abs(m1)) / ((2*n1+1)* factorial(n1-np.abs(m1)))
            m3m3_anal[jj1,jj1] = zn3**2 * 4 * np.pi * n1*(n1+1)*factorial(n1+np.abs(m1)) / ((2*n1+1)* factorial(n1-np.abs(m1)))

            n1n1_anal[jj1,jj1] = (zn1*n1*(n1+1)/(k*R))**2 * 4*np.pi * factorial(n1+np.abs(m1)) / ((2*n1+1)* factorial(n1-np.abs(m1))) \
                                    + zn1dot**2 * n1*(n1+1) * 4*np.pi * factorial(n1+np.abs(m1)) / ((2*n1+1)* factorial(n1-np.abs(m1)))
            n3n3_anal[jj1,jj1] = (zn3*n1*(n1+1)/(k*R))**2 * 4*np.pi * factorial(n1+np.abs(m1)) / ((2*n1+1)* factorial(n1-np.abs(m1))) \
                                    + zn3dot**2 * n1*(n1+1) * 4*np.pi * factorial(n1+np.abs(m1)) / ((2*n1+1)* factorial(n1-np.abs(m1)))


            for jj2 in range(1,nummodes): # -n to n
                n2,m2 = AmatrixGenerator.n_2_lm_all(jj2)
                # update_progress((nummodes*jj1+jj2)/nummodes**2, "Calculating Q overlap integrals")
                m1m1[jj1,jj2]  = np.sum(np.einsum('j...,j...->...',M1_mn[jj2],M1_mn[jj1minusm])  * psurf_weights)
                n1m1[jj1,jj2]  = np.sum(np.einsum('j...,j...->...',N1_mn[jj2],M1_mn[jj1minusm])  * psurf_weights)
                m1n1[jj1,jj2]  = np.sum(np.einsum('j...,j...->...',M1_mn[jj2],N1_mn[jj1minusm])  * psurf_weights)
                n1n1[jj1,jj2]  = np.sum(np.einsum('j...,j...->...',N1_mn[jj2],N1_mn[jj1minusm])  * psurf_weights)

                m3m3[jj1,jj2]  = np.sum(np.einsum('j...,j...->...',M3_mn[jj2],M3_mn[jj1minusm])  * psurf_weights)
                n3m3[jj1,jj2]  = np.sum(np.einsum('j...,j...->...',N3_mn[jj2],M3_mn[jj1minusm])  * psurf_weights)
                m3n3[jj1,jj2]  = np.sum(np.einsum('j...,j...->...',M3_mn[jj2],N3_mn[jj1minusm])  * psurf_weights)
                n3n3[jj1,jj2]  = np.sum(np.einsum('j...,j...->...',N3_mn[jj2],N3_mn[jj1minusm])  * psurf_weights)



        Q1 = np.concatenate( (np.concatenate((m1m1[1:,1:],n1m1[1:,1:]),axis=0), \
                             np.concatenate((m1n1[1:,1:],n1n1[1:,1:]),axis=0)), axis=1)
        Q3 = np.concatenate( (np.concatenate((m3m3[1:,1:],n3m3[1:,1:]),axis=0), \
                             np.concatenate((m3n3[1:,1:],n3n3[1:,1:]),axis=0)), axis=1)

        Q1_anal = np.concatenate( (np.concatenate((m1m1_anal[1:,1:],n1m1_anal[1:,1:]),axis=0), \
                             np.concatenate((m1n1_anal[1:,1:],n1n1_anal[1:,1:]),axis=0)), axis=1)
        Q3_anal = np.concatenate( (np.concatenate((m3m3_anal[1:,1:],n3m3_anal[1:,1:]),axis=0), \
                             np.concatenate((m3n3_anal[1:,1:],n3n3_anal[1:,1:]),axis=0)), axis=1)

        return Q1,Q3,Q1_anal,Q3_anal

    #%% ---------------- quasi vector spherical wave functions --------
    @staticmethod
    def find_eigenstates(perm,theta,phi):
        # construct Lambda matrix
        kappa = np.linalg.inv(perm)
        Lambda = AmatrixGenerator.cart2sph_convert_matrix(kappa,theta,phi)
        ltt = Lambda[1,1]
        lpp = Lambda[2,2]
        ltp = Lambda[1,2]
        lpt = Lambda[2,1]

        # test to see if permittivity is isotropic
        evs = np.linalg.eigvals(perm)
        if np.all(np.isclose(evs/evs[0],1)): # all eigenvalues are equal
            print(' -- isotropic particle detected -- ')
            zeros = np.zeros(theta.shape)
            f1 = zeros
            f2 = zeros
            ones = np.ones(theta.shape)
            l12 = np.array([0.5*(ltt + lpp),  0.5*(ltt + lpp) ])
            v1 = np.array([zeros, zeros , ones])
            v2 = np.array([zeros, ones, zeros])
            v12 = np.array([v1,v2])
            return l12, v12, Lambda, f1, f2

        # if not isotropic we can use the analytic results
        d = np.sqrt(4*ltp*lpt + (ltt -lpp)**2)
        l12 = np.array([0.5*(ltt + lpp + d),  0.5*(ltt + lpp - d) ])

        f1 = -2 * ltp / (ltt - lpp - d)
        f2 =  2 * lpt / (ltt - lpp - d)
        f1[~np.isfinite(f1)] = 0
        f2[~np.isfinite(f2)] = 0

        # define v1 and v2 in spherical basis
        v1 = np.array([np.zeros(f1.shape), f1 , np.ones(f1.shape) ])
        v2 = np.array([np.zeros(f1.shape), np.ones(f2.shape)  , f2])
        v12 = np.array([v1,v2])
        return (l12, v12, Lambda, f1, f2)

    @staticmethod
    def ABmn(m,n,v12,theta,phi):
        v1 = v12[0]
        v2 = v12[1]
        mmn,nmn,_ = AmatrixGenerator.mnp_mn(m,n,theta,phi)
        Amn  = -np.einsum('i...,i...->...', v2,nmn)
        Bmn  =  np.einsum('i...,i...->...', v1,nmn)
        Apmn = -np.einsum('i...,i...->...', v1,mmn)
        Bpmn =  np.einsum('i...,i...->...', v2,mmn)

        return (Amn,Bmn,Apmn,Bpmn)

    @staticmethod
    def XYeh_mn(omega,eps,mu,m,n,T,P,W,psurf_coords):
        '''
        Parameters
        ----------
        omega : Optical frequency
        eps : Electric permittivity matrix (3x3) in cartesian basis
        mu : Magnetic permeability (scalar)
        m,n : mode indices  |m| <= n
        T,P : spherical polar angle arrays over which angular spectrum integration to be perform
        W : integration weights corresponding to integration over sphere. NB/ sum(W) should = 4*pi so include the correct Jacobian
        Xs,Ys,Zs : arrays of points to evaluate the modes

        Returns
        -------
        Xe_mn, Ye_mn, Xh_mn, Yh_mn: arrays of vector modes for electric and magnetic fields

        '''
        Xs,Ys,Zs = psurf_coords[0],psurf_coords[1],psurf_coords[2]
        Xe_mn = np.array([np.zeros(Xs.shape),np.zeros(Xs.shape),np.zeros(Xs.shape)],dtype=complex)
        Ye_mn = np.array([np.zeros(Xs.shape),np.zeros(Xs.shape),np.zeros(Xs.shape)],dtype=complex)
        Xh_mn = np.array([np.zeros(Xs.shape),np.zeros(Xs.shape),np.zeros(Xs.shape)],dtype=complex)
        Yh_mn = np.array([np.zeros(Xs.shape),np.zeros(Xs.shape),np.zeros(Xs.shape)],dtype=complex)

        # find eigenpolarisations for anisotropy
        l12, v12, Lambda, f1, f2 = AmatrixGenerator.find_eigenstates(eps,T,P)
        lrt = Lambda[0,1]
        lrp = Lambda[0,2]
        prefac = 1/(1-f1*f2) * W * np.sin(T) # include integration weights and Jacobian

        # eigenpolarisation wavenumbers and wavevectors in cartesian basis
        k1 = omega * np.sqrt(mu / l12[0])
        k2 = omega * np.sqrt(mu / l12[1])
        k1vc = AmatrixGenerator.sph2cart_comp_vec(np.array([k1,np.zeros(T.shape),np.zeros(T.shape)]),T,P)
        k2vc = AmatrixGenerator.sph2cart_comp_vec(np.array([k2,np.zeros(T.shape),np.zeros(T.shape)]),T,P)

        # construct w polarisation vectors in spherical basis
        w1e = l12[0] * v12[0] + np.array([ lrt * f1 + lrp  , np.zeros(T.shape) , np.zeros(T.shape) ])
        w2e = l12[1] * v12[1] + np.array([ lrt + f2 * lrp  , np.zeros(T.shape) , np.zeros(T.shape) ])
        w1h = (omega / k1 ) * np.array([ np.zeros(T.shape) , - np.ones(T.shape) ,  f1 ])
        w2h = (omega / k2 ) * np.array([ np.zeros(T.shape) , - f2 ,  np.ones(T.shape) ])

        # convert w vectors to cartesian basis
        w1ec = AmatrixGenerator.sph2cart_comp_vec(w1e,T,P)
        w2ec = AmatrixGenerator.sph2cart_comp_vec(w2e,T,P)
        w1hc = AmatrixGenerator.sph2cart_comp_vec(w1h,T,P)
        w2hc = AmatrixGenerator.sph2cart_comp_vec(w2h,T,P)

        Amn,Bmn,Apmn,Bpmn = AmatrixGenerator.ABmn(m,n,v12,T,P)

        ranges = [range(nn) for nn in Xs.shape]
        for posinds in itertools.product(*ranges):
            xs = Xs[posinds]
            ys = Ys[posinds]
            zs = Zs[posinds]
            expk1r = prefac * np.exp(1j * (xs * k1vc[0] + ys * k1vc[1] + zs * k1vc[2]))
            expk2r = prefac * np.exp(1j * (xs * k2vc[0] + ys * k2vc[1] + zs * k2vc[2]))

            Amn_expk1r  = Amn  * expk1r
            Apmn_expk2r = Apmn * expk2r
            Bpmn_expk1r = Bpmn * expk1r
            Bmn_expk2r  = Bmn  * expk2r

            for jj in range(3):
                jjinds = (jj,) + posinds
                Xe_mn[jjinds] = np.sum(w1ec[jj] * Amn_expk1r  + w2ec[jj] * Bmn_expk2r)
                Ye_mn[jjinds] = np.sum(w1ec[jj] * Bpmn_expk1r + w2ec[jj] * Apmn_expk2r)
                Xh_mn[jjinds] = np.sum(w1hc[jj] * Amn_expk1r  + w2hc[jj] * Bmn_expk2r)
                Yh_mn[jjinds] = np.sum(w1hc[jj] * Bpmn_expk1r + w2hc[jj] * Apmn_expk2r)

        # remove nans
        Xe_mn[~np.isfinite(Xe_mn)] = 0
        Ye_mn[~np.isfinite(Ye_mn)] = 0
        Xh_mn[~np.isfinite(Xh_mn)] = 0
        Yh_mn[~np.isfinite(Yh_mn)] = 0
        return (Xe_mn, Ye_mn, Xh_mn, Yh_mn)

    #%% --------------- T matrix calculation -----------
    @staticmethod
    def WignerD(l,alpha,beta,gamma):
        '''
        Ported from Archontis Politis' MATLAB wignerD implementation by M.R.Foreman

        WIGNERD Returns the Wigner D-matrix and d-matrix for SH rotation

           WIGNERD computes the rotation matrices for rotation of functions in the
           spherical harmonic domain, as given directly by Wigner for complex SHs.
           Since it much faster to compute the rotation matrices ith recursive
           formulas, these are included here mostly for comparison.

           Main reference:
               D. A. Varshalovich, A. N. Moskalev, and V. K. Khersonskii.
               Quantum theory of angular momentum. World Scientific Pub., 1988.
               p.77 - 4.3(5)

           Inputs: l degree (or band) index
                   alpha, beta, gamma (z, y', z'')-rotation Euler angles

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

           Archontis Politis, 6/5/2015
           archontis.politis@aalto.fi

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         analytic expressions of small-d for order l = 1, 2, included for
         comparison (taken from Varshalovich etal., "Quantum Theory of Angular
         Momentum", 1988, pg.119
         note that the row and column order is inverted from -m:m compared to the
         book (m:-1:-m)

        '''
        # precompute factorials
        fctrl = np.array([factorial(kk) for kk in range(2*l+1)])

        # if beta close to zero force small d to identity matrix
        Dl = np.zeros((2*l+1,2*l+1),dtype='complex')
        if beta <= np.finfo(np.float64).eps:
            dl = np.eye(2*l+1)
            Dl = np.diag( np.exp(-1j*(alpha+gamma)*np.array(range(-l,l+1))))
        else:
            dl = np.zeros((2*l+1,2*l+1),dtype='complex')
            for m in range(-l,l+1):
                for n in range(-l,l+1):
                    # Varshalovich, Eq.4.3.1(5)
                    k = np.array(range(max(0,n-m),min(l+n,l-m)+1))
                    m1_t = [(-1)**kk for kk in k]
                    # print('\n n = {}, m = {}'.format(n,m))
                    # print(np.array([[l+n,l-n,l+m,l-m,l+n-k,l-m-k,k+m-n,k]]).flatten())
                    # print("fctrl is")
                    # print([fctrl[l+n-k] ,fctrl[l-m-k] , fctrl[k+m-n] , fctrl[k]])
                    fact_t = np.sqrt( fctrl[l+n] * fctrl[l-n] * fctrl[l+m] * fctrl[l-m] ) / \
                                    ( fctrl[l+n-k] * fctrl[l-m-k] * fctrl[k+m-n] * fctrl[k])
                    cos_beta = np.cos(beta/2)**(2*l+n-m-2*k)
                    sin_beta = np.sin(beta/2)**(2*k+m-n)
                    d_l_mn = (-1)**(m-n) * np.sum(m1_t * fact_t * cos_beta * sin_beta)
                    dl[n+l,m+l] = d_l_mn
                    Dl[n+l,m+l] = np.exp(-1j*alpha*m)*d_l_mn*np.exp(-1j*gamma*n)

        # Interestingly, if the Wigner-D are used directly they result in an active
        # body rotation , instead of rotation of the coordinate system (!). Since
        # most operations here are defined in terms of coordinate system rotations,
        # the proper result is given by the conjugate of the Wigner-D. Compare for
        # example with the rotation matrices using the recursive formula function.
        # This is also the same as using e^(ima) * d^l_mn(b) * e^(ing) for the
        # entries of the Wigner-D, something that can be found in the literature as
        # well as the above formula.
        # Compute rotation matrices up to l
        Rl = np.conj(Dl)

        return Dl, dl, Rl


    @staticmethod
    def d1(bta):
        d1 = np.array([ [ (1+np.cos(bta))/2       ,  -np.sin(bta)/np.sqrt(2)  ,  (1-np.cos(bta))/2      ],
                        [ np.sin(bta)/np.sqrt(2)  ,  np.cos(bta)              , -np.sin(bta)/np.sqrt(2) ],
                        [ (1-np.cos(bta))/2       ,  np.sin(bta)/np.sqrt(2)   , (1+np.cos(bta))/2       ]  ])
        return d1

    @staticmethod
    def d2(bta):
        d2 = np.array([ [ (1+np.cos(bta))**2/4          ,  -(1+np.cos(bta))*np.sin(bta)/2        ,  np.sqrt(6)*(np.sin(bta))**2/4         ,   -(1-np.cos(bta))*np.sin(bta)/2         ,   (1-np.cos(bta))**2/4             ],
                        [ (1+np.cos(bta))*np.sin(bta)/2 ,  (1+np.cos(bta))*(2*np.cos(bta)-1)/2   ,  -np.sqrt(3/2)*np.cos(bta)*np.sin(bta) ,   (1-np.cos(bta))*(2*np.cos(bta)+1)/2    ,   -(1-np.cos(bta))*np.sin(bta)/2   ],
                        [ np.sqrt(6)*(np.sin(bta))**2/4 ,  np.sqrt(3/2)*np.sin(bta)*np.cos(bta)  ,  3*(np.cos(bta))**2/2-1/2              ,   -np.sqrt(3/2)*np.cos(bta)*np.sin(bta)  ,   np.sqrt(6)*(np.sin(bta))**2/4    ],
                        [ (1-np.cos(bta))*np.sin(bta)/2 ,  (1-np.cos(bta))*(2*np.cos(bta)+1)/2   ,  np.sqrt(3/2)*np.cos(bta)*np.sin(bta)  ,   (1+np.cos(bta))*(2*np.cos(bta)-1)/2    ,   -(1+np.cos(bta))*np.sin(bta)/2   ],
                        [ (1-np.cos(bta))**2/4          ,  (1-np.cos(bta))*np.sin(bta)/2         ,  np.sqrt(6)*(np.sin(bta))**2/4         ,   (1+np.cos(bta))*np.sin(bta)/2          ,   (1+np.cos(bta))**2/4             ]])
        return d2

    @staticmethod
    def WignerD_rot_mat(alpha,beta,gamma,lmin,lmax):
        ninds = range(AmatrixGenerator.lm_2_n_all(lmin,-lmin),AmatrixGenerator.lm_2_n_all(lmax,lmax) + 1)
        ll = [AmatrixGenerator.n_2_lm_all(nn)[0] for nn in ninds]
        [L1,L2] = np.meshgrid(ll,ll)

        D = np.zeros((len(ninds),len(ninds)),dtype='complex')
        for l in range(lmin,lmax+1):
            Dl,_,_ = AmatrixGenerator.WignerD(l,alpha,beta,gamma)
            D[np.logical_and((L1 == l) , (L2 == l))] = Dl.flatten()
        return D

    @staticmethod
    def calc_Q1Q3(nmax,omega,eps_s,eps_p,mu,psurf_coords,psurf_weights,psurf_norms,Nt,Np):
        '''
        Calculate Q1 and Q3 matrices required to calculate the T matrix of an anisotropic particle
        Parameters
        ----------
        nmax : maximum polar mode index
        omega : optical frequency
        eps_s : electric permittivity (scalar) of isotropic host medium
        eps_p : electric permittivity (3x3 tensor) of nanoparticle
        mu : magnetic permeability (host and nanoparticle assumed to have the same permeability)
        psurf_coords : numpy array of X,Y,Z coordinates defining nanoparticle surface
        psurf_weights : numpy array of integration weights
        psurf_norms : numpy array of NX,NY,NZ defining outward surface normal to nanoparticle surface.
        Nt : number of ordinates to use in the polar direction for integration of angular spectrum
             used to find internal modes
        Np : number of ordinates to use in the azimuthal direction for integration of angular spectrum
             used to find internal modes

        '''
        k_s = omega * np.sqrt(eps_s*mu)

        # define angular coordinates for angular spectrum caclulation of X and Y modes
        as_coords,as_weights,_ = AmatrixGenerator.define_sphere(1,Nt,Np)
        P,T,_ = AmatrixGenerator.cart2sph(as_coords[0],as_coords[1],as_coords[2])

        Ps,Ts,Rs    = AmatrixGenerator.cart2sph(psurf_coords[0],psurf_coords[1],psurf_coords[2])

        # precalculate necessary modes
        nummodes = AmatrixGenerator.lm_2_n_all(nmax,nmax) + 1
        Xe_mn = np.squeeze(np.zeros( (nummodes,3) + psurf_coords[0].shape,dtype=complex) )
        Ye_mn = np.squeeze(np.zeros( (nummodes,3) + psurf_coords[0].shape,dtype=complex) )
        Xh_mn = np.squeeze(np.zeros( (nummodes,3) + psurf_coords[0].shape,dtype=complex) )
        Yh_mn = np.squeeze(np.zeros( (nummodes,3) + psurf_coords[0].shape,dtype=complex) )
        M1_mn = np.squeeze(np.zeros( (nummodes,3) + psurf_coords[0].shape,dtype=complex) )
        N1_mn = np.squeeze(np.zeros( (nummodes,3) + psurf_coords[0].shape,dtype=complex) )
        M3_mn = np.squeeze(np.zeros( (nummodes,3) + psurf_coords[0].shape,dtype=complex) )
        N3_mn = np.squeeze(np.zeros( (nummodes,3) + psurf_coords[0].shape,dtype=complex) )

        for jj in range(1,nummodes): # don't include multipole
            n,m = AmatrixGenerator.n_2_lm_all(jj)
            AmatrixGenerator.update_progress(jj/nummodes, "Pre-calculating modes: (n,m) = ({},{})".format(n,m))
            Xe_mn[jj],Ye_mn[jj],Xh_mn[jj],Yh_mn[jj] =\
                    AmatrixGenerator.XYeh_mn(omega,eps_p,mu,m,n,T,P,as_weights,psurf_coords)

            M3_mn[jj],N3_mn[jj] =  AmatrixGenerator.MN_lmn(k_s,3,m,n,Rs,Ts,Ps)
            M1_mn[jj],N1_mn[jj] =  AmatrixGenerator.MN_lmn(k_s,1,m,n,Rs,Ts,Ps)

        Q1_11 = np.zeros((nummodes,nummodes),dtype=complex)
        Q1_12 = np.zeros((nummodes,nummodes),dtype=complex)
        Q1_21 = np.zeros((nummodes,nummodes),dtype=complex)
        Q1_22 = np.zeros((nummodes,nummodes),dtype=complex)
        Q3_11 = np.zeros((nummodes,nummodes),dtype=complex)
        Q3_12 = np.zeros((nummodes,nummodes),dtype=complex)
        Q3_21 = np.zeros((nummodes,nummodes),dtype=complex)
        Q3_22 = np.zeros((nummodes,nummodes),dtype=complex)

        ksoverwmu = k_s / (omega*mu)
        fdg1 = 1#/k_s ???????? ###################################################
        for jj1 in range(1,nummodes):
            n1,m1 = AmatrixGenerator.n_2_lm_all(jj1)
            jj1minusm = AmatrixGenerator.lm_2_n_all(n1,-m1) # since projections required -m modes
            for jj2 in range(1,nummodes): # -n to n
                n2,m2 = AmatrixGenerator.n_2_lm_all(jj2)

                # update_progress((nummodes*jj1+jj2)/nummodes**2, "Calculating Q overlap integrals")
                q1_11kern = fdg1*np.einsum('j...,j...->...',np.cross(psurf_norms,Xh_mn[jj2],axis=0) , M1_mn[jj1minusm])  \
                                - 1j*ksoverwmu * np.einsum('j...,j...->...',np.cross(psurf_norms,Xe_mn[jj2],axis=0), N1_mn[jj1minusm])
                q1_12kern = fdg1*np.einsum('j...,j...->...',np.cross(psurf_norms,Yh_mn[jj2],axis=0) , M1_mn[jj1minusm])  \
                                - 1j*ksoverwmu * np.einsum('j...,j...->...',np.cross(psurf_norms,Ye_mn[jj2],axis=0), N1_mn[jj1minusm])
                q1_21kern = fdg1*np.einsum('j...,j...->...',np.cross(psurf_norms,Xh_mn[jj2],axis=0) , N1_mn[jj1minusm])  \
                                - 1j*ksoverwmu * np.einsum('j...,j...->...',np.cross(psurf_norms,Xe_mn[jj2],axis=0), M1_mn[jj1minusm])
                q1_22kern = fdg1*np.einsum('j...,j...->...',np.cross(psurf_norms,Yh_mn[jj2],axis=0) , N1_mn[jj1minusm])  \
                                - 1j*ksoverwmu * np.einsum('j...,j...->...',np.cross(psurf_norms,Ye_mn[jj2],axis=0), M1_mn[jj1minusm])

                q3_11kern = fdg1*np.einsum('j...,j...->...',np.cross(psurf_norms,Xh_mn[jj2],axis=0) , M3_mn[jj1minusm])  \
                                - 1j*ksoverwmu * np.einsum('j...,j...->...',np.cross(psurf_norms,Xe_mn[jj2],axis=0), N3_mn[jj1minusm])
                q3_12kern = fdg1*np.einsum('j...,j...->...',np.cross(psurf_norms,Yh_mn[jj2],axis=0) , M3_mn[jj1minusm])  \
                                - 1j*ksoverwmu * np.einsum('j...,j...->...',np.cross(psurf_norms,Ye_mn[jj2],axis=0), N3_mn[jj1minusm])
                q3_21kern = fdg1*np.einsum('j...,j...->...',np.cross(psurf_norms,Xh_mn[jj2],axis=0) , N3_mn[jj1minusm])  \
                                - 1j*ksoverwmu * np.einsum('j...,j...->...',np.cross(psurf_norms,Xe_mn[jj2],axis=0), M3_mn[jj1minusm])
                q3_22kern = fdg1*np.einsum('j...,j...->...',np.cross(psurf_norms,Yh_mn[jj2],axis=0) , N3_mn[jj1minusm])  \
                                - 1j*ksoverwmu * np.einsum('j...,j...->...',np.cross(psurf_norms,Ye_mn[jj2],axis=0), M3_mn[jj1minusm])

                Q1_11[jj1,jj2] = k_s*omega*mu*(-1)**m1*np.sum(q1_11kern * psurf_weights)
                Q1_12[jj1,jj2] = k_s*omega*mu*(-1)**m1*np.sum(q1_12kern * psurf_weights)
                Q1_21[jj1,jj2] = k_s*omega*mu*(-1)**m1*np.sum(q1_21kern * psurf_weights)
                Q1_22[jj1,jj2] = k_s*omega*mu*(-1)**m1*np.sum(q1_22kern * psurf_weights)
                Q3_11[jj1,jj2] = k_s*omega*mu*(-1)**m1*np.sum(q3_11kern * psurf_weights)
                Q3_12[jj1,jj2] = k_s*omega*mu*(-1)**m1*np.sum(q3_12kern * psurf_weights)
                Q3_21[jj1,jj2] = k_s*omega*mu*(-1)**m1*np.sum(q3_21kern * psurf_weights)
                Q3_22[jj1,jj2] = k_s*omega*mu*(-1)**m1*np.sum(q3_22kern * psurf_weights)

        # remove monopole terms and combine
        Q1 = np.concatenate( (np.concatenate((Q1_11[1:,1:],Q1_21[1:,1:]),axis=0), \
                              np.concatenate((Q1_12[1:,1:],Q1_22[1:,1:]),axis=0)), axis=1)
        Q3 = np.concatenate( (np.concatenate((Q3_11[1:,1:],Q3_21[1:,1:]),axis=0), \
                              np.concatenate((Q3_12[1:,1:],Q3_22[1:,1:]),axis=0)), axis=1)
        return Q1,Q3

    @staticmethod
    def calc_Tmatrix_from_Q(Q1,Q3):
        return -Q3 @ np.linalg.inv(Q1)

    @staticmethod
    def calc_Tmatrix(params):
        nmax = params['nmax']
        omega = params['omega']
        eps_s = params['eps_s']
        eps_p = params['eps_p']
        mu = params['mu']
        psurf_coords = params['psurf_coords']
        psurf_weights = params['psurf_weights']
        psurf_norms = params['psurf_norms']
        Nt = params['Nt']
        Np = params['Np']

        Q1,Q3 = AmatrixGenerator.calc_Q1Q3(nmax,omega,eps_s,eps_p,mu,psurf_coords,psurf_weights,psurf_norms,Nt,Np)
        Tmat =  AmatrixGenerator.calc_Tmatrix_from_Q(Q1,Q3)

        if ('rotational_average' in params.keys()):
            rot_avg = params['rotational_average']
        else:
            rot_avg = False

        if rot_avg :
            alpha = params['euler_alpha']
            beta  = params['euler_beta']
            gamma = params['euler_gamma']
            pdf   = params['angular_pdf']
            RTMAT = AmatrixGenerator.average_Tmatrix(Tmat,nmax,alpha,beta,gamma,pdf)
        else:
            alpha = params['euler_alpha']
            beta  = params['euler_beta']
            gamma = params['euler_gamma']

            RTMAT = AmatrixGenerator.rotate_Tmatrix(Tmat,nmax,alpha,beta,gamma)

        return RTMAT

    @staticmethod
    def rotate_Tmatrix(Tmat,nmax,alpha,beta,gamma):
        DR = AmatrixGenerator.WignerD_rot_mat(alpha,beta,gamma,1,nmax)
        iDR = np.linalg.inv(DR)

        RTmat = np.zeros(Tmat.shape,dtype='complex')
        dimt = Tmat.shape[0]
        nt = int(dimt/2)

        RTmat[0:nt,0:nt] = DR @ Tmat[0:nt,0:nt] @ iDR
        RTmat[0:nt,nt:]  = DR @ Tmat[0:nt,nt:]  @ iDR
        RTmat[nt:,0:nt]  = DR @ Tmat[nt:,0:nt]  @ iDR
        RTmat[nt:,nt:]   = DR @ Tmat[nt:,nt:]   @ iDR

        return RTmat

    @staticmethod
    def average_Tmatrix(Tmat,nmax,alpha,beta,gamma,pdf):
        '''
        Parameters
        ----------
        Tmat : Tmatrix in reference frame
        nmax : maximum multipole order
        alpha, beta, gamma : meshgrid of Eular angles corresponding to odf
        pdf : probability density function for orientational average

        Returns
        -------
        avgTmat : rotationally averaged T matrix

        '''
        avgTmat = np.zeros(Tmat.shape,dtype='complex')
        dimt = Tmat.shape[0]
        nt = int(dimt/2)

        ranges = [range(nn) for nn in alpha.shape]
        for posinds in itertools.product(*ranges):
            a = alpha[posinds]
            b = beta[posinds]
            g = gamma[posinds]
            f = pdf[posinds]
            DR = AmatrixGenerator.WignerD_rot_mat(a,b,g,1,nmax)
            iDR = np.linalg.inv(DR)

            avgTmat[0:nt,0:nt] += f*(DR @ Tmat[0:nt,0:nt] @ iDR)
            avgTmat[0:nt,nt:]  += f*(DR @ Tmat[0:nt,nt:]  @ iDR)
            avgTmat[nt:,0:nt]  += f*(DR @ Tmat[nt:,0:nt]  @ iDR)
            avgTmat[nt:,nt:]   += f*(DR @ Tmat[nt:,nt:]   @ iDR)

        return avgTmat/np.sum(f)



    #%% ---------------- homogeneous sphere Mie scattering coefficients--------
    @staticmethod
    def riccati_psi_n(n,z):
        return np.sqrt(np.pi * z / 2) * jv(n + 1/2,z)

    @staticmethod
    def driccati_psi_n(n,z):
        return np.sqrt(np.pi / (2*z)) * (n+1) * jv(n+1/2,z) - np.sqrt(np.pi * z / 2) * jv(n+3/2,z)

    @staticmethod
    def riccati_xi_n(n,z):
        return np.sqrt(np.pi * z / 2) * hankel1(n + 1/2,z)

    @staticmethod
    def driccati_xi_n(n,z):
        return np.sqrt(np.pi / (2*z)) * (n+1) * hankel1(n+1/2,z) - np.sqrt(np.pi * z / 2) * hankel1(n+3/2,z)

    @staticmethod
    def calc_eta_zeta_mn(n,k1,k2,r):
        z1 = k1*r
        z2 = k2*r
        m = k2/k1

        an = ( m*AmatrixGenerator.riccati_psi_n(n,z2) * AmatrixGenerator.driccati_psi_n(n,z1) - AmatrixGenerator.driccati_psi_n(n,z2) * AmatrixGenerator.riccati_psi_n(n,z1) ) /  \
             ( m*AmatrixGenerator.riccati_psi_n(n,z2) * AmatrixGenerator.driccati_xi_n(n,z1)  - AmatrixGenerator.driccati_psi_n(n,z2)  * AmatrixGenerator.riccati_xi_n(n,z1) )
        bn = ( m*AmatrixGenerator.driccati_psi_n(n,z2) * AmatrixGenerator.riccati_psi_n(n,z1) - AmatrixGenerator.riccati_psi_n(n,z2) * AmatrixGenerator.driccati_psi_n(n,z1) ) /  \
             ( m*AmatrixGenerator.driccati_psi_n(n,z2) * AmatrixGenerator.riccati_xi_n(n,z1)  - AmatrixGenerator.driccati_xi_n(n,z1)  * AmatrixGenerator.riccati_psi_n(n,z2) )

        return an, bn

    @staticmethod
    def calc_Tmatrix_sphere(nmax,k1,k2,r):
        nummodes = AmatrixGenerator.lm_2_n_all(nmax,nmax) + 1
        N = np.zeros((nummodes,nummodes),dtype=complex)
        Z = np.zeros((nummodes,nummodes),dtype=complex)

        for jj1 in range(1,nummodes):
            n1,m1 = AmatrixGenerator.n_2_lm_all(jj1)
            an,bn = AmatrixGenerator.calc_eta_zeta_mn(n1,k1,k2,r)
            N[jj1,jj1] = -an
            Z[jj1,jj1] = -bn

        T = np.concatenate( (np.concatenate((Z[1:,1:],0*N[1:,1:]),axis=0), \
                             np.concatenate((0*Z[1:,1:],N[1:,1:]),axis=0)), axis=1)

        return T

    @staticmethod
    def calc_Amat_from_Tmat(k_s,T,ninc,nsca):
        # calculate angles
        pinc, tinc, _ = AmatrixGenerator.cart2sph(ninc[0],ninc[1],ninc[2])
        psca, tsca, _ = AmatrixGenerator.cart2sph(nsca[0],nsca[1],nsca[2])

        # ensure they coordinates are numpy arrays not just floats
        if isinstance(pinc,np.float64):
            pinc = np.array([pinc])
            tinc = np.array([tinc])
            psca = np.array([psca])
            tsca = np.array([tsca])

        dimt = T.shape[0]
        nt = int(dimt/2)
        nmax,_ = AmatrixGenerator.n_2_lm_all(nt) # add one since we assume monopole has been removed
        nummodes = AmatrixGenerator.lm_2_n_all(nmax,nmax) + 1

        T11 = T[0:nt,0:nt]
        T12 = T[0:nt,nt:]
        T21 = T[nt:,0:nt]
        T22 = T[nt:,nt:]

        s11 = 0
        s12 = 0
        s21 = 0
        s22 = 0

        # precalculate factorial terms
        prefacs = np.zeros(nummodes)
        nnarray = np.zeros(nummodes)
        mmarray = np.zeros(nummodes)
        for jji in range(1,nummodes):
            [ni,mi] = AmatrixGenerator.n_2_lm_all(jji)
            nnarray[jji] = ni
            mmarray[jji] = mi
            prefacs[jji] = np.sqrt(factorial(ni - np.abs(mi)) / factorial(ni + np.abs(mi)))

        for jji in range(1,nummodes):
            ni = nnarray[jji]
            mi = mmarray[jji]
            prefi = prefacs[jji]

            pi_inc,tau_inc,_ = AmatrixGenerator.pi_tau_mn(mi,ni,tinc)
            pi_inc = prefi * mi * pi_inc
            tau_inc = prefi * tau_inc
            exp_inc = np.exp(-1j*mi*pinc)

            for jjs in range(1,nummodes):
                ns = nnarray[jjs]
                ms = mmarray[jjs]
                prefs = prefacs[jjs]

                T11mnmn = T11[jjs-1,jji-1]
                T12mnmn = T12[jjs-1,jji-1]
                T21mnmn = T21[jjs-1,jji-1]
                T22mnmn = T22[jjs-1,jji-1]

                pi_sca,tau_sca,_ = AmatrixGenerator.pi_tau_mn(ms,ns,tsca)
                pi_sca = prefs * ms * pi_sca
                tau_sca = prefs * tau_sca
                exp_sca = np.exp(1j*ms*psca)

                alphamnmn = (-1)**(np.abs(ms*0)+np.abs(mi*0)) * (1j)**(ni - ns - 1)  * np.sqrt( (2*ni+1) * (2*ns+1) / (ni*ns*(ni+1)*(ns+1)) )

                s11 +=     alphamnmn * (T11mnmn * pi_sca  * pi_inc  + T21mnmn * tau_sca * pi_inc
                                      + T12mnmn * pi_sca  * tau_inc + T22mnmn * tau_sca * tau_inc ) * exp_sca * exp_inc
                s12 += -1j *alphamnmn * (T11mnmn * pi_sca  * tau_inc + T21mnmn * tau_sca * tau_inc
                                      + T12mnmn * pi_sca  * pi_inc  + T22mnmn * tau_sca * pi_inc ) * exp_sca * exp_inc
                s21 +=  1j *alphamnmn * (T11mnmn * tau_sca * pi_inc  + T21mnmn * pi_sca  * pi_inc
                                      + T12mnmn * tau_sca * tau_inc + T22mnmn * pi_sca  * tau_inc ) * exp_sca * exp_inc
                s22 +=     alphamnmn * (T11mnmn * tau_sca * tau_inc + T21mnmn * pi_sca  * tau_inc
                                      + T12mnmn * tau_sca * pi_inc  + T22mnmn * pi_sca  * pi_inc ) * exp_sca * exp_inc

        Amat = 1j*np.squeeze(np.array([[s11,s12],[s21,s22]]))
        return Amat

    def get_A(self,k1, k2, params, N_lim = 20):
        '''
        Parameters
        ----------
        k1 : array
            Wave vector of incident plane wave (e.g. np.array([0,0,1])).
        k2 : array
            Wave vector of scattered plane wave (e.g. np.array([0,0,1])).
        params : dict
            Dictionary containing particle parameters

        N_lim : int, optional
            Number of terms used in Mie theory sums.

        Returns
        -------
        A : array
            2x2 matrix that describes scattering from wave vector k1 to k2.

        '''

        if 'particle_type' in params.keys():
            Amat = self.get_A_legacy(k1, k2, params, N_lim)
        else:

            # check if Tmatrix has been previously calculated
            if hasattr(self,'tmatrix'):
                # check if parameters have changed
                if params == self.params:
                    # nothing changed so we calculate A directly from existing T matrix
                    Amat = self.calc_Amat_from_Tmat(self.params['k_s'],self.tmatrix,k1,k2)
                    return Amat
                else:
                    print('Input particle parameters have changed. Reinitialising Tmatrix...')

            else:
                print('Initialising Tmatrix...')

            # if we got this far we have to calcaulte Tmatrix and save parameters
            self.params = params
            k_s = params['k_s']
            Tmat = self.calc_Tmatrix(params)

            self.tmatrix = Tmat
            # Tmat = calc_Tmatrix_sphere(params['nmax'],params['k_s'], params['k_p'],params['radius'])

            Amat = self.calc_Amat_from_Tmat(k_s,Tmat,k1,k2)
        return Amat

    @staticmethod
    def empty_Tmatrix_params():
        empty_params = {
           'nmax': None,   # maximum polar index for spherical harmonics
           'omega':None,   # optical frequency
           'eps_s':None,   # permitivitty of surrounding medium
           'eps_p':None,   # permitivitty tensor of particle
           'mu':None,      # magnetic permeability of particle and surroundings
           'psurf_coords':None, # cartesian coordinates defining particle surface
           'psurf_weights':None, # integration weights across particle surface
           'psurf_norms':None,   # cartesian normal to particle surface
           'Nt':None,      # number of polar angles used in angular spectrum integral
           'Np':None,      # number of azimuthal angles used in angular spectrum integral
           'k_s':None,     # wavenumber in surround medium
           'euler_alpha': None,  # euler angle alpha defining particle rotation(s) - multiple values used when rotational averaging performed
           'euler_beta': None,   # euler angle beta defining particle rotation(s)
           'euler_gamma': None,  # euler angle gamma defining particle rotation(s)
           'rotational_average': False,  # flag to perform rotational averaging
           'angular_pdf': None,  #  pdf corresponding to meshgrid of euler angles used for rotational averaging
           }

        return empty_params


    #%% ---------------- Niall legacy functions
    @staticmethod
    def pi(n, mu):
        if n == 0:
            return 0
        elif n == 1:
            return 1
        else:
            return ( (2*n-1)/(n-1) * mu*AmatrixGenerator.pi(n-1, mu) - n/(n-1)*AmatrixGenerator.pi(n-2,mu) )

    @staticmethod
    def tau(n, mu):
        return ( n*mu*AmatrixGenerator.pi(n, mu) - (n+1)*AmatrixGenerator.pi(n-1, mu) )

    @staticmethod
    def mie_cross_section(x, m, k, N_lim = 20):
            psi_x   = sp.special.riccati_jn(N_lim, x)
            psi_mx  = sp.special.riccati_jn(N_lim, m*x)
            phi_x = sp.special.riccati_yn(N_lim, x)
            xi_x  = [psi + 1j*phi for psi, phi in zip(psi_x, phi_x)]

            C_sca = 0
            C_ext = 0
            for n in range(1, N_lim):
                a = (m*psi_mx[0][n]*psi_x[1][n] - psi_x[0][n]*psi_mx[1][n]) / (m*psi_mx[0][n]*xi_x[1][n] - xi_x[0][n]*psi_mx[1][n])
                b = (psi_mx[0][n]*psi_x[1][n] - m*psi_x[0][n]*psi_mx[1][n]) / (psi_mx[0][n]*xi_x[1][n] - m*xi_x[0][n]*psi_mx[1][n])
                C_sca = C_sca + (2*n + 1)*(np.absolute(a)**2 + np.absolute(b)**2)
                C_ext = C_ext + (2*n + 1)*np.real(a+b)

            C_sca = 2*np.pi * C_sca / k**2
            C_ext = 2*np.pi * C_ext / k**2
            return C_sca, C_ext

    @staticmethod
    def get_angle_plane(v1, v2, n):
        '''
        Find the angle between v1 and v2, oriented in accordance with the conventions used in the RMT paper (see appendix).
        n is normal to v1 and v2.

        Parameters
        ----------
        v1 : array
            Three component vector.
        v2 : array
            Three component vector.
        n : array
            Three component vector.

        Returns
        -------
        alpha : float
            Angle between v1 and v2.

        '''

        # Special cases
        # Clean up numerical bugs
        cosine = np.dot(v1,v2)

        if cosine >= 1 or np.isclose(cosine, 1):
            return 0
        elif cosine <= -1 or np.isclose(cosine, -1):
            return np.pi

        theta = np.arccos(cosine)

        # check if n is in the same direction as v1xv2
        k = np.cross(v1, v2)
        k = k/np.linalg.norm(k)

        alignment = np.dot(k,n)
        if(np.isclose(alignment, 1)):
            alpha = theta
        else:
            alpha = -theta

        return alpha

    @staticmethod
    def get_A_legacy(k1, k2, params, N_lim = 20):
        '''
        Parameters
        ----------
        k1 : array
            Wave vector of incident plane wave (e.g. np.array([0,0,1])).
        k2 : array
            Wave vector of scattered plane wave (e.g. np.array([0,0,1])).
        params : dict
            Dictionary containing size paramtere x, relative refractive index m and possibly a circular birefringence dm
            In the case of chiral particles, m is the mean relative refractive index.
            Also needs to contain 'particle_type', which can be either 'rayleigh', 'mie' or 'chiral'.
        N_lim : int, optional
            Number of terms used in Mie theory sums.

        Returns
        -------
        A : array
            2x2 matrix that describes scattering from wave vector k1 to k2.

        '''

        # Normalize wavevectors
        k1 = k1/np.linalg.norm(k1)
        k2 = k2/np.linalg.norm(k2)

        # Cleans up components close to 0 to avoid certain numerical bugs
        for i, component in enumerate(k1):
            if np.isclose(component, 0):
                k1[i] = 0
        for i, component in enumerate(k2):
            if np.isclose(component, 0):
                k2[i] = 0

        # Unpack dictionary
        x = params['x']
        m = params['m']
        particle_type = params['particle_type']

        if particle_type == 'chiral':
            dm = params['dm']
            mL = m + dm
            mR = m - dm

        # Calculate spherical polar coordinate angles for incident and scattered waves
        theta_i = np.arccos(k1[2])
        phi_i = np.arctan2(k1[1], k1[0])

        theta_s = np.arccos(k2[2])
        phi_s = np.arctan2(k2[1], k2[0])

        cos_theta_scattering = np.dot(k1, k2)

        # Calculate spherical polar coordinate basis vectors
        # If statements deal with special case

        # Incident
        if np.allclose(k1, np.array([0,0,-1])):
            e_theta_i = np.array([1,0,0])
            e_phi_i = np.array([0,-1,0])
        else:
            e_theta_i = np.array([np.cos(theta_i)*np.cos(phi_i), np.cos(theta_i)*np.sin(phi_i), -np.sin(theta_i)])
            e_phi_i = np.array([-np.sin(phi_i), np.cos(phi_i), 0])

        # Scattered
        if np.allclose(k2, np.array([0,0,-1])):
            e_theta_s = np.array([1,0,0])
            e_phi_s = np.array([0,-1,0])
        else:
            e_theta_s = np.array([np.cos(theta_s)*np.cos(phi_s), np.cos(theta_s)*np.sin(phi_s), -np.sin(theta_s)])
            e_phi_s = np.array([-np.sin(phi_s), np.cos(phi_s), 0])

        # Find scattering plane coordiante vectors
        # Deal with special case where k1 and k2 are parallel
        # alpha1 and alpha2 are the rotation angles for going from each plane wave's local coordinate system to the scattering plane
        if np.allclose(k1, k2) or np.allclose(k1, -k2):
            e_par_i = e_theta_i
            e_par_s = e_theta_s
            e_per = e_phi_i
            alpha1 = 0
            alpha2 = 0
        else:
            e_per = np.cross(k1, k2)
            e_per = e_per/np.linalg.norm(e_per)

            e_par_i = np.cross(e_per,k1)
            e_par_i = e_par_i/np.linalg.norm(e_par_i)

            e_par_s = np.cross(e_per,k2)
            e_par_s = e_par_s/np.linalg.norm(e_par_s)

            alpha1 = AmatrixGenerator.get_angle_plane(e_theta_i, e_par_i, k1)
            alpha2 = AmatrixGenerator.get_angle_plane(e_par_s, e_theta_s, k2)

        # T1 and T5 are rotation matrices
        # T2 and T4 are for getting the correct sign convention (accordance with Bohren and Huffman)
        T1 = np.array([[np.cos(alpha1), np.sin(alpha1)],[-np.sin(alpha1), np.cos(alpha1)]])
        T2 = np.array([[1,0],[0,-1]])
        T4 = np.array([[1,0],[0,-1]])
        T5 = np.array([[np.cos(alpha2), np.sin(alpha2)],[-np.sin(alpha2), np.cos(alpha2)]])

        # T3 is the scattering plane matrix

        if particle_type == 'rayleigh':
            a1 = -1j*x**3*(m**2 - 1)/(m**2 + 2)
            S1 = a1*1
            S2 = a1*cos_theta_scattering
            T3 = np.array([[S2, 0],[0, S1]])

        elif particle_type == 'mie':
            mu = np.dot(k1,k2)

            psi_x   = sp.special.riccati_jn(N_lim, x)
            psi_mx  = sp.special.riccati_jn(N_lim, m*x)
            phi_x = sp.special.riccati_yn(N_lim, x)
            xi_x  = [psi + 1j*phi for psi, phi in zip(psi_x, phi_x)]

            S1 = 0
            S2 = 0
            for n in range(1, N_lim):
                a = (m*psi_mx[0][n]*psi_x[1][n] - psi_x[0][n]*psi_mx[1][n]) / (m*psi_mx[0][n]*xi_x[1][n] - xi_x[0][n]*psi_mx[1][n])
                b = (psi_mx[0][n]*psi_x[1][n] - m*psi_x[0][n]*psi_mx[1][n]) / (psi_mx[0][n]*xi_x[1][n] - m*xi_x[0][n]*psi_mx[1][n])
                S1 = S1 + (2*n+1)/(n*(n+1))*(a*AmatrixGenerator.pi(n,mu) + b*AmatrixGenerator.tau(n,mu))
                S2 = S2 + (2*n+1)/(n*(n+1))*(a*AmatrixGenerator.tau(n,mu) + b*AmatrixGenerator.pi(n,mu))

            T3 = np.array([[S2, 0],[0, S1]])

        elif particle_type == 'chiral':
            mu = np.dot(k1,k2)

            psi_x   = sp.special.riccati_jn(N_lim, x)
            psi_mLx  = sp.special.riccati_jn(N_lim, mL*x)
            psi_mRx  = sp.special.riccati_jn(N_lim, mR*x)
            phi_x = sp.special.riccati_yn(N_lim, x)
            xi_x  = [psi + 1j*phi for psi, phi in zip(psi_x, phi_x)]

            S1 = 0
            S2 = 0
            S3 = 0
            for n in range(1, N_lim):
                new_pi = AmatrixGenerator.pi(n,mu)
                new_tau = AmatrixGenerator.tau(n,mu)

                WL = m*psi_mLx[0][n]*xi_x[1][n] - xi_x[0][n]*psi_mLx[1][n]
                WR = m*psi_mRx[0][n]*xi_x[1][n] - xi_x[0][n]*psi_mRx[1][n]
                VL = psi_mLx[0][n]*xi_x[1][n] - m*xi_x[0][n]*psi_mLx[1][n]
                VR = psi_mRx[0][n]*xi_x[1][n] - m*xi_x[0][n]*psi_mRx[1][n]
                AL = m*psi_mLx[0][n]*psi_x[1][n] - psi_x[0][n]*psi_mLx[1][n]
                AR = m*psi_mRx[0][n]*psi_x[1][n] - psi_x[0][n]*psi_mRx[1][n]
                BL = psi_mLx[0][n]*psi_x[1][n] - m*psi_x[0][n]*psi_mLx[1][n]
                BR = psi_mRx[0][n]*psi_x[1][n] - m*psi_x[0][n]*psi_mRx[1][n]

                a = (VR*AL + VL*AR)/(WL*VR + VL*WR)
                b = (WL*BR + WR*BL)/(WL*VR + VL*WR)
                c = 1j*(WR*AL - WL*AR)/(WL*VR + VL*WR)

                S1 = S1 + (2*n+1)/(n*(n+1))*(a*new_pi + b*new_tau)
                S2 = S2 + (2*n+1)/(n*(n+1))*(a*new_tau + b*new_pi)
                S3 = S3 + (2*n+1)/(n*(n+1))*c*(new_pi + new_tau)
            T3 = np.array([[S2, S3],[-S3, S1]])

        T = -T5@T4@T3@T2@T1

        return T

    #%% ---------------- define different surface geoemtries --------

    # calculate the coordinates, integration weightings and outward surface normal
    # for a sphere of radius R. Angular coordinates are sampled uniformly with Nt,Np values
    @staticmethod
    def define_sphere(R,Nt,Np):
        # define coordinates
        x,w=p_roots(Nt)
        t = (x+1)/2* np.pi
        p = np.linspace(0,2*np.pi,Np)
        [P,T] = np.meshgrid(p,t)
        # coords = np.array([P,T,R*np.ones(T.shape)])

        # define integation weights
        wp = np.ones(p.shape)
        wp[0]  = 1/2
        wp[-1]  = 1/2
        wt = w*np.pi/2
        wp = 2*np.pi * wp / np.sum(wp)
        [WP,WT] = np.meshgrid(wp,wt)
        weights = WT * WP * np.sin(T)

        # define surface normal
        X,Y,Z = AmatrixGenerator.sph2cart(P,T,R)
        coords = np.array([X,Y,Z])
        normals = np.array([X/R,Y/R,Z/R])

        return coords,weights,normals
