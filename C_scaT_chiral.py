import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import random_matrix.amplitude_matrix.chiral_sphere as rm

wavelength = 400e-9
n = 100
rri = 1.2
brg = 0.044
m = np.reshape(rri*np.ones((n*(n+1))),(n+1,n))
mrs = m + brg #np.reshape(1.2*np.ones((n*(n+1))),(n+1,n))
mls = m - brg #np.reshape(1.2*np.ones((n*(n+1))),(n+1,n))
k = (2*np.pi)/wavelength

theta = np.linspace(0,np.pi,n)
phi = np.linspace(0,2*np.pi,n+1)
theta_grid,phi_grid = np.meshgrid(theta,phi)


ki_x = np.reshape(np.zeros((n*(n+1))),(n+1,n))
ki_y = np.reshape(np.zeros((n*(n+1))),(n+1,n))
ki_z = np.reshape(np.ones((n*(n+1))),(n+1,n))

ks_z = np.cos(theta_grid)
ks_x = np.sin(theta_grid)*np.cos(phi_grid)
ks_y = np.sin(theta_grid)*np.sin(phi_grid)



size_param = np.linspace(0.01,50,1000)
C_scaT = np.zeros((1000))
radius1 = size_param/k
d_theta = np.pi/(n-1)
d_phi = np.pi/(n)


for i in range(0,1000):
    x = np.reshape((size_param[i]*np.ones((n*(n+1)))),(n+1,n))
    A = rm.get_A(ki_x,ki_y,ki_z,ks_x,ks_y,ks_z,x,m,brg)
    S1 = A[3,:,:]; S2 = A[0,:,:]; S3 = A[1,:,:]; S4 = A[2,:,:]
    T = (np.abs(S2-1j*S3)**2+np.abs(S4-1j*S1)**2)*np.sin(theta_grid)/k**2
    inner_integral = np.trapezoid(T,phi,d_phi,axis=0)
    C_scaT[i] = np.trapezoid(inner_integral,theta,d_theta) / (np.pi*radius1[i]**2)

fig, ax = plt.subplots()
plt.plot(size_param,C_scaT,'b-', label = f"m = {rri} and brg = {brg}")

ax.set_title('$Q_{sca}$ for chiral sphere LCP  \u03BB=400nm')
ax.set_xlabel("x size parameter")
ax.set_ylabel('$Q_{sca}$')
plt.legend()
#fig.savefig("/home/sdutta/code/random-matrix/examples/chi_LCP.png")
   

    
