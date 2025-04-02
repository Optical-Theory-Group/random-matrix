import numpy as np
import matplotlib.pyplot as plt


def an_bn(m, x):

    # Get stopping index for sum
    if isinstance(x, np.float64):
        num_stop = int(np.floor(x + 4.05 * x**0.33333 + 2.0) + 1)
    else:
        num_stop = int(np.max(np.floor(x + 4.05 * x**0.33333 + 2.0) + 1))

    jx_0 = np.sin(x) / x
    jx_1 = np.sin(x) / x**2 - np.cos(x) / x
    jmx_0 = np.sin(m * x) / (m * x)
    jmx_1 = np.sin(m * x) / (m * x) ** 2 - np.cos(m * x) / (m * x)

    djx = ((x**2.0 - 2.0) * np.sin(x) + 2.0 * x * np.cos(x)) / x**3.0
    djmx = (((m * x) ** 2.0 - 2.0) * np.sin(m * x) + 2.0 * m * x * np.cos(m * x)) / (
        m * x
    ) ** 3

    yx_0 = -np.cos(x) / x
    yx_1 = -np.cos(x) / x**2 - np.sin(x) / x
    ymx_0 = -np.cos(m * x) / (m * x)
    ymx_1 = -np.cos(m * x) / (m * x) ** 2 - np.sin(m * x) / (m * x)

    dyx = (2.0 * x * np.sin(x) - (x**2 - 2.0) * np.cos(x)) / x**3

    psix = x * jx_1
    psimx = m * x * jmx_1
    dpsix = x * djx + jx_1
    dpsimx = m * x * djmx + jmx_1
    phix = x * yx_1
    dphix = x * dyx + yx_1
    xix = psix + 1j * phix
    dxix = dpsix + 1j * dphix
    a = (m * psimx * dpsix - psix * dpsimx) / (m * psimx * dxix - xix * dpsimx)
    b = (psimx * dpsix - m * psix * dpsimx) / (psimx * dxix - m * xix * dpsimx)
    sum = (2 * 1 + 1) * (np.abs(a) ** 2 + np.abs(b) ** 2)
    # sum = (2*1+1)*np.real(a+b)
    for n in range(2, num_stop + 1):
        # Update all variables with recurrence relations
        # Bessel functions
        new_jx = (2.0 * n - 1.0) / x * jx_1 - jx_0
        jx_0 = jx_1
        jx_1 = new_jx
        new_jmx = (2.0 * n - 1.0) / (m * x) * jmx_1 - jmx_0
        jmx_0 = jmx_1
        jmx_1 = new_jmx
        new_yx = (2.0 * n - 1.0) / x * yx_1 - yx_0
        yx_0 = yx_1
        yx_1 = new_yx
        new_ymx = (2.0 * n - 1.0) / (m * x) * ymx_1 - ymx_0
        ymx_0 = ymx_1
        ymx_1 = new_ymx

        # Derivatives of bessel funtions
        new_djx = -(n + 1) / x * jx_1 + jx_0
        new_djmx = -(n + 1) / (m * x) * jmx_1 + jmx_0
        new_dyx = -(n + 1) / x * yx_1 + yx_0

        new_psix = x * new_jx
        new_psimx = m * x * new_jmx
        new_dpsix = x * new_djx + new_jx
        new_dpsimx = m * x * new_djmx + new_jmx
        new_phix = x * new_yx
        new_dphix = x * new_dyx + new_yx
        new_xix = new_psix + 1j * new_phix
        new_dxix = new_dpsix + 1j * new_dphix
        # Calculate new S terms
        a = (m * new_psimx * new_dpsix - new_psix * new_dpsimx) / (
            m * new_psimx * new_dxix - new_xix * new_dpsimx
        )
        b = (new_psimx * new_dpsix - m * new_psix * new_dpsimx) / (
            new_psimx * new_dxix - m * new_xix * new_dpsimx
        )
        sum = sum + (2 * n + 1) * (np.abs(a) ** 2 + np.abs(b) ** 2)
        # sum = sum + (2*n+1)*np.real(a+b)

    return sum


size_param = np.linspace(0.01, 50, 1000)
m = 1.2
blue = 400 * 10**-9
k1 = 2 * np.pi / blue
radius1 = size_param / k1
red = 800 * 10**-9
k2 = 2 * np.pi / red
radius2 = size_param / k2
Q_ext1 = np.zeros(1000)
Q_ext2 = np.zeros(1000)

for i in range(0, 1000):
    Q_ext1[i] = (an_bn(m, size_param[i]) * ((2 * np.pi) / k1**2)) / (
        np.pi * radius1[i] ** 2
    )
    # Q_ext2[i] = (an_bn(1.5,size_param[i])*((2*np.pi)/k1**2))/(np.pi*radius1[i]**2)

fig, ax = plt.subplots()
plt.plot(size_param, Q_ext1, "b-", label=f"m ={m}")

# plt.plot(size_param,Q_ext2,'r-', label = 'm = 1.5')

ax.set_title("$Q_{sca}$ for  \u03BB=400nm")
ax.set_xlabel("x size parameter")
ax.set_ylabel("$Q_{sca}$")
plt.legend()
fig.savefig("/home/sdutta/code/random-matrix/examples/Q_sca_m.png")
