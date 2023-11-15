
# This comes from the x calculated from quadpy. I.e. the positions of the nodes!
x = np.array([[-0.11484508, -0.11484508, -0.16484508, -0.11484508, -0.19453524,
        -0.21484508, -0.11484508, -0.14765796],
       [ 0.84380316,  0.84380316,  0.84380316,  0.79457776,  0.79457776,
         0.84380316,  0.84380316,  0.82973876],
       [-0.11484508, -0.11484508, -0.16484508, -0.11484508, -0.19453524,
        -0.21484508, -0.11484508, -0.14765796],
       [ 0.67151428,  0.76996507,  0.76996507,  0.72073967,  0.72073967,
         0.76996507,  0.72073967,  0.73480407],
       [-0.67030984, -0.67030984, -0.57030984, -0.67030984, -0.51092953,
        -0.47030984, -0.67030984, -0.60468408],
       [ 0.09845079,  0.09845079,  0.09845079,  0.19690158,  0.19690158,
         0.09845079,  0.09845079,  0.12657959]])

k1_x, k1_y, k2_x, k2_y, d_x, d_y = x
ki_x = k1_x + d_x / 2
ki_y = k1_y + d_y / 2

kj_x = k1_x - d_x / 2
kj_y = k1_y - d_y / 2

ku_x = k2_x + d_x / 2
ku_y = k2_y + d_y / 2

kv_x = k2_x - d_x / 2
kv_y = k2_y - d_y / 2

cartesian_quadpy = np.vstack((ki_x, ki_y, kj_x, kj_y, ku_x, ku_y, kv_x, kv_y))


ki = np.vstack((ki_x, ki_y))
mod_ki = np.linalg.norm(ki,axis=0)

kj = np.vstack((kj_x, kj_y))
mod_kj = np.linalg.norm(kj,axis=0)

ku = np.vstack((ku_x, ku_y))
mod_ku = np.linalg.norm(ku,axis=0)

kv = np.vstack((kv_x, kv_y))
mod_kv = np.linalg.norm(kv,axis=0)

ki_z = np.sqrt(1 - ki_x**2 - ki_y**2)
kj_z = np.sqrt(1 - kj_x**2 - kj_y**2)
ku_z = np.sqrt(1 - ku_x**2 - ku_y**2)
kv_z = np.sqrt(1 - kv_x**2 - kv_y**2)

# ORIGINAL DOMAIN BEFORE QUADPY COMES IN
# THESE ARE THE VERTICES OF THIS SHAPE
x = np.array([[ 0.        ,  0.89302855,  0.        ,  0.45      , -0.9       ,
         0.        ],
       [ 0.        ,  0.89302855,  0.        ,  0.89302855, -0.9       ,
         0.        ],
       [-0.225     ,  0.89302855, -0.225     ,  0.89302855, -0.45      ,
         0.        ],
       [ 0.        ,  0.67151428,  0.        ,  0.67151428, -0.9       ,
         0.44302855],
       [-0.35860571,  0.67151428, -0.35860571,  0.67151428, -0.18278858,
         0.44302855],
       [-0.45      ,  0.89302855, -0.45      ,  0.89302855,  0.        ,
         0.        ],
       [ 0.        ,  0.89302855,  0.        ,  0.67151428, -0.9       ,
         0.        ]])

k1_x, k1_y, k2_x, k2_y, d_x, d_y = x.T
ki_x = k1_x + d_x / 2
ki_y = k1_y + d_y / 2

kj_x = k1_x - d_x / 2
kj_y = k1_y - d_y / 2

ku_x = k2_x + d_x / 2
ku_y = k2_y + d_y / 2

kv_x = k2_x - d_x / 2
kv_y = k2_y - d_y / 2

ki = np.vstack((ki_x, ki_y))
mod_ki = np.linalg.norm(ki,axis=0)

kj = np.vstack((kj_x, kj_y))
mod_kj = np.linalg.norm(kj,axis=0)

ku = np.vstack((ku_x, ku_y))
mod_ku = np.linalg.norm(ku,axis=0)

kv = np.vstack((kv_x, kv_y))
mod_kv = np.linalg.norm(kv,axis=0)

ki_z = np.sqrt(1 - ki_x**2 - ki_y**2)
kj_z = np.sqrt(1 - kj_x**2 - kj_y**2)
ku_z = np.sqrt(1 - ku_x**2 - ku_y**2)
kv_z = np.sqrt(1 - kv_x**2 - kv_y**2)

cartesian = np.vstack((ki_x, ki_y, kj_x, kj_y, ku_x, ku_y, kv_x, kv_y))

scheme = quadpy.tn.grundmann_moeller(6, 1)
coords = scheme.points
cubature_points = cartesian@coords
ki_x, ki_y, kj_x, kj_y, ku_x, ku_y, kv_x, kv_y = cubature_points
cartesian_manual = np.vstack((ki_x, ki_y, kj_x, kj_y, ku_x, ku_y, kv_x, kv_y))

ki = np.vstack((ki_x, ki_y))
mod_ki = np.linalg.norm(ki,axis=0)

kj = np.vstack((kj_x, kj_y))
mod_kj = np.linalg.norm(kj,axis=0)

ku = np.vstack((ku_x, ku_y))
mod_ku = np.linalg.norm(ku,axis=0)

kv = np.vstack((kv_x, kv_y))
mod_kv = np.linalg.norm(kv,axis=0)

ki_z = np.sqrt(1 - ki_x**2 - ki_y**2)
kj_z = np.sqrt(1 - kj_x**2 - kj_y**2)
ku_z = np.sqrt(1 - ku_x**2 - ku_y**2)
kv_z = np.sqrt(1 - kv_x**2 - kv_y**2)

rat_bad = cartesian_quadpy/cartesian_manual

# =============================================================================
##### NON BUGGY EXAMPLE 115
x = np.array([[ 0.  ,  0.89,  0.  ,  0.45, -0.9 ,  0.  ],
       [ 0.  ,  0.89,  0.  ,  0.89, -0.9 ,  0.  ],
       [ 0.  ,  0.67,  0.  ,  0.45, -0.9 ,  0.  ],
       [-0.22,  0.89, -0.22,  0.89, -0.45,  0.  ],
       [ 0.  ,  0.67,  0.  ,  0.67, -0.9 ,  0.44],
       [-0.36,  0.67, -0.36,  0.67, -0.18,  0.44],
       [ 0.  ,  0.45,  0.  ,  0.45, -0.9 ,  0.  ]])


k1_x, k1_y, k2_x, k2_y, d_x, d_y = x.T
ki_x = k1_x + d_x / 2
ki_y = k1_y + d_y / 2

kj_x = k1_x - d_x / 2
kj_y = k1_y - d_y / 2

ku_x = k2_x + d_x / 2
ku_y = k2_y + d_y / 2

kv_x = k2_x - d_x / 2
kv_y = k2_y - d_y / 2

ki = np.vstack((ki_x, ki_y))
mod_ki = np.linalg.norm(ki,axis=0)

kj = np.vstack((kj_x, kj_y))
mod_kj = np.linalg.norm(kj,axis=0)

ku = np.vstack((ku_x, ku_y))
mod_ku = np.linalg.norm(ku,axis=0)

kv = np.vstack((kv_x, kv_y))
mod_kv = np.linalg.norm(kv,axis=0)

ki_z = np.sqrt(1 - ki_x**2 - ki_y**2)
kj_z = np.sqrt(1 - kj_x**2 - kj_y**2)
ku_z = np.sqrt(1 - ku_x**2 - ku_y**2)
kv_z = np.sqrt(1 - kv_x**2 - kv_y**2)

cartesian = np.vstack((ki_x, ki_y, kj_x, kj_y, ku_x, ku_y, kv_x, kv_y))

scheme = quadpy.tn.grundmann_moeller(6, 1)
coords = scheme.points
cubature_points = cartesian@coords
ki_x, ki_y, kj_x, kj_y, ku_x, ku_y, kv_x, kv_y = cubature_points
cartesian_manual = np.vstack((ki_x, ki_y, kj_x, kj_y, ku_x, ku_y, kv_x, kv_y))

ki = np.vstack((ki_x, ki_y))
mod_ki = np.linalg.norm(ki,axis=0)

kj = np.vstack((kj_x, kj_y))
mod_kj = np.linalg.norm(kj,axis=0)

ku = np.vstack((ku_x, ku_y))
mod_ku = np.linalg.norm(ku,axis=0)

kv = np.vstack((kv_x, kv_y))
mod_kv = np.linalg.norm(kv,axis=0)

ki_z = np.sqrt(1 - ki_x**2 - ki_y**2)
kj_z = np.sqrt(1 - kj_x**2 - kj_y**2)
ku_z = np.sqrt(1 - ku_x**2 - ku_y**2)
kv_z = np.sqrt(1 - kv_x**2 - kv_y**2)

# FROM QP
x = np.array([[-0.06, -0.06, -0.06, -0.11, -0.06, -0.14, -0.06, -0.08],
       [ 0.77,  0.77,  0.72,  0.77,  0.72,  0.72,  0.67,  0.73],
       [-0.06, -0.06, -0.06, -0.11, -0.06, -0.14, -0.06, -0.08],
       [ 0.6 ,  0.7 ,  0.6 ,  0.7 ,  0.65,  0.65,  0.6 ,  0.64],
       [-0.77, -0.77, -0.77, -0.67, -0.77, -0.61, -0.77, -0.73],
       [ 0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.2 ,  0.2 ,  0.1 ,  0.13]])

k1_x, k1_y, k2_x, k2_y, d_x, d_y = x
ki_x = k1_x + d_x / 2
ki_y = k1_y + d_y / 2

kj_x = k1_x - d_x / 2
kj_y = k1_y - d_y / 2

ku_x = k2_x + d_x / 2
ku_y = k2_y + d_y / 2

kv_x = k2_x - d_x / 2
kv_y = k2_y - d_y / 2

cartesian_quadpy = np.vstack((ki_x, ki_y, kj_x, kj_y, ku_x, ku_y, kv_x, kv_y))


ki = np.vstack((ki_x, ki_y))
mod_ki = np.linalg.norm(ki,axis=0)

kj = np.vstack((kj_x, kj_y))
mod_kj = np.linalg.norm(kj,axis=0)

ku = np.vstack((ku_x, ku_y))
mod_ku = np.linalg.norm(ku,axis=0)

kv = np.vstack((kv_x, kv_y))
mod_kv = np.linalg.norm(kv,axis=0)

ki_z = np.sqrt(1 - ki_x**2 - ki_y**2)
kj_z = np.sqrt(1 - kj_x**2 - kj_y**2)
ku_z = np.sqrt(1 - ku_x**2 - ku_y**2)
kv_z = np.sqrt(1 - kv_x**2 - kv_y**2)


rat_good = cartesian_quadpy/cartesian_manual