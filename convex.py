import numpy as np
import scipy
import scipy.spatial

vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2,3,4]])
hull = scipy.spatial.ConvexHull(vertices)