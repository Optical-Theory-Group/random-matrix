import numpy as np
import sklearn
import scipy
import time


points = np.random.rand(1000, 8)  # 100,000 points in 8D


start = time.perf_counter()
hull = scipy.spatial.ConvexHull(points)
end = time.perf_counter()
print(f"Time: {end - start}")
print(f"VolumeL")

# Reduce to 5D
start = time.perf_counter()
pca = sklearn.decomposition.PCA(n_components=5)
points_reduced = pca.fit_transform(points)
hull_reduced = scipy.spatial.ConvexHull(points_reduced)
end = time.perf_counter()
print(f"Time: {end - start}")