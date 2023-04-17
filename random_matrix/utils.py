'''
Utility functions used elsewhere. May separate into multiple files if 
it becomes too large.

'''
import numpy as np

def circle(x, r):
    return np.sqrt(r**2-x**2)

def cartesian_to_polar(points_cartesian):
    nx, ny = np.shape(points_cartesian)
    points_polar = np.zeros((nx,ny),dtype=float)
    for i, point in enumerate(points_cartesian):
        x = point[0]
        y = point[1]
        r = np.sqrt(x**2 + y**2)
        t = np.mod(np.arctan2(y, x), 2*np.pi)
        points_polar[i] = np.array([r,t],dtype=float)
    return points_polar

def polar_to_cartesian(points_polar):
    nx, ny = np.shape(points_polar)
    points_cartesian = np.zeros((nx,ny),dtype=float)
    for i, point in enumerate(points_polar):
        r = point[0]
        t = point[1]
        x = r*np.cos(t)
        y = r*np.sin(t)
        points_cartesian[i] = np.array([x,y],dtype=float)
    return points_cartesian

def remove_duplicate_points(points, tolerance = 1e-8):
    n_points = np.shape(points)[0]
    duplicate_indices = []
    
    for i in range(n_points):
        # Skip points that we already know are duplicates
        if i in duplicate_indices:
            continue

        base_point = points[i]
        for j in range(i+1, n_points):
            second_point = points[j]
            if np.all(np.isclose(base_point, second_point, rtol=tolerance)):
                duplicate_indices.append(j)

    unique_value_indices = [i for i in range(n_points) if i not in duplicate_indices]
    unique_points = points[unique_value_indices]
    return unique_points


def draw_ray(ax, theta=0, r_min=0, r_max=1, color="tab:blue", linestyle="--",alpha=1.0):
        x = np.array([r_min*np.cos(theta), r_max*np.cos(theta)])
        y = np.array([r_min*np.sin(theta), r_max*np.sin(theta)])
        ax.plot(x, y, linestyle=linestyle, color=color,alpha=alpha)
 
def draw_circle(ax, r=1, t_min=0, t_max=2*np.pi, color="black", linestyle="-"):
        t = np.linspace(t_min, t_max)
        x = r*np.cos(t)
        y = r*np.sin(t)
        ax.plot(x,y,color=color, linestyle=linestyle)