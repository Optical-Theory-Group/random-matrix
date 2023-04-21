import matplotlib.pyplot as plt
import numpy as np
from .geometry_utils import points_to_ordered_convex_hull_vertices, circle
from .array_utils import get_pairs

def draw_ray(ax, theta=0, r_min=0, r_max=1, color="tab:blue", linestyle="--",alpha=1.0):
    x = np.array([r_min*np.cos(theta), r_max*np.cos(theta)])
    y = np.array([r_min*np.sin(theta), r_max*np.sin(theta)])
    ax.plot(x, y, linestyle=linestyle, color=color,alpha=alpha)
 
def draw_circle(ax, r=1, t_min=0, t_max=2*np.pi, color="black", linestyle="-"):
    t = np.linspace(t_min, t_max)
    x = r*np.cos(t)
    y = r*np.sin(t)
    ax.plot(x,y,color=color, linestyle=linestyle)

def draw_line(ax, start=None, end=None, color="black", linestyle="-"):
    xs = np.array([start[0], end[0]])
    ys = np.array([start[1], end[1]])
    ax.plot(xs,ys,color=color, linestyle=linestyle)

def draw_vertical_chord(ax, x, radius=1, color="black", linestyle="-"):
    y_top = circle(x, radius)
    y_bottom = -y_top
    bottom_point = np.array([x, y_bottom])
    top_point = np.array([x, y_top])
    draw_line(ax, bottom_point, top_point, color=color, linestyle=linestyle)

def draw_horizontal_chord(ax, y, radius=1, color="black", linestyle="-"):
    x_right = circle(y, radius)
    x_left = -x_right
    left_point = np.array([x_left, y])
    right_point = np.array([x_right, y])
    draw_line(ax, left_point, right_point, color=color, linestyle=linestyle)

def draw_convex_polygon(ax, points, color="black", linestyle="-"):
    ordered_points = points_to_ordered_convex_hull_vertices(points)
    pairs = get_pairs(ordered_points, cyclic=True)
    for first_point, second_point in pairs:
        draw_line(ax, start=first_point, end=second_point, color=color, linestyle=linestyle)

def set_up_k_space_plot():
    fig, ax = plt.subplots()
    draw_ray(ax, r_min=-1, theta=0, linestyle="-", color="black", alpha=0.4)
    draw_ray(ax, r_min=-1, theta=np.pi/2, linestyle="-", color="black", alpha=0.4)
    ax.set_aspect("equal")
    draw_circle(ax)
    return ax

