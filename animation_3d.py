from math_util import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List
from mpl_toolkits import mplot3d

def get_box(l, w, h, res):
    line = (np.array([
        [1,1,1],
        [0,1,1],
        [0,0,1],
        [1,0,1],
        [1,1,1],
        [1,1,0],
        [1,0,0],
        [0,0,0],
        [0,1,0],
        [1,1,0],
        [1,1,1],
        [1,0,1],
        [1,0,0],
        [0,0,0],
        [0,0,1],
        [0,1,1],
        [0,1,0]]) - np.array([0.5,0.5,0.5]))*np.array([l,w,h])
    return line

def draw_motion(i, pose_series : List[Pose], line, shape):
    shape.shape[0]
    pose = pose_series[i]
    data = np.array(shape)
    for i in range(shape.shape[0]):
        data[i, :] = pose.transform(shape[i,:])
    line.set_xdata(data[:,0])
    line.set_ydata(data[:,1])
    line.set_3d_properties(data[:,2])

def draw_static_surf(fig, ax, points, tri):
    ax.plot_trisurf(points[0], points[1], points[2], triangles=tri.triangles,
            color='#0fff0f80',edgecolors='#08ff0880',linewidths=0.5, antialiased=True)

def draw_obs_point(fig, ax, start, goal, space_mesh_data, is_obs):
    xv, yv, zv = space_mesh_data[0], space_mesh_data[1], space_mesh_data[2]
    ax.scatter3D(start[0], start[1], start[2], s=20, c='m', marker='o')
    ax.scatter3D(goal[0], goal[1], goal[2], s=20, color='#00ffffff', marker='x')

    for i in range(np.size(is_obs, 2)):
        for j in range(np.size(is_obs, 0)):  # x direction
            for k in range(np.size(is_obs, 1)):  # y direction
                if is_obs[j][k][i] == 1:
                    ax.scatter3D(xv[j][k][i], yv[j][k][i], zv[j][k][i], s=20, c='b',marker='o')
                elif is_obs[j][k][i] == 0.5:
                    ax.scatter3D(xv[j][k][i], yv[j][k][i], zv[j][k][i], s=20, color='#ff9999ff', marker='o')

def draw_find_path(fig, ax, waypoint_series):
    if waypoint_series is not None:
        node_x = []
        node_y = []
        node_z = []
        for node in waypoint_series:
            node_x.append(node[0])
            node_y.append(node[1])
            node_z.append(node[2])
        ax.plot(node_x, node_y, node_z, color='r')
    
def animate_motion(fig, ax, pose_series, shape, space_x, space_y, space_z, dt):

    # Attaching 3D axis to the figure
    line, = ax.plot([], [], [], '-')
    # Number of iterations
    iterations = len(pose_series)

    # Setting the axes properties
    ax.set_xlim3d([-space_x, space_x])
    ax.set_xlabel('X')

    ax.set_ylim3d([-space_y, space_y])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-space_z, space_z])
    ax.set_zlabel('Z')

    ax.set_title('3D Animated Example')

    # Provide starting angle for the view.
    ax.view_init(25, 10)

    return animation.FuncAnimation(fig, draw_motion, iterations, fargs=(pose_series, line, shape), interval=1000.0*dt, blit=False, repeat=True)

