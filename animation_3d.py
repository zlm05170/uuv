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
        [0,1,0]]) - np.array([0.5,0.5,0.5]))* np.array([l,w,h])
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

def animate_motion(pose_series, shape, space_x, space_y, space_z, dt):
    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = plt.axes(projection="3d")
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

    ani = animation.FuncAnimation(fig, draw_motion, iterations, fargs=(pose_series, line, shape), interval=1000.0*dt, blit=False, repeat=True)
    plt.show()