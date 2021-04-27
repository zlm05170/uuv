import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize

def h(R,t):
    return np.array([R*np.cos(t), R*np.sin(t), t])
def hhelix(R,t): # the derivative of helix
    return [-R*np.sin(t), R*np.cos(t), 1]
def helix_vector(R,z):
    helix_path = np.array([R*np.cos(z), R*np.sin(z), -z])
    omage = [0,0,100/2*np.pi]
    r = [R*np.cos(t), R*np.sin(t),t]
    return np.cross(omage, r)

t = np.linspace(0, 10*np.pi, 500)

omage = [0,0,100/2*np.pi]
r = [10*np.cos(1), 10*np.sin(1),1]
# [R*np.cos(t), R*np.sin(t),t]
a = np.cross(omage, r)

print(a)

fig2 = plt.figure(2)
ax3d = fig2.add_subplot(111, projection='3d')
# ax3d.plot(x_true, y_true, z_true, 'b')
# ax3d.plot(x_sample, y_sample, z_sample, 'r*')
# ax3d.plot(x_knots, y_knots, z_knots, 'go')
# ax3d.plot(x_fine, y_fine, z_fine, 'g')
# plt.plot(hhelix[0,:], hhelix[1,:], hhelix[2,:])
# fig2.show()
# plt.show()








