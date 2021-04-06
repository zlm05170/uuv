from actors import Actor
from math_util import *
import numpy as np
import pandas as pd
from matplotlib.tri import Triangulation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from shapely.geometry import Point, MultiPoint, Polygon
#https://shapely.readthedocs.io/en/stable/manual.html
from itertools import product

class Fishnet(Actor):
    def __init__(self, parent : 'Actor' = None, pose : 'Pose' = Pose(), net_radius = 1.0, net_height = 1.0):
        super().__init__(parent, pose)
        self.net_radius = net_radius
        self.net_height = net_height
        self.scenario_len = 5*self.net_radius
        self.scenario_wid = 5*self.net_radius
        self.scenario_height = 2*self.net_height
        self.net_mesh = None
        self.net_tri = None
        self.net_node = None
        # the data will transfer to planner actor and gui
        self.space_mesh_data = self.space_mesh()
        self.net_obstacle_points = None

    def space_mesh(self):
        # Step 1: define scenario range and construct nodes for A* algorithm
        nx, ny, nz = (15, 15, 10)
        x = np.linspace(-self.scenario_len/2, self.scenario_len/2, nx)
        y = np.linspace(-self.scenario_wid/2, self.scenario_wid/2, ny)
        z = np.linspace(-self.scenario_height, 0, nz)
        xv, yv, zv = np.meshgrid(x,y,z,sparse=False, indexing='ij')
        return xv, yv, zv

    def load_net_mesh(self, csv_file):
        m = 10
        n = 6
        df = pd.read_excel(csv_file, sheet_name='Sheet1', header=None)
        net_raw_node = df.values
        self.net_node = np.zeros((n, 3, m)) # 6*3*10
        for i in range(n):
            self.net_node[i,:,:] = net_raw_node[np.newaxis, :, i*m:(i+1)*m]
        
        net_node_ext = np.zeros((3, n*(m+1))) # 3*66 skew shape
        # net_node_ext for visualization only
        index = 0
        for i in range(n):
            net_node_ext[:, index:index+m] = net_raw_node[:,i*m:(i+1)*m]
            net_node_ext[:, index + m] = net_raw_node[:, i*m]
            index = index+m+1

        mv, nv = np.meshgrid(np.arange(m+1), np.arange(n))
        self.net_tri = Triangulation(np.ravel(mv), np.ravel(nv))
        net_node_x = np.ravel(net_node_ext[0,:])
        net_node_y = np.ravel(net_node_ext[1,:])
        net_node_z = np.ravel(net_node_ext[2,:])
        self.net_mesh = (net_node_x,net_node_y,net_node_z)
    
    @staticmethod
    def get_depth_polygon(depth, net_node):
        if depth > max(net_node[:,-1,:].reshape(-1)) or depth < min(net_node[:, -1, :].reshape(-1)):
            print('depth out of range in z direction')
            return None
        depth_layer, dir, node_num_per_layer = net_node.shape
        depth_projection = []
        for i in range(node_num_per_layer):
            for j in range(depth_layer-1):
                if net_node[j, -1, i]>=depth and net_node[j+1, -1, i]<=depth: # here will have problem if the vertical line is not monotonous
                    ratio = (depth-net_node[j, -1, i]) / (net_node[j, -1, i] - net_node[j+1, -1, i])
                    pos_tmp = net_node[j, 0:-1, i] + ratio * (net_node[j, 0:-1, i] - net_node[j+1, 0:-1, i])
                    #depth_projection.append([pos_tmp[0], pos_tmp[1], depth])
                    depth_projection.append(pos_tmp)
                    #print(pos_tmp[0], pos_tmp[1])
        depth_projection = np.array(depth_projection)
        return depth_projection
    
    def update_obstacle_points(self, c_dist): 
        xv, yv, zv = self.space_mesh_data[0], self.space_mesh_data[1], self.space_mesh_data[2]
        obs = np.zeros(xv.shape)
        for i in range(np.size(xv, 2)): # z direction
            d_projection = Fishnet.get_depth_polygon(zv[0][0][i], self.net_node)
            obs_poly = Polygon(d_projection)
            for j in range(np.size(xv, 0)): # x direction
                for k in range(np.size(xv, 1)): # y direction
                    #print(k,j,i)
                    if Point(xv[j][k][i], yv[j][k][i]).within(obs_poly): # node inside fish net
                        obs[j][k][i] = 1
                        #print(k,j,i)

        obs_clear = np.zeros(np.array(xv.shape) + 2*c_dist)
        obs_clear[c_dist: c_dist + xv.shape[0],
                c_dist: c_dist + xv.shape[1],
                c_dist: c_dist + xv.shape[2]] = obs

        neighbor_idx = list(product([-1, 0, 1], repeat=3))  # the index of neighbors in 3D directions
        neighbor_idx.remove((0, 0, 0))

        for m in range(1,c_dist+1):
            for n_idx in neighbor_idx:
                c_offset = m*np.array(n_idx)
                #print(c_offset)

                obs_clear[c_dist + c_offset[0]: c_dist + c_offset[0] + xv.shape[0],
                        c_dist + c_offset[1]: c_dist + c_offset[1] + xv.shape[1],
                        c_dist + c_offset[2]: c_dist + c_offset[2] + xv.shape[2]] = \
                        obs_clear[c_dist + c_offset[0]: c_dist + c_offset[0] + xv.shape[0],
                                    c_dist + c_offset[1]: c_dist + c_offset[1] + xv.shape[1],
                                    c_dist + c_offset[2]: c_dist + c_offset[2] + xv.shape[2]] + obs
        obs_clear = (np.logical_and(obs_clear, obs_clear)).astype(float)
        obs_clear_result = np.logical_xor(obs, obs_clear[c_dist: c_dist+xv.shape[0],
                                                        c_dist: c_dist+xv.shape[1],
                                                        c_dist: c_dist+xv.shape[2]]).astype(float)
        obs = obs + 0.5 * obs_clear_result
        self.net_obstacle_points = obs

