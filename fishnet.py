from actors import Actor
from math_util import *
import numpy as np
import pandas as pd
from matplotlib.tri import Triangulation

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
        self.net_obstacle_points = None

    def load_net_mesh(self, csv_file):

        # Step 1: define scenario range and construct nodes for A* algorithm
        nx, ny, nz = (15, 15, 10)
        m = 10
        n = 6
        x = np.linspace(-self.scenario_len/2, self.scenario_len/2, nx)
        y = np.linspace(-self.scenario_wid/2, self.scenario_wid/2, ny)
        z = np.linspace(-self.scenario_height, 0, nz)
        xv, yv, zv = np.meshgrid(x,y,z,sparse=False, indexing='ij')
        
        df = pd.read_excel(csv_file, sheet_name='Sheet1', header=None)
        net_raw_node = df.values
        net_node = np.zeros((n, 3, m)) # 6*3*10
        for i in range(n):
            net_node[i,:,:] = net_raw_node[np.newaxis, :, i*m:(i+1)*m]
        
        net_node_ext = np.zeros((3, n*(m+1))) # 3*66 skew shape
        # net_node_ext for visualization only
        index=0
        for i in range(n):
            net_node_ext[:, index:index+m] = net_raw_node[:,i*m:(i+1)*m]
            net_node_ext[:, index+m] = net_raw_node[:,i*m]
            index = index+m+1

        mv, nv = np.meshgrid(np.arange(m+1), np.arange(n))
        self.net_tri = Triangulation(np.ravel(mv), np.ravel(nv))
        net_node_x = np.ravel(net_node_ext[0,:])
        net_node_y = np.ravel(net_node_ext[1,:])
        net_node_z = np.ravel(net_node_ext[2,:])

        self.net_mesh = (net_node_x,net_node_y,net_node_z)
    
    def update_obstacle_points(self):
        # Create obstacle points based on net mesh
        pass
        
