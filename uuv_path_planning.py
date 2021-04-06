import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.tri import Triangulation
from matplotlib import cm
from shapely.geometry import Point, MultiPoint, Polygon
#https://shapely.readthedocs.io/en/stable/manual.html
from itertools import product

'''
given depth, return the net polygon in xy plane
net_node: 3d array (depth_layer, 3 (x,y,z direction), node_num_per_layer)
'''
def get_depth_polygon(depth, net_node):
    if depth> max(net_node[:,-1,:].reshape(-1)) or depth < min(net_node[:, -1, :].reshape(-1)):
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

'''
check nodes are obstacles (inside the fish net), or non-obstacle
xv, yv, zv: the input of meshgrid index in x, y, z directions
net_node: 3d array (depth_layer, 3 (x,y,z direction), node_num_per_layer)
c_dist: int type, node number representing clearance distance to keep the uuv safe
return: obs (same size of xv, yv, zv), with
value 0 indicating non-obstacle node, 0.5 means safe clearance node, 1 means an obstacle node
'''
def obs_check(xv, yv, zv, net_node, c_dist):
    obs = np.zeros(xv.shape)
    for i in range(np.size(xv, 2)): # z direction
        d_projection = get_depth_polygon(zv[0][0][i], net_node) # 10*2
        obs_poly = Polygon(d_projection)
        for j in range(np.size(xv, 0)): # x direction
            for k in range(np.size(xv, 1)): # y direction
                #print(k,j,i)
                if Point(xv[j][k][i], yv[j][k][i]).within(obs_poly): # node inside fish net
                    obs[j][k][i] = 1
                    #print(k,j,i)

    obs_clear = np.zeros(np.array(xv.shape) + 2*c_dist)
    obs_clear[c_dist: c_dist+xv.shape[0],
              c_dist: c_dist+xv.shape[1],
              c_dist: c_dist+xv.shape[2]] = obs

    neighbor_idx = list(product([-1, 0, 1], repeat=3))  # the index of neighbors in 3D directions
    neighbor_idx.remove((0, 0, 0))

    for m in range(1, c_dist+1):
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
    obs = obs + 0.5*obs_clear_result
    return obs

'''
reconstruct path
cameFrom_x, cameFrom_y, cameFrom_z: the index of parent node in x,y,z directions
goal: the destination node (a triple tuple)
return: list of nodes  
'''
def reconstruct_path(cameFrom_x, cameFrom_y, cameFrom_z, goal):
    path = [goal]
    cur_node = goal
    while not np.isinf(cameFrom_x[cur_node]):
        cur_node = (int(cameFrom_x[cur_node]), int(cameFrom_y[cur_node]), int(cameFrom_z[cur_node]))
        path.insert(0, cur_node)

    return path

'''
https://en.wikipedia.org/wiki/A*_search_algorithm
start, goal: a node index (a triple tuple) containing the indexes in x,y,z direction
nx, ny, nz: the max indexing number in x, y, z directions
xv, yv, zv: the node position in x, y, z directions
is_obs: determin a node is an obstacle or not
return: if success, a list of node from start to goal; otherwise None
note the meaning of f = g + h in A star algorithm
'''
def A_star_search(start, goal, nx, ny, nz, xv, yv, zv, is_obs):
    if any(x < y for x, y in zip(start, (0,0,0))) or \
        any(x >= y for x, y in zip(start, (nx, ny, nz))) or \
        is_obs[start] == 1 or \
        any(x < y for x, y in zip(goal, (0, 0, 0))) or \
        any(x >= y for x, y in zip(goal, (nx, ny, nz))) or \
        is_obs[goal] == 1:
        print('start or goal are either out of range, or inside the fish net')
        return None

    openSet = [start] # store the nodes that visited at the first time
    openSet_f = [0] # store the f value of the corresponding node in openSet
    in_openSet = np.zeros((nx, ny, nz)) # used to check if a node is in openset
    in_openSet[start] = 1

    g = np.inf * np.ones((nx, ny, nz))  # the cost of the cheapest path from start to g[i,j,k] currently known.
    g[start] = 0

    f = np.inf * np.ones((nx, ny, nz))  # f[i,j,k] is the heuristic guess of the cheapest path from start to goal via node [i,j,k]
    f[start] = g[start] \
               + np.linalg.norm(np.array([xv[goal],yv[goal],zv[goal]])-np.array([xv[start],yv[start],zv[start]]))

    cameFrom_x = np.inf * np.ones((nx, ny, nz))  # cameFrom_x[i,j,k] is the parent node (in x direction) of node [i,j,k] with the cheapest cost
    cameFrom_y = np.inf * np.ones((nx, ny, nz))
    cameFrom_z = np.inf * np.ones((nx, ny, nz))

    neighbor_idx = list(product([-1, 0, 1], repeat=3)) # the index of neighbors in 3D directions
    neighbor_idx.remove((0,0,0))

    while len(openSet) != 0:
        openSet_f_min_idx = int(np.argmin(openSet_f))
        cur_node = openSet[openSet_f_min_idx]
        if cur_node == goal:
            print('find goal')
            return reconstruct_path(cameFrom_x, cameFrom_y, cameFrom_z, goal)

        del openSet[openSet_f_min_idx]
        del openSet_f[openSet_f_min_idx]
        in_openSet[cur_node] = 0

        for n_idx in neighbor_idx:
            neighbor = tuple(np.array(n_idx) + np.array(cur_node))
            #print(neighbor)
            #print(neighbor < (nx,ny,nz))
            if all(x >= y for x, y in zip(neighbor, (0,0,0))) and \
                    all(x < y for x, y in zip(neighbor, (nx,ny,nz))) and\
                    is_obs[neighbor] == 0:  # ensure the neighbor is not out of scenario bounds and not an obstacle node
                if np.sign(goal[-1]- start[-1])==0:
                    g_tmp = g[cur_node] \
                            + np.linalg.norm(np.array([xv[neighbor], yv[neighbor], zv[neighbor]]) - np.array([xv[cur_node], yv[cur_node], zv[cur_node]]))
                else:
                    if np.sign(goal[-1]- start[-1]) == np.sign(neighbor[-1] - cur_node[-1]):
                        g_tmp = g[cur_node] \
                                + 2*np.linalg.norm(np.array([xv[neighbor], yv[neighbor], zv[neighbor]]) - np.array(
                                                            [xv[cur_node], yv[cur_node], zv[cur_node]]))
                    else:
                        g_tmp = g[cur_node] \
                                + np.linalg.norm(np.array([xv[neighbor], yv[neighbor], zv[neighbor]]) - np.array(
                                                          [xv[cur_node], yv[cur_node], zv[cur_node]]))

                if g_tmp < g[neighbor]:
                    cameFrom_x[neighbor] = cur_node[0]
                    cameFrom_y[neighbor] = cur_node[1]
                    cameFrom_z[neighbor] = cur_node[2]

                    g[neighbor] = g_tmp
                    f[neighbor] = g[neighbor] +\
                                  np.linalg.norm(np.array([xv[goal], yv[goal], zv[goal]]) - np.array([xv[cur_node], yv[cur_node], zv[cur_node]]))

                    if in_openSet[neighbor] == 0:  # neighbor not in the openSet
                        openSet.append(neighbor)
                        openSet_f.append(f[neighbor])
                        in_openSet[neighbor] = 1
    return None  # openSet is empty but goal was never reached


# Step 1: define scenario range and construct nodes for A* algorithm
net_radius = 30  # m
net_height = 30  # m

scenario_len = 5*net_radius
scenario_wid = 5*net_radius
scenario_height = 2*net_height

nx, ny, nz = (15, 15, 10)  # number of node in x,y,z directions
x = np.linspace(-scenario_len/2, scenario_len/2, nx)
y = np.linspace(-scenario_wid/2, scenario_wid/2, ny)
z = np.linspace(-scenario_height, 0, nz)
xv, yv, zv = np.meshgrid(x, y, z, sparse=False, indexing='ij')  # nodes used for A star algorithm
m = 10
n = 5 + 1

df = pd.read_excel('/home/luman/ws/uuv_planning/uuv/fish_net_10_5_0.5.xlsx', sheet_name='Sheet1', header=None)
net_raw_node=df.values
net_node = np.zeros((n, 3, m))
for i in range(n):
    net_node[i,:,:] = net_raw_node[np.newaxis, :, i*m:(i+1)*m] # 0:10, 11:20

# net_node_ext for visualization only
net_node_ext = np.zeros((3, n*(m+1))) # m+1 is for the last node equal the first node
index = 0
for i in range(n):
    net_node_ext[:, index:index+m] = net_raw_node[:,i*m:(i+1)*m]
    net_node_ext[:, index+m] = net_raw_node[:,i*m]
    index = index+m+1

mv, nv = np.meshgrid(np.arange(m+1), np.arange(n))
tri = Triangulation(np.ravel(mv), np.ravel(nv))
net_node_x = np.ravel(net_node_ext[0,:])
net_node_y = np.ravel(net_node_ext[1,:])
net_node_z = np.ravel(net_node_ext[2,:])
print(net_node_x.shape)
fig = plt.figure()
ax = fig.gca(projection='3d')
#surf = ax.plot_trisurf(net_node_x, net_node_y, net_node_z, triangles=tri.triangles,
#                cmap='viridis', linewidths=0.0, antialiased=True)
#surf = ax.plot_wireframe(net_node_x, net_node_y, net_node_z, rstride=10, cstride=10)
surf = ax.plot_trisurf(net_node_x, net_node_y, net_node_z, triangles=tri.triangles,
                color='#0fff0f80',edgecolors='#08ff0880',linewidths=0.5, antialiased=True)
ax.set_xlim(-scenario_len/2, scenario_len/2)
ax.set_ylim(-scenario_wid/2, scenario_wid/2)
ax.set_zlim(-scenario_height, 2)

#fig.colorbar(surf, shrink=0.5, aspect=5)

#plot vertical line
# line_data = net_node[:,:,0] # 6*3*10
# # the initial point in each layer
# ax.scatter3D(line_data[:,0], line_data[:,1], line_data[:,2], s=40, c='r', marker='o')

# plot scatter of scenario nodes
is_obs = obs_check(xv, yv, zv, net_node, 1)
for i in range(np.size(is_obs, 2)):  # z direction
    for j in range(np.size(is_obs, 0)):  # x direction
        for k in range(np.size(is_obs, 1)):  # y direction
            if is_obs[j][k][i] == 1:
                ax.scatter3D(xv[j][k][i], yv[j][k][i], zv[j][k][i], s=20, c='b',marker='o')
            elif is_obs[j][k][i] == 0.5:
                ax.scatter3D(xv[j][k][i], yv[j][k][i], zv[j][k][i], s=20, color='#ff9999ff', marker='o')


# start_idx = (2, 2, 6)
# end_idx = (12, 13, 6)

# if all(x >= y for x, y in zip(start_idx, (0,0,0))) and \
#    all(x < y for x, y in zip(start_idx, (nx, ny, nz))):
#     ax.scatter3D(xv[start_idx], yv[start_idx], zv[start_idx], s=20, c='m', marker='o')
# else:
#     print("start point is out of range")
# if all(x >= y for x, y in zip(end_idx, (0,0,0))) and \
#    all(x < y for x, y in zip(end_idx, (nx, ny, nz))):
#     ax.scatter3D(xv[end_idx], yv[end_idx], zv[end_idx], s=20, color='#00ffffff', marker='x')
# else:
#     print("end point is out of range")

# final_path = A_star_search(start_idx, end_idx, nx, ny, nz, xv, yv, zv, is_obs)
# if final_path is not None:
#     node_x = []
#     node_y = []
#     node_z = []
#     for node in final_path:
#         print((node))
#         node_x.append(xv[node])
#         node_y.append(yv[node])
#         node_z.append(zv[node])
#     ax.plot(node_x, node_y, node_z, color='r')

plt.show()