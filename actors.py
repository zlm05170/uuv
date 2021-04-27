import random
import re
from itertools import product
from typing import List

import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib.pyplot import cla, step
from scipy import interpolate
from shapely.geometry.base import EMPTY

from math_util import *


class Actor:

    def __init__(self, parent : 'Actor' = None, pose : 'Pose' = Pose()):
        self.pose : 'Pose' = pose
        self.parent = parent

    def communicate(self):
        pass

    def update(self, dt, t):
        pass

    def cleanup(self):
        pass


class RigidBody(Actor):

    def __init__(self, parent : 'Actor' = None, pose : 'Pose' = Pose(), mass = 1, cg = [0,0,0], moment_of_inertia = np.identity(3).tolist()):
        super().__init__(parent, pose)
        self.mass = mass  # kg
        self.center_of_gravity = np.array(cg)
        self.moment_of_inertia = np.array(moment_of_inertia)  # relative to cog
        self.velocity = np.zeros(3)  # world frame
        self.rotational_velocity = np.zeros(3)  # world frame
        # external_force is in world frame, relative to model origin
        self.external_force_world_frame = np.zeros(3)
        # external_torque is in world frame, relative to model origin
        self.external_torque_world_frame = np.zeros(3)
        self.added_mass_6x6 = np.zeros((6, 6))  # relative to model origin

    def add_force_torque_model_frame(self, force_torque : ' ForceTorque', point : 'np.ndarray'):
        self.external_force_world_frame += self.pose.rotation.rotate(force_torque.force)
        self.external_torque_world_frame += self.pose.rotation.rotate(force_torque.torque)
        self.external_torque_world_frame += self.pose.rotation.rotate(np.cross(point, force_torque.force))

    def add_force_torque_world_frame(self, force_torque : ' ForceTorque', point : 'np.ndarray'):
        self.external_force_world_frame += force_torque.force
        self.external_torque_world_frame += force_torque.torque
        self.external_torque_world_frame += np.cross(self.pose.rotation.rotate(point), force_torque.force)
        
    def get_velocity_model_frame(self):
        return (~self.pose.rotation).rotate(self.velocity), (~self.pose.rotation).rotate(self.rotational_velocity)

    def get_gravitational_force(self, gravitational_acceleration):
        return self.mass * gravitational_acceleration

    def get_paralleled_moment_of_inertia(self, arm):
        return self.moment_of_inertia + (np.dot(arm, arm)*np.identity(3) - np.outer(arm, arm)) * self.mass

    def get_smtrx(self, v):
        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])

    def set_added_mass_6x6(self, added_mass : np.ndarray):
        self.added_mass_6x6 = added_mass

    def update(self, dt, t):
        super().update(dt, t)
        # Get external force & torque in model frame
        world_to_model = ~self.pose.rotation
        external_force_model_frame = world_to_model.rotate(self.external_force_world_frame)
        external_torque_model_frame = world_to_model.rotate(self.external_torque_world_frame)

        # Set up 6x6 inertia matrix in model frame
        inertia_matrix_6x6 = np.zeros((6, 6))
        inertia_matrix_6x6[0:3, 0:3] = np.identity(3) * self.mass
        inertia_matrix_6x6[3:6, 3:6] = self.get_paralleled_moment_of_inertia(-self.center_of_gravity)
        inertia_matrix_6x6[0:3, 3:6] = self.get_smtrx(self.center_of_gravity) * self.mass
        inertia_matrix_6x6[3:6, 0:3] = -inertia_matrix_6x6[0:3, 3:6]
        acceleration_translational_and_rotational_model_frame = np.linalg.inv(inertia_matrix_6x6 + self.added_mass_6x6).dot(np.concatenate((external_force_model_frame, external_torque_model_frame), axis=0))

        # Get acceleration in world frame, relative to model origin
        acceleration_translational_world_frame = self.pose.rotation.rotate(acceleration_translational_and_rotational_model_frame[0:3])
        acceleration_rotational_world_frame = self.pose.rotation.rotate(acceleration_translational_and_rotational_model_frame[3:6]) + np.cross(self.rotational_velocity, self.velocity)

        # Integrate acceleration to get velocity in world frame, relative to model origin
        velocity_translational_world_frame = self.velocity + acceleration_translational_world_frame * dt
        velocity_rotational_world_frame = self.rotational_velocity + acceleration_rotational_world_frame * dt
        self.velocity = velocity_translational_world_frame
        self.rotational_velocity = velocity_rotational_world_frame

        # Integrate velocity to get position and rotation
        self.pose.position = self.pose.position + velocity_translational_world_frame * dt
        velocity_rotational_model_frame = world_to_model.rotate(velocity_rotational_world_frame)
        self.pose.rotation = self.pose.rotation**Quaternion.make_from_exp(velocity_rotational_model_frame * dt*0.5)

    def cleanup(self):
        self.external_force_world_frame = np.zeros(3)
        self.external_torque_world_frame =  np.zeros(3)
        self.set_added_mass_6x6(np.zeros((6,6)))


class HydrodynamicResponseActor(Actor):
    def __init__(self, parent : 'Actor' = None, pose : 'Pose' = Pose(), restoring_matrix = np.zeros((6,6)), added_mass = np.zeros((6,6)), damping= np.zeros((6,6))):
        super().__init__(parent, pose)
        self.restoring_matrix = np.array(restoring_matrix, dtype=float)
        self.added_mass = np.array(added_mass, dtype=float)
        self.damping = np.array(damping, dtype=float)

    def get_restoring_force_torque(self, pose : 'Pose'):
        displacement_vector = np.concatenate((pose.position, pose.rotation.get_euler()), axis = 0 )
        force_torque_vector = self.restoring_matrix @ displacement_vector
        return ForceTorque(force = force_torque_vector[0:3], torque=force_torque_vector[3:6])

    def get_damping_force_torque(self, relative_translational_velocity_model_frame, relative_rotational_velocity_model_frame):
        velocity_vector = np.concatenate((relative_translational_velocity_model_frame, relative_rotational_velocity_model_frame), axis = 0)
        force_torque_vector = self.damping @ velocity_vector
        return ForceTorque(force = force_torque_vector[0:3], torque=force_torque_vector[3:6])

    def get_force_torque(self, pose : 'Pose', relative_translational_velocity_model_frame, relative_rotational_velocity_model_frame):
        restoring_force_torque = self.get_restoring_force_torque(pose) # world frame
        damping_force_torque = self.get_damping_force_torque(relative_translational_velocity_model_frame, relative_rotational_velocity_model_frame) # model frame
        force = (~self.pose.rotation).rotate(restoring_force_torque.force) + damping_force_torque.force
        torque = (~self.pose.rotation).rotate(restoring_force_torque.torque) + damping_force_torque.torque
        return ForceTorque(-force, -torque)

    def communicate(self):
        if isinstance(self.parent, RigidBody):
            tvel, rvel = self.parent.get_velocity_model_frame()
            self.parent.add_force_torque_model_frame(
                self.get_force_torque(
                    self.parent.pose, tvel, rvel), self.pose.position)
            self.parent.set_added_mass_6x6(self.added_mass)

      
class AbstractThruster(Actor):
    def __init__(self, parent : 'Actor' = None, pose : 'Pose' = Pose()):
        super().__init__(parent, pose)
        self.force_torque = ForceTorque()

    def get_force_torque(self) -> 'ForceTorque':
        return self.force_torque

    def communicate(self):    
        if isinstance(self.parent, RigidBody):
            self.parent.add_force_torque_model_frame(self.get_force_torque(), self.pose.position)     


class Thruster(AbstractThruster):
    def __init__(self, parent : 'Actor' = None, pose : 'Pose' = Pose()):
        super().__init__(parent, pose)
        self.thrust: float = 0.0
        self.torque: float = 0.0
        self.normal = np.array([1.0, 0.0, 0.0]) # local frame

    def get_force_torque(self):
        force = self.thrust * self.normal
        torque = self.torque * self.normal
        return ForceTorque(force, torque)

    def communicate(self):
        self.force_torque = ForceTorque(force=self.thrust * self.normal, torque= self.torque * self.normal)
        super().communicate()


class MoveToWayPointPoseController(Actor):
    '''
    This controller calculates the total ForceTorque required to move to way point
    '''
    def __init__(self, parent : 'Actor' = None, pose : 'Pose' = Pose()):
        super().__init__(parent, pose)
        self.target_pose = Pose()
        self.current_pose = Pose()
        self.current_velocity = np.zeros(3)
        self.current_angular_velocity = np.zeros(3)
        self.pid_position = np.zeros(3)
        self.pid_rotation = np.zeros(3)

        self.desired_force_torque_in_model = ForceTorque()

    def get_commanded_force_torque_in_world(self):
        # Implementation 1: direct 6DOF PID control
        position_error = self.target_pose.position - self.current_pose.position # difference of current position to target position
        rotation_error = self.target_pose.rotation ** ~self.current_pose.rotation # difference of current rotation to target rotation, in quaternion

        output_force = position_error * self.pid_position[0] + -self.current_velocity * self.pid_position[2]
        output_torque = rotation_error.get_euler() * self.pid_rotation[0] + -self.current_angular_velocity * self.pid_rotation[2] 
        
        # TODO: Better control
        return ForceTorque(output_force, output_torque)

    def communicate(self):
        if isinstance(self.parent, AbstractThruster):
            self.parent.force_torque = self.desired_force_torque_in_model
            if isinstance(self.parent.parent, RigidBody):
                self.current_pose = self.parent.parent.pose
                self.current_velocity = self.parent.parent.velocity
                self.current_angular_velocity = self.parent.parent.rotational_velocity

    def update(self, dt, t):
        desired_force_torque_in_world = self.get_commanded_force_torque_in_world()
        self.desired_force_torque_in_model.force = (~self.current_pose.rotation).rotate(desired_force_torque_in_world.force)
        self.desired_force_torque_in_model.torque = (~self.current_pose.rotation).rotate(desired_force_torque_in_world.torque)
        super().update(dt, t)


class WayPointPlanner(Actor): #
    def __init__(self, parent: 'MoveToWayPointController' = None, 
        pose: 'pose' = Pose()):
        super().__init__(parent, pose)
        self.start_pose = Pose()
        self.target_pose = Pose()
        self.uuv_pos = None
        self.fishnet_obs = None
        self.fishnet_space_mesh_data = None
        self.final_path = []
        self.a_star_search_needed = True
        self.current_path_point_target_index = 0
    
    def set_fishnet(self, fishnet: 'Fishnet'):
        self.fishnet_obs = fishnet.net_obstacle_points
        self.fishnet_space_mesh_data = fishnet.space_mesh_data
    
    def point_is_reached(self, target_point, current_point):
        return np.linalg.norm(target_point - current_point) < 0.1

    def instruct_controller(self):
        if self.point_is_reached(self.final_path[self.current_path_point_target_index], self.uuv_pos):
            if self.current_path_point_target_index < len(self.final_path)-1:
                self.current_path_point_target_index += 1
        return self.final_path[self.current_path_point_target_index]

    def communicate(self):
        if isinstance(self.parent, MoveToWayPointPoseController):
            self.uuv_pos = self.parent.parent.parent.pose.position              
            if len(self.final_path) != 0:
                tmp = self.instruct_controller()
                self.parent.target_pose.position = tmp 
                         
    
    @staticmethod
    def reconstruct_path(cameFrom_x, cameFrom_y, cameFrom_z, goal):
        path = [goal]
        cur_node = goal
        while not np.isinf(cameFrom_x[cur_node]):
            cur_node = (int(cameFrom_x[cur_node]), int(cameFrom_y[cur_node]), int(cameFrom_z[cur_node]))
            path.insert(0, cur_node)

        return path

    def a_star_search(self):   
        start = (7, 10, 6)
        goal = (12, 13, 6)
    
        nx, ny, nz = (15, 15, 10)  # number of node in x,y,z directions
        xv, yv, zv = self.fishnet_space_mesh_data[0], self.fishnet_space_mesh_data[1], self.fishnet_space_mesh_data[2]
        # start = self.start_pose.position
        # goal = self.target_pose.position
        is_obs = self.fishnet_obs

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
                # print('find goal')
                return WayPointPlanner.reconstruct_path(cameFrom_x, cameFrom_y, cameFrom_z, goal)

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

    def search_decision(self):
        if len(self.final_path) != 0:
            self.a_star_search_needed = False
        else:
            self.a_star_search_needed = True
    @staticmethod
    def interpolate_fun(path, num):
        node_x,node_y,node_z=[],[],[]
        curve_path = []
        for node in path:
            # print((node))
            node_x.append(node[0])
            node_y.append(node[1])
            node_z.append(node[2])
        node_x=np.array(node_x)
        node_y=np.array(node_y)
        node_z=np.array(node_z)
        
        tck, u = interpolate.splprep([node_x, node_y, node_z], s=2)
        x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
        u_fine = np.linspace(0,1,num)
        x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
        for i in range(len(x_fine)):
            curve_path.append([x_fine[i],y_fine[i], z_fine[i]])
        return curve_path

    def update(self, dt, t):
        super().update(dt, t)
        xv, yv, zv = self.fishnet_space_mesh_data[0], self.fishnet_space_mesh_data[1], self.fishnet_space_mesh_data[2]
        self.search_decision()
        final_path_ini = []
        if self.a_star_search_needed:
            final_path_index = self.a_star_search()
            for node in final_path_index:
                final_path_ini.append([xv[node],yv[node],zv[node]])
            self.final_path = WayPointPlanner.interpolate_fun(final_path_ini, num=30) # number of interpolate number
            print(self.final_path)

        
    def cleanup(self):
        pass


# class HelixPlanner(Actor):
#     def __init__(self, parent: 'Actor', pose: 'Pose'):
#         super().__init__(parent=parent, pose=pose)
#         self.final_path = []
    
#     def update(self, dt, t):
#         return super().update(dt, t)
#         step = np.linspace(0, 10*np.pi, 50) 
#         x = np.sin(step)
#         y = np.cos(step)
#         z = -step
#         self.final_path = np.concatenate([[x],[y],[z]])
#         print(self.final_path)


