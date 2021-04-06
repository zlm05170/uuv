from math_util import *
import numpy as np
from typing import List
import random


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
        self.target_pose = Pose()
        self.uuv = None
        self.fishnet_obs = None

    def get_uuv(self):
        if self.uuv == None:
            self.uuv = self.parent.parent.parent
        return self.uuv

    def set_fishnet(self, fishnet: 'Fishnet'):
        self.fishnet_obs = fishnet.net_obstacle_points

    def communicate(self):
        if isinstance(self.parent, MoveToWayPointPoseController):
            self.uuv_pos = self.parent.parent.parent.pose.position
            print(self.uuv_pos)
            

    def a_star_search(self):
        pass

    def update(self, dt, t):
        # final_path = a_star_search(start_idx, end_idx, nx, ny, nz, xv, yv, zv, is_obs)
        pass

    def cleanup(self):
        pass

