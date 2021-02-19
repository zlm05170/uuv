from actors import *
from animation_3d import *
import matplotlib.pyplot as plt
import numpy as np
import json, os
from typing import List

if __name__ == '__main__':
    #%%
    # Define simulation parameter
    t = 0.0
    fps = 60.0 # frame per second
    dt = 1.0/fps
    frame_count = 0
    stop_time = 5.0
    simulation_is_running = True

    #%%
    # Define objects in the scene
    scene = {}
    # Main dimensions of UUV
    # http://www.ijsimm.com/Full_Papers/Fulltext2006/text5-3_114-125.pdf
    # Reference point of the UUV is at aft center
    uuv_length = 5
    uuv_radius = 0.5
    scene.update(uuv = RigidBody(mass = 113.2, moment_of_inertia=[[6100, 0,0,], [0,5970, 0], [0,0,9590]]))
    scene.update(uuv_hydro = HydrodynamicResponseActor(
        parent = scene['uuv'],
        damping=[
            [252.98, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1029.51, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1029.51, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 97.78, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 142.22, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 71.11]],
        added_mass=[
            [0.6, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 107.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 107.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0023, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 6.23, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 6.23]]
    ))
    scene.update(aft_thruster = Thruster(parent = scene['uuv'], pose=Pose(position=[0,0,0])))
    scene.update(aft_thruster_controller = MoveToWayPointController(parent = scene['aft_thruster'], pose=Pose(position=[0,0,0])))
    pose_series = []
    #%%
    # Simulation start
    while simulation_is_running:
        ## Record phase
        pose_series.append(scene['uuv'].pose.copy())
        frame_count += 1
        ## Signal passing phase
        # Apply Thruster to RigidBody
        for key, scene_object in scene.items():
            if isinstance(scene_object, Thruster):
                if isinstance(scene_object.parent, RigidBody):
                    scene_object.parent.add_force_torque_model_frame(scene_object.get_force_torque(), scene_object.pose.position)

        # Apply HydrodynamicResponseActor to RigidBody
        for key, scene_object in scene.items():
            if isinstance(scene_object, HydrodynamicResponseActor):
                if isinstance(scene_object.parent, RigidBody):
                    tvel, rvel = scene_object.parent.get_velocity_model_frame()
                    scene_object.parent.add_force_torque_model_frame(
                        scene_object.get_force_torque(
                            scene_object.parent.pose, tvel, rvel), scene_object.pose.position)
                    scene_object.parent.set_added_mass_6x6(scene_object.added_mass)

        for key, scene_object in scene.items():
            if isinstance(scene_object, MoveToWayPointController):
                if isinstance(scene_object.parent, Thruster):
                    scene_object.parent.thrust = scene_object.custom_thrust
                    scene_object.parent.torque = scene_object.custom_torque
                    scene_object.parent.normal = scene_object.custom_normal

        ## Step phase
        for key, scene_object in scene.items():
            if isinstance(scene_object, Actor):
                scene_object.update(dt, t)

        # Cleanup
        for key, scene_object in scene.items():
            if isinstance(scene_object, Actor):
                scene_object.cleanup()
        
        t += dt
        if  (t > stop_time):
            simulation_is_running = False
    uuv_box = get_box(uuv_length, uuv_radius, uuv_radius, 5)
    animate_motion(pose_series, uuv_box, 10, 10, 10, dt)
