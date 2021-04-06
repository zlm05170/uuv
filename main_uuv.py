from fishnet import Fishnet
from actors import *
from math_util import *
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
    stop_time = 0.01
    simulation_is_running = True

    #%%
    # Define objects in the scene
    scene = {}
    # Main dimensions of UUV
    # http://www.ijsimm.com/Full_Papers/Fulltext2006/text5-3_114-125.pdf
    # Reference point of the UUV is at aft center
    uuv_length = 15.0
    uuv_radius = 2.0
    scene.update(uuv = RigidBody(mass = 113.2, moment_of_inertia=[[6100, 0,0,], [0,5970, 0], [0,0,9590]]))
    scene.update(uuv_hydro = HydrodynamicResponseActor(
        parent = scene['uuv'],
        damping=[
            [252.98, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1029.51, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1029.51, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 970.78, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1420.22, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 710.11]],
        added_mass=[
            [0.6, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 107.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 107.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0023, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 6.23, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 6.23]]
    ))
    scene.update(abstract_thruster = AbstractThruster(parent=scene['uuv']))
    scene.update(direct_controller = MoveToWayPointPoseController(parent=scene['abstract_thruster']))
    scene['direct_controller'].target_pose = Pose(position=[15,15,0], rotation=Quaternion.make_from_euler(pitch = math.radians(15), yaw=math.radians(45)))
    scene['direct_controller'].pid_position = np.array([1000,0,100])
    scene['direct_controller'].pid_rotation = np.array([10000,0,15000])

    scene.update(fishnet = Fishnet(net_radius = 30, net_height = 30))
    scene['fishnet'].load_net_mesh('/home/luman/ws/uuv_planning/uuv/fish_net_10_5_0.5.xlsx')
    scene['fishnet'].update_obstacle_points(1)

    scene.update(planner = WayPointPlanner(parent = scene['direct_controller']))
    scene['planner'].target_pose = Pose(position = [12,13,6], rotation = Quaternion.make_from_euler(pitch = math.radians(0), yaw=math.radians(0))) 
    scene['planner'].set_fishnet(scene['fishnet'])

    pose_series = [] 
    fishnet_pose_series = [] 
    waypoint_series = []
    #%%
    # Simulation start
    while simulation_is_running:
        ## Record phase
        pose_series.append(scene['uuv'].pose.copy())
        fishnet_pose_series.append(scene['fishnet'].pose.copy())
        
        frame_count += 1
        
        ## Communicate
        for key, scene_object in scene.items():
            if isinstance(scene_object, Actor):
                scene_object.communicate()
        
        ## Update
        for key, scene_object in scene.items():
            if isinstance(scene_object, Actor):
                scene_object.update(dt, t)

        # Cleanup
        for key, scene_object in scene.items():
            if isinstance(scene_object, Actor):
                scene_object.cleanup()

        waypoint_series.append(scene['planner'].final_path)
        t += dt
        if  (t > stop_time):
            simulation_is_running = False

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    uuv_box = get_box(uuv_length, uuv_radius, uuv_radius, 5)
    uuv_anime = animate_motion(fig, ax, pose_series, uuv_box, 80, 80, 80, dt)
    fishnet_surf = draw_static_surf(fig, ax, scene['fishnet'].net_mesh, scene['fishnet'].net_tri)
    fishnet_obs = draw_obs_point(fig, ax, scene['fishnet'].space_mesh_data, scene['fishnet'].net_obstacle_points)
    find_path = draw_find_path(fig, ax, scene['fishnet'].space_mesh_data, waypoint_series)

    ax.set_xlim(-scene['fishnet'].scenario_len/2, scene['fishnet'].scenario_len/2)
    ax.set_ylim(-scene['fishnet'].scenario_wid/2, scene['fishnet'].scenario_wid/2)
    ax.set_zlim(-scene['fishnet'].scenario_height, 2)
    plt.show()
    
    
