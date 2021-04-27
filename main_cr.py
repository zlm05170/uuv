from asv_cr import SurfaceVessel_CR
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
    stop_time = 10
    simulation_is_running = True

    #%%
    # Define objects in the scene
    scene = {}
    scene.update(ship_a = SurfaceVessel_CR(pose=Pose(position=[0.0,0.0,0.0]),ship_length=10, speed = 10.0, scene=scene))
    scene.update(ship_b = SurfaceVessel_CR(pose=Pose(position=[1.5,0.5,300.0]),ship_length=20, speed = 15.0, scene=scene))
    scene.update(ship_c = SurfaceVessel_CR(pose=Pose(position=[-1.5,0.5,100.0]),ship_length=30,speed= 20,scene=scene))
    pose_series_a = [] 
    pose_series_b = []
    pose_series_c = []
    ship_ab_cr_series = []
    ship_ac_cr_series = []
    #%%
    # Simulation start
    while simulation_is_running:
        ## Record phase
        pose_series_a.append(scene['ship_a'].pose.position[0:2])
        pose_series_a.append(scene['ship_b'].pose.position[0:2])
        pose_series_a.append(scene['ship_c'].pose.position[0:2])
        frame_count += 1
        
        ## Communicate: for a,b,c -> a(detect bc); b(detect ac); c(detect ab)
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
       
        ab_cr = scene['ship_a'].aware_cr_dict['ship_b']
        ac_cr = scene['ship_a'].aware_cr_dict['ship_c']
        if ab_cr is None:
            ab_cr = 0
        ship_ab_cr_series.append(ab_cr)
        ship_ac_cr_series.append(ac_cr)

        t += dt
        if  (t > stop_time):
            simulation_is_running = False
    fig, (ax1,ax2) = plt.subplots(1,2,sharey=True)
    print(ship_ab_cr_series)
    ax1.plot(np.arange(0,stop_time,dt), ship_ab_cr_series)
    ax2.plot(np.arange(0,stop_time,dt), ship_ac_cr_series)
    plt.show()


    
    
