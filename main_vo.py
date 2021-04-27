from actors import *
from animation_3d import *
import numpy as np
import matplotlib.pyplot as plt
from asv_vo import SurfaceVessel_VO

if __name__ == '__main__':
    # define simulation parameter
    t = 0.0
    fps = 60.0
    dt = 1.0/fps
    frame_count = 0
    stop_time = 30
    simulation_is_running = True

    scene = {}
    scene.update(ship_a = SurfaceVessel_VO(pose=Pose(position=[0.0,0.0,45.0]), ship_length = 10, speed = 5.0, 
                goal = np.array([50.0,50.0]), scene = scene))
    scene.update(ship_b = SurfaceVessel_VO(pose=Pose(position=[50.0,0.0,315.0]), ship_length = 10, speed = 5.0, 
                goal = np.array([0.0,50.0]), scene = scene))
    scene.update(ship_c = SurfaceVessel_VO(pose=Pose(position=[50.0,50.0,225.0]), ship_length = 10, speed = 5.0, 
                goal = np.array([0.0,0.0]), scene = scene))
    scene.update(ship_d = SurfaceVessel_VO(pose=Pose(position=[0.0,50.0,135.0]), ship_length = 10, speed = 5.0, 
                goal = np.array([50.0,0.0]), scene = scene))

    pose_series_a = []
    pose_series_b = []
    pose_series_c = []
    pose_series_d = []
    v = []
    while simulation_is_running:
        frame_count += 1
        pose_series_a.append(list(scene['ship_a'].pose.position[:2]))
        pose_series_b.append(list(scene['ship_b'].pose.position[:2]))
        pose_series_c.append(list(scene['ship_c'].pose.position[:2]))
        pose_series_d.append(list(scene['ship_d'].pose.position[:2]))
        v.append(list(scene['ship_a'].v))

        # print(v)
        for key, scene_object in scene.items():
            if isinstance(scene_object, Actor):
                scene_object.communicate()
        for key, scene_object in scene.items():
            if isinstance(scene_object, Actor):
                scene_object.update(dt,t)
        for key, scene_object in scene.items():
            if isinstance(scene_object, Actor):
                scene_object.cleanup()
       
        t += dt
        if t > stop_time:
            simulation_is_running = False
    
    # fig, axs = plt.subplots(4)
    # tra = draw_vo_trajectory(fig, axs[0], pose_series_a)
    # tra = draw_vo_trajectory(fig, axs[1], pose_series_b)
    # tra = draw_vo_trajectory(fig, axs[2], pose_series_c)
    # tra = draw_vo_trajectory(fig, axs[3], pose_series_d)
    fig,ax = plt.subplots()
    vo_anime = animate_motion_2d(fig, ax, np.array(pose_series_a),np.array(pose_series_b),np.array(pose_series_c),np.array(pose_series_d),50,50,dt)
    # vo_anime = animate_motion_2d_2ship(fig, ax, np.array(pose_series_a),np.array(pose_series_c),50,50,dt)
    plt.show()

        
