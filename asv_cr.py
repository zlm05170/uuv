from typing import Dict, List
from numpy.lib.function_base import select
from actors import Actor
from math_util import Pose
import numpy as np

class SurfaceVessel_CR(Actor):
    def __init__(self, parent : 'Actor' = None, pose : 'Pose' = Pose(), ship_length =1, speed = 1, 
                scene = None):
        super().__init__(parent, pose)
        self.ship_length = ship_length
        self.velocity_xy = SurfaceVessel_CR.get_velocity(speed, self.pose.position[2])
        self.velocity_yaw = 0
        self.scene_ref: Dict = scene 
        self.aware_dict: Dict = {}
        self.aware_cr_dict: Dict = {}

    def get_cpa(self, target_ship: 'SurfaceVessel'):
        position_os_ts = target_ship.pose.position[0:2] - self.pose.position[0:2]
        velocity_os_ts = target_ship.velocity_xy - self.velocity_xy
        TCPA = -np.dot(position_os_ts, velocity_os_ts)/ np.dot(velocity_os_ts, velocity_os_ts)
        DCPA = np.linalg.norm(position_os_ts + TCPA * velocity_os_ts)
        cap = self.pose.position[0:2] + (-TCPA) * self.velocity_xy
        return TCPA, DCPA

    @classmethod
    def get_velocity(cls, speed, heading):
        ship_spd_x = speed * np.sin(np.deg2rad(heading))
        ship_spd_y = speed * np.cos(np.deg2rad(heading))
        return np.array([ship_spd_x, ship_spd_y])

    @classmethod
    def get_colreg_coefficient(cls, bearing):
        d = 0
        b = bearing / 180
        if 0 <= bearing < 112.5:
            d = 1.1 - 0.2 * b 
        if 112.5 <= bearing < 180:
            d = 1.1 - 0.4 * b
        if 180 <= bearing < 247.5:
            d = 1.1 - 0.2 * (2 - b)
        if 247.5 <= bearing < 360:
            d = 1.1 - 0.2 * (2 - b)
        return d

    def get_cr(self, target_ship: 'SurfaceVessel_CR'):
        position_os_ts = target_ship.pose.position[0:2] - self.pose.position[0:2]
        velocity_os_ts = target_ship.velocity_xy - self.velocity_xy
        distance_rel = np.linalg.norm(position_os_ts)
        speed_rel = np.linalg.norm(velocity_os_ts)
        
        angle_os_ts = np.arctan2(position_os_ts[0], position_os_ts[1])
        if angle_os_ts < 0:
            angle_os_ts = angle_os_ts + 2 * np.pi
        relative_bearing = angle_os_ts - np.deg2rad(self.velocity_xy[1])
        bearing = np.rad2deg(relative_bearing)
        
        tcpa, dcpa = self.get_cpa(target_ship)
        u = np.zeros(4)
        w = np.array([0.4, 0.367,0.133, 0.1])

        d1 = SurfaceVessel_CR.get_colreg_coefficient(bearing)
        d2 = d1 * 2
        D1 = 13 * self.ship_length
        temp1 = np.radians(bearing - 19) 
        D2 = 1.7 * np.cos(temp1)+np.sqrt(4.4+2.89*np.square(np.cos(temp1)))

        # u bearing
        u[3] = 0.5 *(np.cos(temp1) + np.sqrt(1.5+np.square(np.cos(temp1))))-5/17

        if d2 < np.fabs(dcpa):
            u[0] = 0
        if  d1 < np.fabs(dcpa) <= d2:
            temp2 = np.pi/(d2-d1)*(np.fabs(dcpa)-(d2+d1)/2) # 0.3938
            u[0] = 0.5-0.5*np.sin(temp2)
        if np.fabs(dcpa) <= d1:
            u[0] = 1

        if dcpa <= D1:
            t1 = np.sqrt(np.square(D1)-np.square(dcpa))/speed_rel
        elif dcpa > D1:
            t1 = (D1-dcpa)/speed_rel
        
        if dcpa <= D2:
            t2 = np.sqrt(np.square(D2)-np.square(dcpa))/speed_rel
        elif dcpa > D2:
            t2 =(D2-dcpa)/speed_rel
        
        if t2 < np.fabs(tcpa):
            u[1] = 0
        if t1 < np.fabs(tcpa) < t2:
            u[1] = np.square((t2-np.fabs(tcpa))/(t2-t1))
        if 0 <= np.fabs(tcpa) <= t1:
            u[1] = 1
        
        if D2 < distance_rel:
            u[2] = 0
        if D1 < distance_rel <= D2:
            u[2] = np.square((D2-distance_rel)/(D2-D1))
        if D1 >= distance_rel:
            u[2] = 1
        return np.dot(w, u)
    
    # @classmethod
    def get_distance(self, target_ship: 'SurfaceVessel_CR'):
        position_os_ts = target_ship.pose.position[0:2] - self.pose.position[0:2]
        return np.linalg.norm(position_os_ts)
    
    # put the risk target vessel into the aware_dict
    def communicate(self):
        super().communicate()
        self.aware_dict.clear()
        self.aware_cr_dict.clear()
        for key, scene_object in self.scene_ref.items():
            if isinstance(scene_object, SurfaceVessel_CR):
                if scene_object is not self:  
                    d = self.get_distance(scene_object)
                    if d < 500:                               
                        self.aware_dict[key] = scene_object

    def update(self, dt, t):        
        super().update(dt, t)
        for key, awared_object in self.aware_dict.items():
            self.aware_cr_dict[key] = self.get_cr(awared_object)
        self.pose.position[0:2] += self.velocity_xy * dt
        a = 0

            
        