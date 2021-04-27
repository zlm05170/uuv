import numpy as np
from actors import Actor
from math_util import Pose
from math import cos, sin, tan, atan2, asin, pi

class SurfaceVessel_VO(Actor):
    def __init__(self, parent: 'Actor' = None, pose: 'Pose' = Pose(), ship_length = 1, speed =1,
                goal = np.zeros(2), scene = None):
        super().__init__ (parent, pose)
        self.v_max = speed
        self.goal = goal
        self.ship_length = ship_length
        self.v_des = np.zeros(2)
        self.v = np.zeros(2)
        self.scene_ref: Dict = scene 
        self.aware_dict = {}

        self.ws_model = {}
        self.ws_model.update(length_sum = 20)
        self.ws_model.update(obstacle = [])
        self.ws_model.update(boundary = [])

    
    def get_distance(self, target_ship:'SurfaceVessel_VO'):
        position_os_ts = target_ship.pose.position[:2] - self.pose.position[:2]
        return np.linalg.norm(position_os_ts)

    def compute_v_des(self):
        dis_ship_goal = np.linalg.norm(self.goal-self.pose.position[:2])
        v = self.v_max * (self.goal - self.pose.position[:2])/dis_ship_goal
        if dis_ship_goal < 0.5:
            v = [0, 0]
        return v

    def communicate(self): # it has two mission: calculate the desired velocity; sesearch the ts
        super().communicate()
        self.aware_dict.clear()
        for key, scene_object in self.scene_ref.items():
            if isinstance(scene_object, SurfaceVessel_VO):
                if scene_object is not self:
                    dis = self.get_distance(scene_object)
                    if dis < 500:
                        self.aware_dict[key] = scene_object
                        
    def RVO_update(self, target_ships):
        RVO_os_ts_ls = []
        for key, target_ship in target_ships.items():
            # if isinstance(target_ship, SurfaceVessel_VO):
            p_a = self.pose.position[:2]
            p_b = target_ship.pose.position[:2]
            v_a = self.v # current velocity
            v_b = target_ship.v # current velocity
            transl_os_ts = [p_a[0] + 0.5*(v_b[0]+v_a[0]), p_a[1] + 0.5*(v_b[1]+v_a[1])]
            distance_os_ts = np.linalg.norm(p_a - p_b)
            angle_os_ts = atan2(p_b[1]- p_a[1], p_b[0]-p_a[0])
            if distance_os_ts < 0.5* (self.ship_length + target_ship.ship_length):
                distance_os_ts = 0.5* (self.ship_length + target_ship.ship_length)
            angle_os_ts_ort = asin(0.5* (self.ship_length + target_ship.ship_length)/distance_os_ts)

            angel_ort_left = angle_os_ts  + angle_os_ts_ort 
            angel_ort_right = angle_os_ts - angle_os_ts_ort
            angel_ort_left_ = [cos(angel_ort_left), sin(angel_ort_left)]
            angel_ort_right_ = [cos(angel_ort_right), sin(angel_ort_right)]

            RVO_os_ts = [transl_os_ts, angel_ort_left_, angel_ort_right_, distance_os_ts, 0.5* (self.ship_length + target_ship.ship_length)]
            RVO_os_ts_ls.append(RVO_os_ts)
        v_a_post = SurfaceVessel_VO.RVO_intersect(p_a, self.v_des, RVO_os_ts_ls)
        return v_a_post

    @staticmethod
    def RVO_intersect(p_a, v_des, RVO_os_ts_ls):
        v_des_norm = np.linalg.norm(v_des)
        accessible_v = []
        inaccessible_v = []

        ### First, check the angel heading to the goal
        is_assessible = True
        for RVO_os_ts in RVO_os_ts_ls:
            diff = [v_des[0] + p_a[0] - RVO_os_ts[0][0], v_des[1]+p_a[1]-RVO_os_ts[0][1]]
            angel_diff = atan2(diff[1], diff[0])
            angel_left = atan2(RVO_os_ts[1][1], RVO_os_ts[1][0])
            angel_right = atan2(RVO_os_ts[2][1], RVO_os_ts[2][0])
            if SurfaceVessel_VO.RVO_inside(angel_diff,angel_left,angel_right):
                is_assessible = False
                break            
        if  is_assessible:   
            accessible_v.append(v_des)
        else:
            inaccessible_v.append(v_des)   
        ###
        ### Second, check the other available zone    
        for i in np.arange(0, 2*pi, 0.1):
            for j in np.arange(1.0, v_des_norm+v_des_norm/5, v_des_norm/5):
                temp_v = [j*cos(i), j*sin(i)]                
                is_assessible = True
                for RVO_os_ts in RVO_os_ts_ls:
                    diff = [temp_v[0] + p_a[0] -RVO_os_ts[0][0], temp_v[1]+p_a[1]-RVO_os_ts[0][1]]
                    angel_diff = atan2(diff[1], diff[0])
                    angel_left = atan2(RVO_os_ts[1][1], RVO_os_ts[1][0])
                    angel_right = atan2(RVO_os_ts[2][1], RVO_os_ts[2][0])
                    if SurfaceVessel_VO.RVO_inside(angel_diff,angel_left,angel_right):
                        is_assessible = False
                        break
                if is_assessible:
                    accessible_v.append(temp_v) 
                else:
                    inaccessible_v.append(temp_v)
        ###
        if accessible_v:
            v_a = min(accessible_v, key= lambda v: np.linalg.norm(v - v_des))
        # elif inaccessible_v:
        #     tc_v = dict()
        #     for inacc_v in inaccessible_v:
        #         tc_v[tuple(v)] = 0
        #         tc = []
        #         for RVO_os_ts in RVO_os_ts_ls:
        #             diff = [inacc_v[0] + p_a[0] -RVO_os_ts[0][0], inacc_v[1]+p_a[1]-RVO_os_ts[0][1]]
        #             angel_diff = atan2(diff[1], diff[0])
        #             angel_left = atan2(RVO_os_ts[1][1], RVO_os_ts[1][0])
        #             angel_right = atan2(RVO_os_ts[2][1], RVO_os_ts[2][0])
        #             if SurfaceVessel_VO.RVO_inside(angel_diff,angel_left,angel_right):
        #                 smal_angle = abs(angel_diff - 0.5*(angel_left+angel_right))
        #                 if abs(RVO_os_ts[3]* sin(smal_angle)) >= RVO_os_ts[4]:
        #                     RVO_os_ts[4] = abs(RVO_os_ts[3]* sin(smal_angle))
        #                 big_angle = asin(abs(RVO_os_ts[3]* sin(smal_angle))/RVO_os_ts[4])
        #                 dist_tg = abs(RVO_os_ts[3]*cos(smal_angle))-abs(RVO_os_ts[4]*cos(big_angle))
        #                 if dist_tg < 0:
        #                     dist_tg = 0                    
        #                 tc_v = dist_tg/np.linalg.norm(diff)
        #                 tc.append(tc_v)
        #         tc_v[tuple(inacc_v)] =min(tc)
        #     wt = 0.2
        #     v_a = min(inaccessible_v, key=lambda v: ((wt/tc_v[tuple(v)+np.linalg.norm(v-v_des)]))
        return v_a

    @staticmethod
    def RVO_inside(angel_diff,angel_left,angel_right):
        if abs(angel_right - angel_left) <= pi:
            if angel_right <= angel_diff <= angel_left:
                return True
            else:
                return False
        else:
            if (angel_left <0) and (angel_right >0):
                angel_left += 2*pi
                if angel_diff < 0:
                    angel_diff += 2*pi
                if angel_right <= angel_diff <= angel_left:
                    return True
                else:
                    return False
            if (angel_left >0) and (angel_right <0):
                angel_right += 2*pi
                if angel_diff < 0:
                    angel_diff += 2*pi
                if angel_left <= angel_diff <= angel_right:
                    return True
                else:
                    return False

    def update(self, dt, t):
        super().update(dt, t)
        dis_ship_goal = np.linalg.norm(self.goal-self.pose.position[:2])
        if dis_ship_goal < 0.5:
            pass
        else:
            self.v_des = self.compute_v_des()
            self.v = self.RVO_update(self.aware_dict)              
            self.pose.position[0:2] += np.array(self.v) * dt


    
