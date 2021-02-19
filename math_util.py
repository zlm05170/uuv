import numpy as np
import math

class Quaternion:
    
    def __init__(self, w = 1.0, x = 0.0, y = 0.0, z = 0.0):
        self.__data = np.array([w, x, y, z])

    @classmethod
    def make_from_w_xyz(cls, w, xyz):
        return Quaternion(w, xyz[0], xyz[1], xyz[2])

    @classmethod
    def make_from_wxyz(cls, wxyz):
        return Quaternion(wxyz[0], wxyz[1], wxyz[2], wxyz[3])

    @classmethod
    def make_from_exp(cls, eta):
        eta_norm = np.linalg.norm(eta)
        w = math.cos(eta_norm)
        xyz_norm = math.sin(eta_norm)
        xyz_norm_scaled = eta * xyz_norm
        x = xyz_norm_scaled[0] / eta_norm if xyz_norm_scaled[0] != 0.0 else 0.0
        y = xyz_norm_scaled[1] / eta_norm if xyz_norm_scaled[1] != 0.0 else 0.0
        z = xyz_norm_scaled[2] / eta_norm if xyz_norm_scaled[2] != 0.0 else 0.0
        return Quaternion(w, x, y, z)
    
    @classmethod
    def make_identity(cls):
        return Quaternion()

    def w(self):
        return self.__data[0]
    
    def x(self):
        return self.__data[1]
    
    def y(self):
        return self.__data[2]
    
    def z(self):
        return self.__data[3]

    def set_w(self, w):
        self.__data[0] = w
    
    def set_x(self, x):
        self.__data[1] = x
    
    def set_y(self, y):
        self.__data[2] = y
    
    def set_z(self, z):
        self.__data[3] = z

    def xyz(self):
        return self.__data[1:4]

    def __str__(self):
        return str(self.__data)

    def __add__(self, q : 'Quaternion'):
        return Quaternion(self.w() + q.w(), self.x() + q.x(), self.y() + q.y(), self.z() + q.z())

    def __mul__(self, s):
        return Quaternion(self.__data[0] * s, self.__data[1] * s, self.__data[2] * s, self.__data[3] * s)

    __rmul__ = __mul__

    def __pow__(self, q : 'Quaternion'):
        w = self.w() * q.w() - np.dot(self.xyz(), q.xyz())
        xyz = np.cross(self.xyz(), q.xyz()) + self.w()*q.xyz() + q.w()*self.xyz()
        return Quaternion.make_from_w_xyz(w, xyz)
    
    def __invert__(self):
        return Quaternion(self.w(), -self.x(), -self.y(), -self.z())
    
    def log(self):
        sin_theta_2 = (1 - self.w()**2)**0.5
        theta_2 = math.atan2(self.w(), sin_theta_2)
        n = self.xyz() / sin_theta_2
        return theta_2 * n

    def rotate(self, vec):
        return vec + 2 * np.cross(self.xyz(), np.cross(self.xyz(), vec) + self.w() * vec)

    def get_matrix(self):
        tx = 2*self.x()
        ty = 2*self.y()
        tz = 2*self.z()
        twx = tx*self.w()
        twy = ty*self.w()
        twz = tz*self.w()
        txx = tx*self.x()
        txy = tx*self.y()
        txz = tx*self.z()
        tyy = ty*self.y()
        tyz = ty*self.z()
        tzz = tz*self.z()

        return np.array([
            [1 - tyy - tzz, txy + twz, txz - twy],
            [txy - twz, 1 - txx - tzz, tyz + twx],
            [txz + twy, tyz - twx, 1 - txx - tyy]
        ])
    
    def get_euler(self):
        euler = np.zeros(3)
        x = self.x()
        y = self.y()
        z = self.z()
        w = self.w()
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        euler[0] = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        if (abs(sinp) >= 1.0):
            euler[1] = math.copysign(np.pi / 2, sinp)
        else:
            euler[1] = math.asin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        euler[2] = math.atan2(siny_cosp, cosy_cosp)
        return euler

class DualQuaternion():

    def __init__(self, a : Quaternion = Quaternion.make_identity(), b : Quaternion = Quaternion(0,0,0,0)):
        self.__data = [a, b]
    
    def __str__(self):
        return self.a().__str__() + ',' + self.b().__str__()
        
    def a(self):
        return self.__data[0]
    
    def b(self):
        return self.__data[1]
    
    def set_a(self, a : Quaternion):
        self.__data[0] = a

    def set_b(self, b : Quaternion):
        self.__data[1] = b
    
    @classmethod
    def make_from_array(cls, arr):
        return DualQuaternion(Quaternion.make_from_wxyz(arr[0:4]), Quaternion.make_from_wxyz(arr[4:8]))
    
    @classmethod
    def make_identity(cls):
        return DualQuaternion()
    
    def __invert__(self):
        return DualQuaternion(~self.a(), ~self.b())

    def __add__(self, p):
        return DualQuaternion(self.a() + p.a(), self.b() + p.b())
    
    def __mul__(self, s):
        return DualQuaternion(self.a() * s, self.b() * s)

    __rmul__ = __mul__

    def __pow__(self, p):
        return DualQuaternion(self.a()**p.a(), self.a()**p.b() + self.b()**p.a())

    def dual_transpose(self):
        return DualQuaternion(self.b(), self.a())

    # def get_translation(self) -> np.ndarray:
    #     return (self.b() ** ~self.a()).xyz() * 2
    
    # def get_rotation(self) -> Quaternion:
    #     return self.a()

    # def transform(self, vec) -> np.ndarray:
    #     return self.get_rotation().rotate(vec) + self.get_translation

class Pose():
    
    def __init__(self, position = np.zeros(3), rotation : Quaternion = Quaternion()):
        super().__init__()
        self.position = np.array(position)
        self.rotation = Quaternion(w=rotation.w(), x=rotation.x(), y=rotation.y(), z=rotation.z())

    def copy(self):
        return Pose(self.position, self.rotation)
        
    def __pow__(self, p : 'Pose'):
        return Pose(self.position + self.rotation.rotate(p.position), self.rotation ** p.rotation)
    
    def __invert__(self):
        return Pose((~self.rotation).rotate(-self.position), ~self.rotation)

    def transform(self, vec):
        return self.rotation.rotate(vec) + self.position
    
    def to_dual_quaternion(self):
        return DualQuaternion(self.rotation, 0.5 * Quaternion(0, self.position[0], self.position[1], self.position[2]) ** self.rotation)

    @classmethod
    def make_from_dual_quaternion(cls, dq : 'DualQuaternion'):
        return Pose(dq.get_translation(), dq.get_rotation())


class ForceTorque():
    def __init__(self, force = [0.0,0.0,0.0], torque = [0.0,0.0,0.0]):
        self.force = np.array(force)
        self.torque = np.array(torque) 
