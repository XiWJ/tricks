from tkinter.messagebox import NO
import numpy as np

class Vec3(object):
    def __init__(self, x=0.0, y=0.0, z=0.0) -> None:
        self.x = x
        self.y = y
        self.z = z

    def from_array(self, array):
        self.x = array[0]
        self.y = array[1]
        self.z = array[2]
    
    def to_array(self):
        return np.array([self.x, self.y, self.z], dtype=np.float32)
    
    def print(self):
        print(self.x, self.y, self.z)
    
    # operator override
    def __add__(self, vec3):
        return Vec3(self.x + vec3.x, self.y + vec3.y, self.z + vec3.z)

    def __sub__(self, vec3):
        return Vec3(self.x - vec3.x, self.y - vec3.y, self.z - vec3.z)

    def __mul__(self, num):
        return Vec3(self.x * num, self.y * num, self.z * num)

    def __truediv(self, num):
        assert np.abs(num) > 1e-10
        return Vec3(self.x / num, self.y / num, self.z / num)

class Q(object):
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.w = w
    
    def get_norm(self) -> float:
        return np.sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)

    def get_conjugation(self):
        return Q(-self.x, -self.y, -self.z, self.w)

    def get_inverse(self):
        q_conjugation = self.get_conjugation()
        q_lengthSqr = self.get_norm()**2
        return Q(q_conjugation.x/q_lengthSqr, q_conjugation.y/q_lengthSqr,\
                 q_conjugation.z/q_lengthSqr, q_conjugation.w/q_lengthSqr)

    def print(self):
        print(self.x, self.y, self.z, self.w)
    
    # operator override
    def __mul__(self, num):
        return Q(self.x * num, self.y * num, self.z * num, self.w * num)

    def __truediv__(self, num):
        assert np.abs(num) > 1e-10
        return Q(self.x / num, self.y / num, self.z / num, self.w / num)

def MultQuaternionAndQuaternion(qA: Q, qB: Q) -> Q:
    q = Q()

    q.w = qA.w * qB.w - qA.x * qB.x - qA.y * qB.y - qA.z * qB.z
    q.x = qA.w * qB.x + qA.x * qB.w + qA.y * qB.z - qA.z * qB.y
    q.y = qA.w * qB.y + qA.y * qB.w + qA.z * qB.x - qA.x * qB.z
    q.z = qA.w * qB.z + qA.z * qB.w + qA.x * qB.y - qA.y * qB.x

    return q

def MakeQuaternion(angle_radian: float, u: Vec3) -> Q:
    q = Q()
    halfAngle = 0.5 * angle_radian
    sinHalf = np.sin(halfAngle)

    q.w = np.cos(halfAngle)
    q.x = sinHalf * u.x
    q.y = sinHalf * u.y
    q.z = sinHalf * u.z

    return q

def MultQuaternionAndVector(q: Q, v: Vec3) -> Vec3:
    uv, uuv = Vec3(), Vec3()
    qvec = Vec3(q.x, q.y, q.z)
    uv.from_array(np.cross(qvec.to_array(), v.to_array()))
    uuv.from_array(np.cross(qvec.to_array(), uv.to_array()) * 2.0)
    uv *= (2.0 * q.w)

    return v + uv + uuv

def NormalizeQuaternion(q: Q) -> Q:
    qq = Q(q.x, q.y, q.z, q.w)
    lengthSqr = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w

    if lengthSqr < 1e-10:
        qq.w = 1.0
        return qq
    
    qq /= np.sqrt(lengthSqr)

    return qq

def QuatFromTwoUnitVectors(u: Vec3, v: Vec3) -> Q:
    r = 1.0 + np.dot(u.to_array(), v.to_array())
    n = Vec3()

    # if u and v are parallel
    if r < 1e-7:
        r = 0.0
        n = Vec3(-u.y, u.x, 0.0) if np.abs(u.x) > np.abs(u.z) else Vec3(0.0, -u.z, u.y)
    else:
        n.from_array(np.cross(u.to_array(), v.to_array()))

    q = Q(n.x, n.y, n.z, r)
    return NormalizeQuaternion(q)

if __name__ == '__main__':
    v = Vec3(1.0, 0.0, 0.0)
    u = Vec3(0.0, 1.0, 0.0)
    w = Vec3(0.0, 0.0, 1.0)

    q_v2u = QuatFromTwoUnitVectors(v, u)
    u2 = MultQuaternionAndVector(q_v2u, v)
    (u - u2).print() # computation error

    q_u2v = MakeQuaternion(np.deg2rad(-90), w)
    v2 = MultQuaternionAndVector(q_u2v, u)
    (v - v2).print()
    
    MultQuaternionAndQuaternion(q_u2v, q_v2u).print()
    q_v2u_2 = q_u2v.get_inverse()
    u3 = MultQuaternionAndVector(q_v2u_2, v)
    (u - u3).print()