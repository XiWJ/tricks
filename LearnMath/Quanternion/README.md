# Application of Quaternion in Industry

## 0. Quaternion & Vec3

- quaternion Q

```python
class Q(object):
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.w = w
    
    ...

    def print(self):
        print(self.x, self.y, self.z, self.w)
    
    # operator override
    def __mul__(self, num):
        return Q(self.x * num, self.y * num, self.z * num, self.w * num)

    def __truediv__(self, num):
        assert np.abs(num) > 1e-10
        return Q(self.x / num, self.y / num, self.z / num, self.w / num)
```

- Vec3

```python
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
```

## 1. Definition
$$q = xi + yj + zk + w$$

其中$a, b, c$是虚部，$w$是实部。$i^2=j^2=k^2=ijk=-1$.

```python
class Q(object):
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.w = w
```

## 2. Property

### 2.1 Norm
- 四元数$q=xi+yj+zk+w$的模长：

$$||q|| = \sqrt{x^2 + y^2 + z^2 + w^2}.$$

```python
class Q(object):
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.w = w
    
    def get_norm(self) -> float:
        return np.sqrt(self.x**2 + self.y**2 + self.z**2 + self.w**2)
```

### 2.2 Conjugation of quaternion

- 四元数$q=xi+yj+zk+w$的共轭四元数$q^*$

$$q^* = -xi-yj-zk+w$$

- 共轭四元数满足条件：

$$qq^*=q^*q=||q||^2$$

```python
class Q(object):
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def get_conjugation(self):
        return Q(-self.x, -self.y, -self.z, self.w)
```

### 2.3 Inverse of quaternion

- 四元数$q=xi+yj+zk+w$的逆四元数$q^{-1}$

$$q^{-1}=\frac{q^*}{||q||^2}$$

```python
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
```

## 3. Operation

### 3.1 Make quaternion with axis and angle

- 绕轴$\mathbf{u}$逆时针旋转$\theta$角度的四元数表示：

$$q = \sin{\frac{\theta}{2}}\mathbf{u}.x~i + \sin{\frac{\theta}{2}}\mathbf{u}.y~j + \sin{\frac{\theta}{2}}\mathbf{u}.z~k + \cos{\frac{\theta}{2}}$$

```python
def MakeQuaternion(angle_radian: float, u: Vec3) -> Q:
    q = Q()
    halfAngle = 0.5 * angle_radian
    sinHalf = np.sin(halfAngle)

    q.w = np.cos(halfAngle)
    q.x = sinHalf * u.x
    q.y = sinHalf * u.y
    q.z = sinHalf * u.z

    return q
```

### 3.2 Multiplication with two quaternion
- 满足分配律和结合律，但不满足交换律，$q_Aq_B \neq q_Aq_B$

- $q_Aq_B$:

$$
\begin{aligned}
q_Aq_B &= (q_A.w * q_B.x + q_A.x * q_B.w + q_A.y * q_B.z - q_A.z * q_B.y)i + \\
&= (q_A.w * q_B.y + q_A.y * q_B.w + q_A.z * q_B.x - q_A.x * q_B.z)j + \\
&= (q_A.w * q_B.z + q_A.z * q_B.w + q_A.x * q_B.y - q_A.y * q_B.x)k + \\
&= (q_A.w * q_B.w - q_A.x * q_B.x - q_A.y * q_B.y - q_A.z * q_B.z)
\end{aligned}
$$

```python
def MultQuaternionAndQuaternion(qA: Q, qB: Q) -> Q:
    q = Q()

    q.w = qA.w * qB.w - qA.x * qB.x - qA.y * qB.y - qA.z * qB.z
    q.x = qA.w * qB.x + qA.x * qB.w + qA.y * qB.z - qA.z * qB.y
    q.y = qA.w * qB.y + qA.y * qB.w + qA.z * qB.x - qA.x * qB.z
    q.z = qA.w * qB.z + qA.z * qB.w + qA.x * qB.y - qA.y * qB.x

    return q
```

### 3.3 Multiplication with quaternion and vec3

- 对三维向量$\mathbf{v}$应用四元数$q$：

$$\mathbf{v}' = \mathbf{v} + 2.0 * q.w * q.xyz \times \mathbf{v} + 2.0 * q.xyz \times (q.xyz \times \mathbf{v})$$

```python
def MultQuaternionAndVector(q: Q, v: Vec3) -> Vec3:
    uv, uuv = Vec3(), Vec3()
    qvec = Vec3(q.x, q.y, q.z)
    uv.from_array(np.cross(qvec.to_array(), v.to_array()))
    uuv.from_array(np.cross(qvec.to_array(), uv.to_array()) * 2.0)
    uv *= (2.0 * q.w)

    return v + uv + uuv
```

### 3.4 Normalize quaternion

- 四元数$q=xi+yj+zk+w$的归一化结果为：

$$\hat{q} = \frac{q}{||q||}$$

```python
def NormalizeQuaternion(q: Q) -> Q:
    qq = Q(q.x, q.y, q.z, q.w)
    lengthSqr = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w

    if lengthSqr < 1e-10:
        qq.w = 1.0
        return qq
    
    qq /= np.sqrt(lengthSqr)

    return qq
```

### 3.5 Quaternion from u to v

- 向量$\mathbf{u}$到向量$\mathbf{v}$的四元数$q$为：

$$q = Normalize(cross(u, v), 1.0 + dot(u, v))$$

```python
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
```