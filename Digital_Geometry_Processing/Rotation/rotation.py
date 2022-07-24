
import numpy as np

def rotate_with_aligned_axis(alpha, beta, gamma):
    alpha_rad = np.deg2rad(alpha)
    beta_rad = np.deg2rad(beta)
    gamma_rad = np.deg2rad(gamma)

    R_x = np.eye(4, dtype=np.float32)
    R_y = np.eye(4, dtype=np.float32)
    R_z = np.eye(4, dtype=np.float32)

    R_x[1, 1] = np.cos(alpha_rad); R_x[1, 2] = - np.sin(alpha_rad)
    R_x[2, 1] = np.sin(alpha_rad); R_x[2, 2] = np.cos(alpha_rad)

    R_y[0, 0] = np.cos(beta_rad); R_y[0, 2] = np.sin(beta_rad)
    R_y[2, 0] = - np.sin(beta_rad); R_y[2, 2] = np.cos(beta_rad)

    R_z[0, 0] = np.cos(gamma_rad); R_z[0, 2] = - np.sin(gamma_rad)
    R_z[1, 0] = np.sin(gamma_rad); R_z[1, 1] = np.cos(gamma_rad)

    return np.matmul(R_x, np.matmul(R_y, R_z))


def rotate_with_arbitrary_axis(axis, theta):
    theta_rad = np.deg2rad(theta)
    axis = axis / np.linalg.norm(axis)
    axis = np.reshape(axis, (3, 1))

    N = np.zeros(shape=(3, 3), dtype=np.float32)
    N[0, 1] = - axis[2]; N[0, 2] = axis[1]
    N[1, 0] = axis[2]; N[1, 2] = - axis[0]
    N[2, 0] = - axis[1]; N[2, 1] = axis[0]

    R = np.cos(theta_rad) * np.eye(3, dtype=np.float32) + (1 - np.cos(theta_rad)) * np.matmul(axis, axis.T) + np.sin(theta_rad) * N

    return R



def rotate_vFrom_to_vTo(vFrom, vTo):
    vFrom = vFrom / np.linalg.norm(vFrom)
    vTo = vTo / np.linalg.norm(vTo)

    c = np.dot(vFrom, vTo)
    R = np.eye(3, dtype=np.float32)

    if np.abs(c - 1.0) < 1e-4 or np.abs(c + 1.0) < 1e-4:
        return R
    
    v = np.cross(vFrom, vTo)
    v_x = np.array([
        [0., -v[2], v[1]],
        [v[2], 0., -v[0]],
        [-v[1], v[0], 0.]
    ], dtype=np.float32)

    R += v_x + v_x @ v_x / (1 + c)

    return R

def compute_transformation(X, Y):
    """
    X -- [N, 3]
    Y -- [N, 3]
    return R|t
    """
    X_mean = np.mean(X, axis=0)
    Y_mean = np.mean(Y, axis=0)

    X_hat = X - X_mean
    Y_hat = Y - Y_mean
    s = np.sum(np.linalg.norm(Y_hat, axis=1), axis=0) / np.sum(np.linalg.norm(X_hat, axis=1), axis=0)

    A = s * X_hat.T @ Y_hat
    U, _, Vh = np.linalg.svd(A, full_matrices=True)

    R = Vh.T @ U.T

    t = Y_mean - s * R @ X_mean

    return s, R, t


def rotate_manager():
    y = np.array([0, 1, 0, 0], np.float32)
    R1 = rotate_with_aligned_axis(0, 90, 0)
    R2 = rotate_with_arbitrary_axis(y[:3], -90)
    R3 = rotate_vFrom_to_vTo(np.array([1, 0, 0]), np.array([0, 0, 1]))

    X = np.random.random(size=(4, 3))
    Y = 100 * X @ R2.T + 0.5
    s, R, t = compute_transformation(X, Y)
    print(s, R, t)

if __name__ == "__main__":
    rotate_manager()