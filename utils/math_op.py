import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation as R


def inv(T: np.ndarray):
    """
    Inverse a SE(3) matrix
    """
    assert T.shape == (4, 4), f"Wrong shape: {T.shape}, should be (4, 4)"
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv


def skew(v: np.ndarray):
    """
    Skew-symmetric matrix
    """
    assert v.shape == (3,) or v.shape == (
        3,
        1,
    ), f"Wrong shape: {v.shape}, should be (3,) or (3, 1)"
    v = v.reshape(3, 1)
    return np.array(
        [[0, -v[2, 0], v[1, 0]], [v[2, 0], 0, -v[0, 0]], [-v[1, 0], v[0, 0], 0]]
    )


def _compute_left_Jac(rot_vec: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rot_vec)
    tol = 1e-9
    if theta < tol:
        theta = tol
    n = rot_vec / theta
    n_skew = skew(n)
    return (
        np.eye(3)
        + ((1 - np.cos(theta)) / theta) * n_skew
        + (1 - np.sin(theta) / theta) * n_skew @ n_skew
    )


def _compute_right_Jac(rot_vec: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rot_vec)
    tol = 1e-9
    if theta < tol:
        theta = tol
    n = rot_vec / theta
    n_skew = skew(n)
    return (
        np.eye(3)
        - (theta / 2.0) * n_skew
        + (1.0 - 0.5 * theta / np.tan(theta / 2.0)) * n_skew @ n_skew
    )


def transMat2Vec(T: np.ndarray):
    """
    Convert a 4x4 transformation matrix to a parameter vector.

    Args:
        T (np.ndarray): 4x4 transformation matrix

    Returns:
        np.ndarray: Parameter vector [rx, ry, rz, tx, ty, tz]
    """
    # Extract rotation matrix and convert to rotation vector
    rot_matrix = T[:3, :3]
    rot_vec = R.from_matrix(rot_matrix).as_rotvec()

    # Extrack translation vector
    G_inv = _compute_right_Jac(rot_vec)
    trans_vec = (G_inv @ T[:3, 3].reshape(3, 1)).flatten()

    # Combine into parameter vector
    return np.concatenate([rot_vec, trans_vec])


def vec2transMat(params: np.ndarray):
    """
    Convert a parameter vector to a 4x4 transformation matrix.

    Args:
        params (np.ndarray): Parameter vector [rx, ry, rz, tx, ty, tz]

    Returns:
        np.ndarray: 4x4 transformation matrix
    """
    # Extract rotation vector and convert to rotation matrix
    rot_vec = params[:3]
    rot_matrix = R.from_rotvec(rot_vec).as_matrix()

    # Extract translation vector
    G = _compute_left_Jac(rot_vec)
    trans_vec = G @ params[3:6].reshape(3, 1)

    # Create transformation matrix
    T = np.eye(4)
    T[:3, :3] = rot_matrix
    T[:3, 3] = trans_vec.flatten()

    return T


if __name__ == "__main__":
    N = 100
    for _ in range(N):
        T = np.eye(4)
        T[:3, :3] = R.from_euler("xyz", np.random.uniform(-3, 3, size=(3,))).as_matrix()
        T[:3, 3] = np.random.uniform(-10, 10, size=(3,))
        # print("Original\n", T)

        params = transMat2Vec(T)
        # print("Parameter vector:", params)

        T_reconstructed = vec2transMat(params)
        # print("Reconstructed transformation matrix:\n", T_reconstructed)

        error = np.linalg.norm(inv(T) @ T_reconstructed - np.eye(4))
        if error > 1e-6:
            print(f"Error too high: {error}")
