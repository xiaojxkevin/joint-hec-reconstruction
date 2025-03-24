import numpy as np
from scipy.spatial.transform import Rotation
from scipy.linalg import svd, lstsq, solve
from utils.math_op import skew, inv, minimize_L1_norm


def compute_As_Bs(eye2obj_poses: np.ndarray, hand2base_poses: np.ndarray):
    """
    Returns: As [N, 4, 4] (relative hand motions), Bs [N, 4, 4] (relative eye motions)
    """
    As, Bs = [], []
    for i in range(1, hand2base_poses.shape[0]):
        eye_pair = eye2obj_poses[i - 1], eye2obj_poses[i]
        hand_pair = hand2base_poses[i - 1], hand2base_poses[i]
        As.append(inv(hand_pair[0]) @ hand_pair[1])
        Bs.append(inv(eye_pair[0]) @ eye_pair[1])
    As = np.asarray(As, dtype=np.float64)
    Bs = np.asarray(Bs, dtype=np.float64)
    return As, Bs


def solve_hand_eye_se3(As: np.ndarray, Bs: np.ndarray):
    """
    Input:
        As: An array of shape (n, 4, 4) containing SE(3) matrices, where each element is a homogeneous transformation A_i.
        Bs: An array of shape (n, 4, 4) containing SE(3) matrices, where each element is a homogeneous transformation B_i.
    Returns:
        R_X: A 3x3 rotation matrix.
        t_X: A 3x1 translation vector.
        lambda_: A scale factor.
    """
    n = As.shape[0]
    assert n > 0, "Number of SE(3) matrices should be greater than 0."
    # Step 1: Compute the rotation component R_X.
    M = np.zeros((3, 3), dtype=np.float64)
    for i in range(n):
        # Extract the rotation matrices from A_i and B_i.
        RA_i = As[i, :3, :3]
        RB_i = Bs[i, :3, :3]

        # Convert the rotation matrices to rotation vectors (axis-angle representation).
        w_Ai = Rotation.from_matrix(RA_i).as_rotvec()
        w_Bi = Rotation.from_matrix(RB_i).as_rotvec()
        M += np.outer(w_Ai, w_Bi)

    # Solve the orthogonal Procrustes problem using SVD.
    U, _, Vt = svd(M, check_finite=False)
    R_X = U @ Vt  # Ensure the determinant is +1 (if it's a reflection, adjust Vt).
    if np.linalg.det(R_X) < 0:
        assert False, "Reflection detected. Adjusted the sign of the last row of Vt."
        Vt[-1, :] *= -1
        R_X = U @ Vt

    # Step 2: Compute the translation component t_X.
    C = []
    d = []
    for i in range(n):
        # Extract the translation vectors from A_i and B_i.
        tA_i = As[i, :3, 3].reshape(3, 1)
        tB_i = Bs[i, :3, 3].reshape(3, 1)

        # Construct the skew-symmetric matrix for tB_i (denoted as tB_i^).
        tB_skew = skew(tB_i)
        # Compute C_i and d_i.
        RA_i = As[i, :3, :3]
        Ci = tB_skew @ R_X.T @ (np.eye(3, dtype=np.float64) - RA_i)
        di = tB_skew @ R_X.T @ tA_i
        C.append(Ci)
        d.append(di.reshape(-1, 1))  # Ensure the vector is a column vector.

    # Form the linear system: C_stack @ t_X = d_stack.
    C_stack = np.vstack(C)
    d_stack = np.vstack(d)
    t_X, residues, _, s = lstsq(C_stack, d_stack, check_finite=False)
    print("<>" * 20)
    print("The singular values are: ", s)
    print(f"The residues for tx is: {residues}")
    print("<>" * 20)
    # t_X_, _, _ = minimize_L1_norm(C_stack, d_stack)
    # t_X_ = t_X_.reshape(3, 1)
    # print("<>" * 20)
    # print(f"The residues for tx_ is: {np.linalg.norm(C_stack @ t_X_ - d_stack)**2}")
    # print("<>" * 20)

    # Step 3: Compute the scale factor lambda.
    lambda_sum = 0.0
    for i in range(n):
        RA_i = As[i, :3, :3]
        tA_i = As[i, :3, 3].reshape(3, 1)
        tB_i = Bs[i, :3, 3].reshape(3, 1)

        numerator = tB_i.T @ R_X.T @ ((RA_i - np.eye(3)) @ t_X + tA_i)
        denominator = np.linalg.norm(tB_i) ** 2
        assert denominator != 0, "Denominator should not be zero"
        lambda_i = numerator.item() / denominator
        ############################# for debugging #############################
        print("Debuging: lambda_i")
        print(lambda_i)
        ############################# for debugging #############################
        lambda_sum += lambda_i

    lambda_ = lambda_sum / n

    return R_X, t_X.flatten(), lambda_


# Example usage.
if __name__ == "__main__":
    # Generate example SE(4) matrices (identity matrices).
    n = 5
    As = np.tile(np.eye(4), (n, 1, 1))  # Shape: (n, 4, 4)
    Bs = np.tile(np.eye(4), (n, 1, 1))  # Shape: (n, 4, 4)

    R_X, t_X, lambda_ = solve_hand_eye_se3(As, Bs)
    print("Rotation R_X:\n", R_X)
    print("Translation t_X:", t_X)
    print("Scale lambda:", lambda_)
